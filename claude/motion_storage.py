#!/usr/bin/python3

"""
Motion Detection Storage and Transfer Module for Raspberry Pi Camera

This module extends the FastAPI MJPEG server with advanced motion detection,
efficient storage management, and optimized network transfer capabilities.
"""

import asyncio
import base64
import cv2
import datetime
import hashlib
import io
import json
import logging
import numpy as np
import os
import queue
import requests
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set

# Helper function to detect Raspberry Pi
def is_raspberry_pi():
    """Detect if running on a Raspberry Pi using multiple methods"""
    # Method 1: Check for Pi-specific file locations
    if os.path.exists("/opt/vc/bin/raspivid") or os.path.exists("/usr/bin/vcgencmd"):
        return True
        
    # Method 2: Check CPU info
    try:
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read()
            if "BCM" in cpuinfo or "Raspberry Pi" in cpuinfo:
                return True
    except Exception:
        pass
        
    # Method 3: Check for Pi-specific environment variables
    if "RASPBERRY_PI" in os.environ:
        return True
        
    # Default to false if no indicators found
    return False

# Constants for WiFi signal strength monitoring
WIFI_AVAILABLE = True  # Will be updated based on platform check
IS_RASPBERRY_PI = is_raspberry_pi()  # Determine if we're running on a Raspberry Pi

# Constants
MIN_UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1MB
MAX_UPLOAD_CHUNK_SIZE = 5 * 1024 * 1024  # 5MB
MAX_RETRIES = 5
WIFI_CHECK_INTERVAL = 30  # seconds

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("motion_storage")

# Motion detection variables - initialized here for module scope
prev_frame = None
motion_detected = False
motion_regions = []

# Global shutdown event for signaling threads to terminate
shutdown_requested = threading.Event()

# Storage configuration
@dataclass
class StorageConfig:
    # RAM Buffer settings
    ram_buffer_seconds: int = 30       # Seconds to keep in RAM before/after motion
    max_ram_segments: int = 300        # Maximum segments in RAM (10 seconds at 30fps)
    
    # Local storage settings
    local_storage_path: str = "/tmp/motion_events"
    max_disk_usage_mb: int = 1000      # 1GB max local storage
    min_motion_duration_sec: int = 3   # Minimum motion duration to save
    
    # Remote storage settings
    remote_storage_url: str = "http://192.168.1.100:8080/storage"
    remote_api_key: str = "your_api_key_here"
    chunk_upload: bool = True          # Enable chunked uploads
    
    # Transfer settings
    upload_throttle_kbps: int = 500    # Throttle uploads to 500 KB/s
    transfer_retry_interval_sec: int = 60
    transfer_schedule_active: bool = True
    transfer_schedule_start: int = 1   # 1 AM 
    transfer_schedule_end: int = 5     # 5 AM
    
    # Thumbnail settings
    generate_thumbnails: bool = True
    thumbnail_width: int = 320
    thumbnail_height: int = 240
    thumbnails_per_event: int = 3      # Number of thumbnails per event
    
    # WiFi monitoring settings
    wifi_monitoring: bool = True
    wifi_adapter: str = "wlan0"
    wifi_signal_threshold_low: int = -75   # dBm
    wifi_signal_threshold_good: int = -65  # dBm
    wifi_throttle_poor: int = 100      # KB/s when signal is poor
    wifi_throttle_medium: int = 300    # KB/s when signal is medium
    wifi_throttle_good: int = 800      # KB/s when signal is good

# Circular buffer for RAM storage
class CircularFrameBuffer:
    """Stores recent frames in RAM using a circular buffer"""
    
    def __init__(self, max_size=300):  # Default: 10 seconds @ 30fps
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        
    def add_frame(self, frame, timestamp):
        with self.lock:
            self.buffer.append((frame, timestamp))
            
    def get_recent_frames(self, seconds):
        with self.lock:
            if not self.buffer:
                return []
                
            cutoff_time = self.buffer[-1][1] - datetime.timedelta(seconds=seconds)
            return [frame for frame, ts in self.buffer if ts >= cutoff_time]

# Motion event recorder
class MotionEventRecorder:
    """Records motion events and saves them to local storage"""
    
    def __init__(self, frame_buffer, config):
        self.frame_buffer = frame_buffer
        self.config = config
        self.recording = False
        self.current_event = None
        self.events_queue = queue.PriorityQueue()
        self.worker_thread = threading.Thread(target=self._process_events)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.config.local_storage_path, exist_ok=True)
        
    def start_recording(self, motion_regions):
        """Start recording a motion event"""
        if self.recording:
            return
            
        self.recording = True
        self.current_event = {
            "start_time": datetime.datetime.now(),
            "frames": [],
            "regions": motion_regions,
            "id": self._generate_event_id()
        }
        
        # Get buffer frames from before motion started
        pre_frames = self.frame_buffer.get_recent_frames(
            self.config.ram_buffer_seconds
        )
        self.current_event["frames"].extend(pre_frames)
        
        logger.info(f"Started recording motion event {self.current_event['id']}")
        
    def add_frame(self, frame, motion_regions):
        """Add a frame to the current motion event"""
        if not self.recording:
            return
            
        self.current_event["frames"].append((frame, datetime.datetime.now()))
        self.current_event["regions"] = motion_regions
        
    def stop_recording(self):
        """Stop recording the current motion event"""
        if not self.recording:
            return
            
        self.recording = False
        self.current_event["end_time"] = datetime.datetime.now()
        
        # Check if event meets minimum duration
        duration = (self.current_event["end_time"] - 
                   self.current_event["start_time"]).total_seconds()
        
        if duration >= self.config.min_motion_duration_sec:
            # Add to processing queue with priority based on size
            priority = len(self.current_event["frames"])
            self.events_queue.put((priority, self.current_event))
            logger.info(f"Stopped recording motion event {self.current_event['id']} "
                       f"({len(self.current_event['frames'])} frames, {duration:.1f}s)")
        else:
            logger.info(f"Discarded motion event {self.current_event['id']} "
                       f"(duration {duration:.1f}s < minimum {self.config.min_motion_duration_sec}s)")
        
        self.current_event = None
        
    def _process_events(self):
        """Worker thread to process recorded events"""
        while not shutdown_requested.is_set():
            try:
                try:
                    # Use a timeout to check shutdown_requested periodically
                    _, event = self.events_queue.get(timeout=1.0)
                    self._save_event_to_disk(event)
                    self.events_queue.task_done()
                except queue.Empty:
                    # No items in queue, just continue and check shutdown flag
                    continue
            except Exception as e:
                logger.error(f"Error processing motion event: {e}")
                time.sleep(1)
        
        logger.info("Motion event processor thread exiting")
        # Process any remaining items in the queue before exiting
        while not self.events_queue.empty():
            try:
                _, event = self.events_queue.get_nowait()
                self._save_event_to_disk(event)
                self.events_queue.task_done()
            except (queue.Empty, Exception):
                break
                
    def _save_event_to_disk(self, event):
        """Save a motion event to disk"""
        # Create directory structure
        event_dir = os.path.join(
            self.config.local_storage_path,
            event["id"]
        )
        os.makedirs(event_dir, exist_ok=True)
        
        try:
            # Use temporary file for video assembly to avoid SD card wear
            with tempfile.NamedTemporaryFile(suffix='.h264') as temp_video:
                # Use hardware encoding to create video
                self._encode_video(event["frames"], temp_video.name)
                
                # Generate thumbnails if enabled
                thumbnails = []
                if self.config.generate_thumbnails:
                    thumbnails = self._generate_thumbnails(event["frames"])
                
                # Save metadata
                metadata = {
                    "id": event["id"],
                    "start_time": event["start_time"].isoformat(),
                    "end_time": event["end_time"].isoformat(),
                    "duration": (event["end_time"] - event["start_time"]).total_seconds(),
                    "regions": event["regions"],
                    "frame_count": len(event["frames"]),
                    "resolution": {
                        "width": event["frames"][0][0].shape[1],
                        "height": event["frames"][0][0].shape[0]
                    },
                    "has_thumbnails": len(thumbnails) > 0,
                    "processed": False
                }
                
                # Write metadata file
                with open(os.path.join(event_dir, "metadata.json"), "w") as f:
                    json.dump(metadata, f, indent=2)
                    
                # Save thumbnails
                if thumbnails:
                    thumbnails_dir = os.path.join(event_dir, "thumbnails")
                    os.makedirs(thumbnails_dir, exist_ok=True)
                    
                    for i, thumbnail in enumerate(thumbnails):
                        thumbnail_path = os.path.join(thumbnails_dir, f"thumb_{i}.jpg")
                        cv2.imwrite(thumbnail_path, thumbnail)
                    
                # Copy video to final location
                final_video_path = os.path.join(event_dir, "motion.mp4")
                shutil.copy(temp_video.name, final_video_path)
                
            # Notify transfer manager
            transfer_manager.add_event(event["id"])
            
        except Exception as e:
            logger.error(f"Error saving event {event['id']} to disk: {e}")
        
    def _encode_video(self, frames, output_path):
        """Encode frames into a video file using hardware acceleration if available"""
        if not frames:
            raise ValueError("No frames to encode")
            
        # Get video dimensions from first frame
        height, width = frames[0][0].shape[:2]
        
        # Try to use hardware encoding if available
        try:
            # Check if we're on Raspberry Pi and use hardware encoding
            if IS_RASPBERRY_PI:
                # Use H.264 hardware encoding via ffmpeg
                fourcc = cv2.VideoWriter_fourcc(*'H264')
                logger.debug("Using hardware H.264 encoding on Raspberry Pi")
            else:
                # Fallback to standard encoding
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                logger.debug("Using standard mp4v encoding (non-Raspberry Pi device)")
                
            # Create video writer
            out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
            
            # Write frames
            for frame, _ in frames:
                out.write(frame)
                
            # Release the writer
            out.release()
            
            # For Raspberry Pi, we might need to convert the raw H.264 to MP4
            if IS_RASPBERRY_PI and output_path.endswith('.h264'):
                logger.debug("Converting raw H.264 to MP4 format")
                mp4_output = output_path.replace('.h264', '.mp4')
                cmd = ["ffmpeg", "-i", output_path, "-c:v", "copy", mp4_output]
                subprocess.run(cmd, check=True)
                
        except Exception as e:
            logger.error(f"Error encoding video: {e}")
            # Fallback to basic encoding if hardware encoding fails
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
            
            for frame, _ in frames:
                out.write(frame)
                
            out.release()
        
    def _generate_thumbnails(self, frames):
        """Generate thumbnails from selected frames in the event"""
        if not frames:
            return []
            
        thumbnails = []
        
        # Get evenly distributed frames
        indices = []
        if len(frames) <= self.config.thumbnails_per_event:
            indices = list(range(len(frames)))
        else:
            step = len(frames) // self.config.thumbnails_per_event
            indices = [i * step for i in range(self.config.thumbnails_per_event)]
            # Make sure we include the frame with the most motion
            # This would require additional logic to detect the frame with most motion
        
        # Generate thumbnails
        for idx in indices:
            if idx < len(frames):
                frame = frames[idx][0]
                # Resize frame to thumbnail size
                thumbnail = cv2.resize(
                    frame, 
                    (self.config.thumbnail_width, self.config.thumbnail_height)
                )
                thumbnails.append(thumbnail)
                
        return thumbnails
        
    def _generate_event_id(self):
        """Generate a unique ID for a motion event"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        random_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
        return f"motion-{timestamp}-{random_suffix}"

# WiFi Signal Monitor
class WiFiMonitor:
    """Monitors WiFi signal strength and adjusts transfer speed accordingly"""
    
    def __init__(self, config):
        self.config = config
        self.current_signal = None
        self.current_throttle = config.upload_throttle_kbps
        
        # Update WiFi availability based on platform
        global WIFI_AVAILABLE
        if not IS_RASPBERRY_PI:
            logger.info("Non-Raspberry Pi platform detected, disabling WiFi monitoring")
            WIFI_AVAILABLE = False
            
        self.enabled = config.wifi_monitoring and WIFI_AVAILABLE
        self.lock = threading.Lock()
        
        if self.enabled:
            # Check if the WiFi adapter exists
            try:
                # First check if iwconfig exists
                if shutil.which("iwconfig") is None:
                    logger.warning("iwconfig command not found, disabling WiFi monitoring")
                    self.enabled = False
                else:
                    # Check if adapter exists
                    subprocess.check_output(["iwconfig", config.wifi_adapter], stderr=subprocess.STDOUT)
                    self.monitor_thread = threading.Thread(target=self._monitor_signal)
                    self.monitor_thread.daemon = True
                    self.monitor_thread.start()
                    logger.info(f"WiFi signal monitoring enabled for adapter {config.wifi_adapter}")
            except (subprocess.SubprocessError, FileNotFoundError):
                logger.warning(f"WiFi adapter {config.wifi_adapter} not found or iwconfig not available")
                self.enabled = False
        else:
            logger.info("WiFi monitoring disabled in configuration or not supported on this platform")
    
    def get_current_throttle(self):
        """Get the current throttle value based on WiFi signal strength"""
        if not self.enabled:
            return self.config.upload_throttle_kbps
            
        with self.lock:
            return self.current_throttle
    
    def _monitor_signal(self):
        """Monitor WiFi signal strength and adjust throttle accordingly"""
        while not shutdown_requested.is_set():
            try:
                signal_strength = self._get_wifi_signal_strength()
                
                with self.lock:
                    self.current_signal = signal_strength
                    
                    # Adjust throttle based on signal strength
                    if signal_strength is not None:
                        if signal_strength <= self.config.wifi_signal_threshold_low:
                            # Poor signal
                            self.current_throttle = self.config.wifi_throttle_poor
                            logger.debug(f"WiFi signal poor: {signal_strength} dBm, "
                                       f"throttling to {self.current_throttle} KB/s")
                        elif signal_strength <= self.config.wifi_signal_threshold_good:
                            # Medium signal
                            self.current_throttle = self.config.wifi_throttle_medium
                            logger.debug(f"WiFi signal medium: {signal_strength} dBm, "
                                       f"throttling to {self.current_throttle} KB/s")
                        else:
                            # Good signal
                            self.current_throttle = self.config.wifi_throttle_good
                            logger.debug(f"WiFi signal good: {signal_strength} dBm, "
                                       f"throttling to {self.current_throttle} KB/s")
                
            except Exception as e:
                logger.error(f"Error monitoring WiFi signal: {e}")
                
            # Wait before checking again - use smaller sleeps to check shutdown flag more frequently
            for _ in range(int(WIFI_CHECK_INTERVAL)):
                if shutdown_requested.is_set():
                    break
                time.sleep(1)
                
        logger.info("WiFi signal monitoring thread exiting")
    
    def _get_wifi_signal_strength(self):
        """Get WiFi signal strength in dBm using iwconfig"""
        if not self.enabled:
            return None
            
        try:
            # Use iwconfig to get signal strength (Linux/Raspberry Pi)
            cmd = ["iwconfig", self.config.wifi_adapter]
            output = subprocess.check_output(cmd, universal_newlines=True, stderr=subprocess.STDOUT)
            
            # Parse output to find signal level
            for line in output.split("\n"):
                # Look for signal level in different formats
                if "Signal level" in line:
                    # Format: Signal level=-70 dBm
                    parts = line.split("Signal level=")
                    if len(parts) > 1:
                        signal_part = parts[1].split(" ")[0]
                        if "dBm" in signal_part:
                            return int(signal_part.replace("dBm", ""))
                        else:
                            return int(signal_part)
                            
                elif "Signal:" in line:
                    # Alternative format: Signal: -65 dBm
                    parts = line.split("Signal:")
                    if len(parts) > 1:
                        signal_part = parts[1].strip().split(" ")[0]
                        try:
                            return int(signal_part)
                        except ValueError:
                            pass
                            
                elif "quality" in line.lower() and "signal" in line.lower():
                    # Format: Quality=70/70 Signal level=-57 dBm
                    if "level" in line:
                        parts = line.split("level=")
                        if len(parts) > 1:
                            signal_part = parts[1].split(" ")[0]
                            try:
                                return int(signal_part)
                            except ValueError:
                                pass
            
            logger.debug(f"Could not parse signal strength from iwconfig output")
            return None
            
        except subprocess.SubprocessError as e:
            logger.error(f"Error running iwconfig: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting WiFi signal strength: {e}")
            return None

# Transfer Manager for handling uploads to remote storage
class TransferManager:
    """Manages transfer of motion events to remote storage"""
    
    def __init__(self, config, wifi_monitor=None):
        self.config = config
        self.wifi_monitor = wifi_monitor
        self.transfer_queue = queue.PriorityQueue()
        self.pending_events = set()
        self.active_transfers = set()
        self.worker_thread = threading.Thread(target=self._transfer_worker)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        
        # Create a timer thread for disk usage checks
        self.disk_check_thread = threading.Thread(target=self._disk_check_worker)
        self.disk_check_thread.daemon = True
        self.disk_check_thread.start()
        
    def add_event(self, event_id):
        """Add an event to the transfer queue"""
        if event_id in self.pending_events:
            return
            
        self.pending_events.add(event_id)
        
        # Check if we should transfer now or during scheduled time
        current_hour = datetime.datetime.now().hour
        in_transfer_window = (self.config.transfer_schedule_start <= current_hour < 
                             self.config.transfer_schedule_end)
                             
        if not self.config.transfer_schedule_active or in_transfer_window:
            # Immediate transfer with normal priority
            self.transfer_queue.put((50, event_id))
            logger.info(f"Added event {event_id} to transfer queue (immediate)")
        else:
            # Scheduled transfer with low priority
            self.transfer_queue.put((100, event_id))
            logger.info(f"Added event {event_id} to transfer queue (scheduled)")
            
    def _transfer_worker(self):
        """Worker thread to process transfers"""
        while not shutdown_requested.is_set():
            try:
                try:
                    # Use a timeout to check shutdown_requested periodically
                    _, event_id = self.transfer_queue.get(timeout=1.0)
                    
                    event_dir = os.path.join(self.config.local_storage_path, event_id)
                    if not os.path.exists(event_dir):
                        if event_id in self.pending_events:
                            self.pending_events.remove(event_id)
                        self.transfer_queue.task_done()
                        continue
                    
                    # Check if we're in the transfer window
                    current_hour = datetime.datetime.now().hour
                    in_transfer_window = (self.config.transfer_schedule_start <= current_hour < 
                                        self.config.transfer_schedule_end)
                    
                    if self.config.transfer_schedule_active and not in_transfer_window:
                        # Reschedule for later with low priority
                        self.transfer_queue.put((100, event_id))
                        self.transfer_queue.task_done()
                        time.sleep(5)
                        continue
                    
                    # Check for shutdown before beginning transfer
                    if shutdown_requested.is_set():
                        # Put the item back in the queue and exit
                        self.transfer_queue.put((50, event_id))
                        self.transfer_queue.task_done()
                        break
                    
                    # Mark as active transfer
                    self.active_transfers.add(event_id)
                    
                    # Attempt to transfer
                    success = False
                    if self.config.chunk_upload:
                        success = self._upload_event_chunked(event_id)
                    else:
                        success = self._upload_event(event_id)
                    
                    if success:
                        # Clean up local storage
                        self._cleanup_event(event_id)
                        if event_id in self.pending_events:
                            self.pending_events.remove(event_id)
                    else:
                        # Retry later with higher priority if we're not shutting down
                        if not shutdown_requested.is_set():
                            time.sleep(self.config.transfer_retry_interval_sec)
                            self.transfer_queue.put((25, event_id))
                        
                    # Remove from active transfers
                    if event_id in self.active_transfers:
                        self.active_transfers.remove(event_id)
                        
                    self.transfer_queue.task_done()
                    
                    # Throttle transfers (short pause between events)
                    if not shutdown_requested.is_set():
                        time.sleep(1)
                        
                except queue.Empty:
                    # No items in queue, just continue and check shutdown flag
                    continue
                    
            except Exception as e:
                logger.error(f"Error in transfer worker: {e}")
                if not shutdown_requested.is_set():
                    time.sleep(5)
        
        logger.info("Transfer worker thread exiting")
                
    def _upload_event(self, event_id):
        """Upload a motion event to remote storage (single request)"""
        event_dir = os.path.join(self.config.local_storage_path, event_id)
        metadata_path = os.path.join(event_dir, "metadata.json")
        video_path = os.path.join(event_dir, "motion.mp4")
        
        if not os.path.exists(metadata_path) or not os.path.exists(video_path):
            logger.error(f"Event files missing for {event_id}")
            return False
            
        try:
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Check if we should throttle based on WiFi signal
            throttle_kbps = self.config.upload_throttle_kbps
            if self.wifi_monitor:
                throttle_kbps = self.wifi_monitor.get_current_throttle()
                
            # Upload video file with throttling if needed
            if throttle_kbps > 0:
                # Use curl or similar with throttling
                cmd = [
                    "curl", "-X", "POST",
                    "-F", f"metadata={json.dumps(metadata)}",
                    "-F", f"video=@{video_path}",
                    "-H", f"X-API-Key: {self.config.remote_api_key}",
                    "--limit-rate", f"{throttle_kbps}k",
                    self.config.remote_storage_url
                ]
                
                # Add thumbnails if available
                thumbnails_dir = os.path.join(event_dir, "thumbnails")
                if os.path.exists(thumbnails_dir):
                    for i, thumb_file in enumerate(os.listdir(thumbnails_dir)):
                        if thumb_file.endswith('.jpg'):
                            thumb_path = os.path.join(thumbnails_dir, thumb_file)
                            cmd.extend(["-F", f"thumbnail_{i}=@{thumb_path}"])
                
                logger.info(f"Uploading event {event_id} with throttle {throttle_kbps} KB/s")
                process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                stdout, stderr = process.communicate()
                
                success = process.returncode == 0
                if not success:
                    logger.error(f"Upload failed for {event_id}: {stderr.decode()}")
                
            else:
                # Use requests library
                files = {'video': open(video_path, 'rb')}
                
                # Add thumbnails if available
                thumbnails_dir = os.path.join(event_dir, "thumbnails")
                if os.path.exists(thumbnails_dir):
                    for i, thumb_file in enumerate(os.listdir(thumbnails_dir)):
                        if thumb_file.endswith('.jpg'):
                            thumb_path = os.path.join(thumbnails_dir, thumb_file)
                            files[f'thumbnail_{i}'] = open(thumb_path, 'rb')
                
                logger.info(f"Uploading event {event_id}")
                response = requests.post(
                    self.config.remote_storage_url,
                    files=files,
                    data={'metadata': json.dumps(metadata)},
                    headers={'X-API-Key': self.config.remote_api_key}
                )
                
                # Close all file handles
                for f in files.values():
                    f.close()
                    
                success = response.status_code == 200
                if not success:
                    logger.error(f"Upload failed for {event_id}: {response.text}")
                    
            if success:
                logger.info(f"Successfully uploaded event {event_id}")
            
            return success
                
        except Exception as e:
            logger.error(f"Error uploading event {event_id}: {e}")
            return False
    
    def _upload_event_chunked(self, event_id):
        """Upload a motion event to remote storage using chunked upload"""
        event_dir = os.path.join(self.config.local_storage_path, event_id)
        metadata_path = os.path.join(event_dir, "metadata.json")
        video_path = os.path.join(event_dir, "motion.mp4")
        upload_id = None  # Track upload ID for cleanup
        
        # Validate files exist
        if not os.path.exists(metadata_path):
            logger.error(f"Metadata file missing for {event_id}")
            return False
            
        if not os.path.exists(video_path):
            logger.error(f"Video file missing for {event_id}")
            return False
            
        try:
            # Load metadata
            with open(metadata_path, 'r') as f:
                try:
                    metadata = json.load(f)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid metadata JSON for {event_id}: {e}")
                    return False
                
            # Get file size
            video_size = os.path.getsize(video_path)
            if video_size == 0:
                logger.error(f"Empty video file for {event_id}")
                return False
                
            # Determine chunk size based on file size
            chunk_size = min(MAX_UPLOAD_CHUNK_SIZE, max(MIN_UPLOAD_CHUNK_SIZE, video_size // 10))
            
            # Check if we should throttle based on WiFi signal
            throttle_kbps = self.config.upload_throttle_kbps
            if self.wifi_monitor:
                throttle_kbps = self.wifi_monitor.get_current_throttle()
                
            # First, initialize chunked upload
            init_data = {
                'metadata': json.dumps(metadata),
                'file_size': video_size,
                'chunk_size': chunk_size,
                'total_chunks': (video_size + chunk_size - 1) // chunk_size
            }
            
            # Upload thumbnails separately if they exist
            thumbnails = []
            thumbnails_dir = os.path.join(event_dir, "thumbnails")
            if os.path.exists(thumbnails_dir):
                for thumb_file in os.listdir(thumbnails_dir):
                    if thumb_file.endswith('.jpg'):
                        thumb_path = os.path.join(thumbnails_dir, thumb_file)
                        try:
                            with open(thumb_path, 'rb') as f:
                                thumb_data = f.read()
                                thumbnails.append(base64.b64encode(thumb_data).decode('utf-8'))
                        except Exception as e:
                            logger.warning(f"Failed to read thumbnail {thumb_file} for {event_id}: {e}")
                            # Continue without this thumbnail
            
            if thumbnails:
                init_data['thumbnails'] = thumbnails
                
            logger.info(f"Initializing chunked upload for event {event_id} "
                      f"({video_size / 1024 / 1024:.2f} MB, {init_data['total_chunks']} chunks)")
            
            # Initialize upload with retry logic
            init_retry_count = 0
            init_success = False
            
            while init_retry_count < MAX_RETRIES and not init_success:
                try:
                    response = requests.post(
                        f"{self.config.remote_storage_url}/chunked/init",
                        json=init_data,
                        headers={'X-API-Key': self.config.remote_api_key},
                        timeout=30  # Add timeout to prevent hanging
                    )
                    
                    if response.status_code == 200:
                        init_success = True
                        # Get upload ID
                        upload_id = response.json().get('upload_id')
                        if not upload_id:
                            logger.error(f"No upload ID received for {event_id}")
                            return False
                    else:
                        init_retry_count += 1
                        logger.warning(f"Failed to initialize upload for {event_id} (attempt {init_retry_count}/{MAX_RETRIES}): {response.text}")
                        if init_retry_count < MAX_RETRIES:
                            time.sleep(2 ** init_retry_count)  # Exponential backoff
                            
                except (requests.RequestException, IOError) as e:
                    init_retry_count += 1
                    logger.warning(f"Network error initializing upload for {event_id} (attempt {init_retry_count}/{MAX_RETRIES}): {e}")
                    if init_retry_count < MAX_RETRIES:
                        time.sleep(2 ** init_retry_count)  # Exponential backoff
            
            if not init_success:
                logger.error(f"Failed to initialize upload for {event_id} after {MAX_RETRIES} attempts")
                return False
                
            # Upload chunks
            try:
                with open(video_path, 'rb') as f:
                    chunk_index = 0
                    retries = 0
                    total_chunks = init_data['total_chunks']
                    
                    while True:
                        # Read chunk
                        chunk_data = f.read(chunk_size)
                        if not chunk_data:
                            break
                            
                        # Check if we should throttle uploads
                        if throttle_kbps > 0:
                            # Calculate delay based on chunk size and throttle
                            delay = (len(chunk_data) * 8) / (throttle_kbps * 1000)
                            time.sleep(delay)
                            
                        # Upload chunk
                        chunk_b64 = base64.b64encode(chunk_data).decode('utf-8')
                        
                        upload_success = False
                        chunk_retry_count = 0
                        
                        while not upload_success and chunk_retry_count < MAX_RETRIES:
                            try:
                                chunk_response = requests.post(
                                    f"{self.config.remote_storage_url}/chunked/upload",
                                    json={
                                        'upload_id': upload_id,
                                        'chunk_index': chunk_index,
                                        'chunk_data': chunk_b64
                                    },
                                    headers={'X-API-Key': self.config.remote_api_key},
                                    timeout=60  # Add timeout for large chunks
                                )
                                
                                if chunk_response.status_code == 200:
                                    upload_success = True
                                    chunk_index += 1
                                    logger.debug(f"Uploaded chunk {chunk_index}/{total_chunks} "
                                               f"for {event_id}")
                                else:
                                    chunk_retry_count += 1
                                    logger.warning(f"Failed to upload chunk {chunk_index} for {event_id} "
                                                f"(attempt {chunk_retry_count}/{MAX_RETRIES}): "
                                                f"{chunk_response.text}")
                                    if chunk_retry_count < MAX_RETRIES:
                                        time.sleep(2 ** chunk_retry_count)  # Exponential backoff
                            
                            except (requests.RequestException, IOError) as e:
                                chunk_retry_count += 1
                                logger.warning(f"Network error uploading chunk {chunk_index} for {event_id} "
                                            f"(attempt {chunk_retry_count}/{MAX_RETRIES}): {e}")
                                if chunk_retry_count < MAX_RETRIES:
                                    time.sleep(2 ** chunk_retry_count)  # Exponential backoff
                        
                        if not upload_success:
                            logger.error(f"Failed to upload chunk {chunk_index} for {event_id} after {MAX_RETRIES} attempts")
                            return False
            
                    # All chunks uploaded successfully, now finalize
                    finalize_success = False
                    finalize_retry_count = 0
                    
                    while not finalize_success and finalize_retry_count < MAX_RETRIES:
                        try:
                            finalize_response = requests.post(
                                f"{self.config.remote_storage_url}/chunked/finalize",
                                json={'upload_id': upload_id},
                                headers={'X-API-Key': self.config.remote_api_key},
                                timeout=30
                            )
                            
                            if finalize_response.status_code == 200:
                                finalize_success = True
                                logger.info(f"Successfully uploaded event {event_id} in {chunk_index} chunks")
                                return True
                            else:
                                finalize_retry_count += 1
                                logger.warning(f"Failed to finalize upload for {event_id} "
                                            f"(attempt {finalize_retry_count}/{MAX_RETRIES}): "
                                            f"{finalize_response.text}")
                                if finalize_retry_count < MAX_RETRIES:
                                    time.sleep(2 ** finalize_retry_count)  # Exponential backoff
                        
                        except (requests.RequestException, IOError) as e:
                            finalize_retry_count += 1
                            logger.warning(f"Network error finalizing upload for {event_id} "
                                        f"(attempt {finalize_retry_count}/{MAX_RETRIES}): {e}")
                            if finalize_retry_count < MAX_RETRIES:
                                time.sleep(2 ** finalize_retry_count)  # Exponential backoff
                    
                    if not finalize_success:
                        logger.error(f"Failed to finalize upload for {event_id} after {MAX_RETRIES} attempts")
                        return False
                    
            except IOError as e:
                logger.error(f"File I/O error during chunked upload for {event_id}: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Error in chunked upload for {event_id}: {e}")
            
            # Try to clean up remote upload if it was initialized but failed
            if upload_id:
                try:
                    # Best effort to cancel the upload
                    requests.post(
                        f"{self.config.remote_storage_url}/chunked/cancel",
                        json={'upload_id': upload_id},
                        headers={'X-API-Key': self.config.remote_api_key},
                        timeout=10
                    )
                    logger.info(f"Sent cancellation request for upload {upload_id}")
                except Exception as cancel_err:
                    logger.warning(f"Failed to cancel upload {upload_id}: {cancel_err}")
                    
            return False
            
    def _cleanup_event(self, event_id):
        """Clean up local storage after successful upload"""
        event_dir = os.path.join(self.config.local_storage_path, event_id)
        try:
            shutil.rmtree(event_dir)
            logger.info(f"Cleaned up event {event_id}")
        except Exception as e:
            logger.error(f"Error cleaning up event {event_id}: {e}")
    
    def _disk_check_worker(self):
        """Periodic worker to check disk usage and clean up if needed"""
        while not shutdown_requested.is_set():
            try:
                self.check_disk_usage()
            except Exception as e:
                logger.error(f"Error in disk check: {e}")
                
            # Check disk usage every 5 minutes - use smaller sleeps to check shutdown flag more frequently
            for _ in range(60):  # 60 * 5 seconds = 300 seconds (5 minutes)
                if shutdown_requested.is_set():
                    break
                time.sleep(5)
        
        logger.info("Disk check worker thread exiting")
            
    def check_disk_usage(self):
        """Check and clean up disk if needed"""
        if not os.path.exists(self.config.local_storage_path):
            return
            
        # Get total size of storage directory
        total_size = 0
        event_times = {}
        
        for event_id in os.listdir(self.config.local_storage_path):
            event_dir = os.path.join(self.config.local_storage_path, event_id)
            if not os.path.isdir(event_dir):
                continue
                
            # Skip events that are being transferred
            if event_id in self.active_transfers:
                continue
                
            # Get size and timestamp
            dir_size = 0
            event_time = os.path.getctime(event_dir)
            
            for path, dirs, files in os.walk(event_dir):
                for f in files:
                    file_path = os.path.join(path, f)
                    dir_size += os.path.getsize(file_path)
                    
            total_size += dir_size
            event_times[event_id] = (event_time, dir_size)
            
        # Check if we need to clean up
        max_size_bytes = self.config.max_disk_usage_mb * 1024 * 1024
        
        if total_size > max_size_bytes:
            # Sort events by time (oldest first)
            sorted_events = sorted(event_times.items(), key=lambda x: x[1][0])
            
            # Remove oldest events until under limit
            bytes_to_remove = total_size - max_size_bytes
            bytes_removed = 0
            
            for event_id, (_, size) in sorted_events:
                # Skip events that are pending upload
                if event_id in self.pending_events:
                    continue
                    
                event_dir = os.path.join(self.config.local_storage_path, event_id)
                try:
                    shutil.rmtree(event_dir)
                    bytes_removed += size
                    logger.info(f"Removed old event {event_id} to free space")
                    
                    if bytes_removed >= bytes_to_remove:
                        break
                except Exception as e:
                    logger.error(f"Error removing event {event_id}: {e}")
            
            logger.info(f"Disk cleanup: removed {bytes_removed / 1024 / 1024:.2f} MB, "
                      f"current usage: {(total_size - bytes_removed) / 1024 / 1024:.2f} MB / "
                      f"{self.config.max_disk_usage_mb} MB")

# Initialize components
storage_config = StorageConfig()
frame_buffer = CircularFrameBuffer(max_size=storage_config.max_ram_segments)
wifi_monitor = WiFiMonitor(storage_config)
transfer_manager = TransferManager(storage_config, wifi_monitor)
motion_recorder = MotionEventRecorder(frame_buffer, storage_config)

# Integration functions

def integrate_with_fastapi_server(app, camera_config):
    """Integrate motion storage module with FastAPI server"""
    
    @app.get("/storage/status")
    async def get_storage_status():
        """Get storage status"""
        # Check disk usage first
        transfer_manager.check_disk_usage()
        
        # Get list of pending events
        pending_events = []
        for event_id in transfer_manager.pending_events:
            event_dir = os.path.join(storage_config.local_storage_path, event_id)
            metadata_path = os.path.join(event_dir, "metadata.json")
            
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    pending_events.append(metadata)
                except Exception:
                    pending_events.append({"id": event_id, "error": "Could not read metadata"})
        
        # Get storage stats
        total_size = 0
        event_count = 0
        
        if os.path.exists(storage_config.local_storage_path):
            for path, dirs, files in os.walk(storage_config.local_storage_path):
                for f in files:
                    file_path = os.path.join(path, f)
                    total_size += os.path.getsize(file_path)
            
            event_count = len([d for d in os.listdir(storage_config.local_storage_path) 
                             if os.path.isdir(os.path.join(storage_config.local_storage_path, d))])
        
        # WiFi signal info
        wifi_info = None
        if wifi_monitor and wifi_monitor.enabled:
            wifi_info = {
                "signal_strength": wifi_monitor.current_signal,
                "current_throttle": wifi_monitor.current_throttle
            }
        
        return {
            "storage": {
                "path": storage_config.local_storage_path,
                "size_mb": round(total_size / (1024 * 1024), 2),
                "max_size_mb": storage_config.max_disk_usage_mb,
                "usage_percent": round((total_size / (storage_config.max_disk_usage_mb * 1024 * 1024)) * 100, 1) 
                               if storage_config.max_disk_usage_mb > 0 else 0,
                "event_count": event_count
            },
            "transfer": {
                "pending_count": len(transfer_manager.pending_events),
                "active_transfers": len(transfer_manager.active_transfers),
                "throttle_kbps": storage_config.upload_throttle_kbps,
                "schedule_active": storage_config.transfer_schedule_active,
                "schedule_window": f"{storage_config.transfer_schedule_start}:00-{storage_config.transfer_schedule_end}:00",
                "wifi_monitoring": wifi_info
            },
            "pending_events": pending_events
        }

    @app.post("/storage/config")
    async def update_storage_config(config: Dict[str, Any]):
        """Update storage configuration"""
        global storage_config
        
        if "max_disk_usage_mb" in config:
            storage_config.max_disk_usage_mb = config["max_disk_usage_mb"]
        if "upload_throttle_kbps" in config:
            storage_config.upload_throttle_kbps = config["upload_throttle_kbps"]
        if "transfer_schedule_active" in config:
            storage_config.transfer_schedule_active = config["transfer_schedule_active"]
        if "transfer_schedule_start" in config:
            storage_config.transfer_schedule_start = config["transfer_schedule_start"]
        if "transfer_schedule_end" in config:
            storage_config.transfer_schedule_end = config["transfer_schedule_end"]
        if "remote_storage_url" in config:
            storage_config.remote_storage_url = config["remote_storage_url"]
        if "remote_api_key" in config:
            storage_config.remote_api_key = config["remote_api_key"]
        if "generate_thumbnails" in config:
            storage_config.generate_thumbnails = config["generate_thumbnails"]
        if "wifi_monitoring" in config:
            storage_config.wifi_monitoring = config["wifi_monitoring"]
            
        return {"message": "Storage configuration updated"}

    @app.post("/storage/transfer/{event_id}")
    async def force_transfer(event_id: str):
        """Force transfer of a specific event"""
        event_dir = os.path.join(storage_config.local_storage_path, event_id)
        
        if not os.path.exists(event_dir):
            return {"error": "Event not found", "status": "error"}
            
        # Add to transfer queue with high priority
        transfer_manager.add_event(event_id)
        transfer_manager.transfer_queue.put((10, event_id))  # High priority
        
        return {"message": f"Transfer of event {event_id} initiated", "status": "success"}

    @app.get("/storage/events")
    async def list_events():
        """List all motion events in storage"""
        events = []
        
        if os.path.exists(storage_config.local_storage_path):
            for event_id in os.listdir(storage_config.local_storage_path):
                event_dir = os.path.join(storage_config.local_storage_path, event_id)
                if not os.path.isdir(event_dir):
                    continue
                    
                metadata_path = os.path.join(event_dir, "metadata.json")
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        # Add transfer status
                        metadata["transfer_status"] = "pending" if event_id in transfer_manager.pending_events else "none"
                        if event_id in transfer_manager.active_transfers:
                            metadata["transfer_status"] = "active"
                            
                        # Add file size
                        video_path = os.path.join(event_dir, "motion.mp4")
                        if os.path.exists(video_path):
                            metadata["file_size_mb"] = round(os.path.getsize(video_path) / (1024 * 1024), 2)
                            
                        # Check for thumbnails
                        thumbnails_dir = os.path.join(event_dir, "thumbnails")
                        metadata["has_thumbnails"] = os.path.exists(thumbnails_dir)
                        
                        events.append(metadata)
                    except Exception as e:
                        events.append({
                            "id": event_id,
                            "error": str(e),
                            "transfer_status": "error"
                        })
        
        return {"events": events, "count": len(events)}
    
    @app.get("/storage/events/{event_id}/thumbnail")
    async def get_event_thumbnail(event_id: str):
        """Get thumbnail for a motion event"""
        event_dir = os.path.join(storage_config.local_storage_path, event_id)
        thumbnails_dir = os.path.join(event_dir, "thumbnails")
        
        if not os.path.exists(thumbnails_dir):
            # Try to generate thumbnail on the fly if video exists
            video_path = os.path.join(event_dir, "motion.mp4")
            if os.path.exists(video_path):
                try:
                    # Create thumbnails directory
                    os.makedirs(thumbnails_dir, exist_ok=True)
                    
                    # Open video and extract frame
                    cap = cv2.VideoCapture(video_path)
                    success, frame = cap.read()
                    if success:
                        # Resize to thumbnail size
                        thumb = cv2.resize(
                            frame, 
                            (storage_config.thumbnail_width, storage_config.thumbnail_height)
                        )
                        # Save thumbnail
                        thumb_path = os.path.join(thumbnails_dir, "thumb_0.jpg")
                        cv2.imwrite(thumb_path, thumb)
                    cap.release()
                except Exception as e:
                    logger.error(f"Error generating thumbnail for {event_id}: {e}")
        
        # Look for thumbnails
        if os.path.exists(thumbnails_dir):
            thumbnails = [f for f in os.listdir(thumbnails_dir) if f.endswith('.jpg')]
            if thumbnails:
                thumb_path = os.path.join(thumbnails_dir, thumbnails[0])
                from fastapi.responses import FileResponse
                return FileResponse(thumb_path, media_type="image/jpeg")
        
        # Return 404 if no thumbnail found
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Thumbnail not found")

    @app.post("/storage/cleanup")
    async def cleanup_storage():
        """Force storage cleanup"""
        transfer_manager.check_disk_usage()
        return {"message": "Storage cleanup initiated"}

def modify_frame_buffer_write(original_write_method, stream_buffer_instance=None):
    """
    Modify the FrameBuffer.write method to integrate with motion storage
    
    Args:
        original_write_method: The original write method of the FrameBuffer instance
        stream_buffer_instance: The specific FastAPI FrameBuffer instance to use
                              (needed to avoid confusion with CircularFrameBuffer)
    """
    # We need to store a reference to the actual stream buffer instance from the FastAPI module
    # This is different from our global frame_buffer which is a CircularFrameBuffer
    stream_buffer = stream_buffer_instance
    
    def write_wrapper(self_or_frame, *args, **kwargs):
        """
        Wrapper to handle both direct calls and calls from PiCamera2's FileOutput.
        This wrapper detects whether it's being called directly as a method or 
        indirectly through PiCamera2's FileOutput._write method and adapts accordingly.
        
        PiCamera2 calling patterns:
        1. Direct method call: instance.write(buf) -> write(instance, buf)
        2. FileOutput._write: self._fileoutput.write(frame) -> write(frame)
        """
        global prev_frame, motion_detected, motion_regions
        global frame_buffer  # This refers to the CircularFrameBuffer for storing frames
        
        # Detect calling pattern and adapt
        if hasattr(self_or_frame, 'raw_frame'):  # It's a direct method call
            # This is a direct call with self as first arg
            instance = self_or_frame
            buf = args[0] if args else kwargs.get('buf')
            # Call the original method through the instance's _original_write attribute
            result = instance._original_write(buf, *args[1:], **kwargs)
        else:  # It's called from PiCamera2 FileOutput._write
            # In this case, self_or_frame is actually the frame data
            buf = self_or_frame
            # Use the saved original method from the stream buffer instance (from FastAPI)
            result = stream_buffer._original_write(buf)
            
        try:
            # Now process the frame using the stream_buffer (the actual FastAPI FrameBuffer)
            if stream_buffer and hasattr(stream_buffer, 'raw_frame') and stream_buffer.raw_frame is not None:
                # Add the frame to our CircularFrameBuffer for motion detection
                frame_buffer.add_frame(stream_buffer.raw_frame.copy(), datetime.datetime.now())
                
                # Handle motion recording if motion is detected
                if motion_detected:
                    if not motion_recorder.recording:
                        motion_recorder.start_recording(motion_regions)
                    motion_recorder.add_frame(stream_buffer.raw_frame.copy(), motion_regions)
                elif motion_recorder.recording:
                    motion_recorder.stop_recording()
        except Exception as e:
            logger.error(f"Error in frame buffer integration: {e}")
            
        return result
        
    return write_wrapper

# Function to initialize the module
def initialize(app=None, camera_config=None, external_storage_config=None):
    """Initialize the motion storage module and integrate with FastAPI server"""
    
    # Reset the shutdown event (in case it was previously set)
    global shutdown_requested
    shutdown_requested.clear()
    
    # Use the provided storage_config if available, otherwise keep using our default
    global storage_config, frame_buffer, wifi_monitor, transfer_manager, motion_recorder
    
    # Update our storage config with values from external_storage_config if provided
    if external_storage_config is not None:
        logger.info("Using storage configuration from FastAPI server")
        # Copy all attributes from the external config to our internal config
        for attr in dir(external_storage_config):
            # Skip private/special attributes
            if attr.startswith('_'):
                continue
            # Skip methods/callables
            if callable(getattr(external_storage_config, attr)):
                continue
            # Copy the attribute value if it exists in our storage_config too
            if hasattr(storage_config, attr):
                setattr(storage_config, attr, getattr(external_storage_config, attr))
        logger.info(f"Using storage path: {storage_config.local_storage_path}")
    
    # Reinitialize components with updated config
    frame_buffer = CircularFrameBuffer(max_size=storage_config.max_ram_segments)
    wifi_monitor = WiFiMonitor(storage_config)
    transfer_manager = TransferManager(storage_config, wifi_monitor)
    motion_recorder = MotionEventRecorder(frame_buffer, storage_config)
    
    # Create storage directory if it doesn't exist
    os.makedirs(storage_config.local_storage_path, exist_ok=True)
    
    # Log initialization
    logger.info("Motion Storage Module initialized")
    logger.info(f"Storage path: {storage_config.local_storage_path}")
    logger.info(f"RAM buffer capacity: {storage_config.max_ram_segments} frames")
    logger.info(f"WiFi monitoring: {'Enabled' if wifi_monitor.enabled else 'Disabled'}")
    logger.info(f"Chunked uploads: {'Enabled' if storage_config.chunk_upload else 'Disabled'}")
    logger.info(f"Thumbnail generation: {'Enabled' if storage_config.generate_thumbnails else 'Disabled'}")
    
    # Integrate with FastAPI server if provided
    if app is not None:
        integrate_with_fastapi_server(app, camera_config)
        logger.info("Integrated with FastAPI server")
        
    return {
        'frame_buffer': frame_buffer,
        'motion_recorder': motion_recorder,
        'transfer_manager': transfer_manager,
        'wifi_monitor': wifi_monitor,
        'storage_config': storage_config,
        'modify_frame_buffer_write': modify_frame_buffer_write,  # Now expects an additional stream_buffer_instance parameter
        'shutdown': shutdown  # Add the shutdown function to the returned resources
    }

def shutdown():
    """Shutdown all background threads and processes"""
    logger.info("Shutting down motion storage module...")
    
    # Set the shutdown event to signal all threads to exit
    global shutdown_requested
    shutdown_requested.set()
    
    # Give threads time to exit gracefully
    logger.info("Waiting for background threads to exit...")
    
    # Wait a bit for threads to notice the shutdown event
    time.sleep(2)
    
    logger.info("Motion storage module shutdown complete")
    
    return True

if __name__ == "__main__":
    print("This module should be imported and initialized from the main FastAPI server.")

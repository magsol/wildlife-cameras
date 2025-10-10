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
from contextlib import ExitStack
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set

# Import optical flow analyzer for classification
from optical_flow_analyzer import OpticalFlowAnalyzer, MotionPatternDatabase

# Import centralized configuration (optional - for standalone use)
try:
    from config import get_config, StorageConfig as CentralizedStorageConfig
    _has_centralized_config = True
except ImportError:
    _has_centralized_config = False

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

# Configure logging - Set to DEBUG to capture all levels
logging.basicConfig(
    level=logging.DEBUG,  # Changed from INFO to DEBUG to capture more detailed logs
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("motion_storage")

# Force this module's logger to DEBUG level to ensure we capture all logs
logger.setLevel(logging.DEBUG)

# Add an explicit startup message to verify logging is working
logger.critical("MOTION_STORAGE MODULE LOADED - LOGGING INITIALIZED AT DEBUG LEVEL")

# Motion detection variables - initialized here for module scope
prev_frame = None
motion_detected = False
motion_regions = []

# Global shutdown event for signaling threads to terminate
shutdown_requested = threading.Event()

# Optical flow globals - will be set by main module
_optical_flow_analyzer = None
_motion_pattern_db = None

def set_optical_flow_components(analyzer, pattern_db):
    """Set optical flow analyzer and pattern database from main module"""
    global _optical_flow_analyzer, _motion_pattern_db
    _optical_flow_analyzer = analyzer
    _motion_pattern_db = pattern_db
    logger.info("Optical flow components set in motion_storage module")

# Storage configuration
@dataclass
# DEPRECATED: StorageConfig is now in config.py
# This class is kept for backward compatibility but is no longer used internally
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

    # Optical flow storage settings
    store_optical_flow_data: bool = True
    optical_flow_signature_dir: str = "flow_signatures"
    optical_flow_database_path: str = "motion_patterns.db"

    # Motion classification settings
    motion_classification_enabled: bool = True
    min_classification_confidence: float = 0.5
    save_flow_visualizations: bool = True

    # Performance optimization
    optical_flow_max_resolution: Tuple[int, int] = (320, 240)  # Downscale for flow computation

# Circular buffer for RAM storage
class CircularFrameBuffer:
    """Stores recent frames in RAM using a circular buffer"""
    
    def __init__(self, max_size=300):  # Default: 10 seconds @ 30fps
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        logger.debug(f"[CRITICAL_DEBUG] CircularFrameBuffer initialized with max_size={max_size}")
        self.frames_added = 0  # Track total frames added for debugging
        
    def add_frame(self, frame, timestamp):
        with self.lock:
            frame_time = timestamp.strftime("%H:%M:%S.%f")[:-3]
            
            # Enhanced logging for CircularFrameBuffer
            self.frames_added += 1
            
            # Check frame quality
            frame_ok = frame is not None
            frame_shape = frame.shape if frame_ok and hasattr(frame, 'shape') else 'N/A'
            
            # Critical debug log for first 5 frames, then every 30th frame
            if self.frames_added <= 5 or self.frames_added % 30 == 0:
                logger.debug(f"[CRITICAL_DEBUG] {frame_time} CircularFrameBuffer.add_frame called")
                logger.debug(f"[CRITICAL_DEBUG] {frame_time} Frame #{self.frames_added}, valid: {frame_ok}, shape: {frame_shape}")
                logger.debug(f"[CRITICAL_DEBUG] {frame_time} Buffer before add: {len(self.buffer)}/{self.buffer.maxlen}")
            
            # Add to buffer
            try:
                self.buffer.append((frame, timestamp))
                
                # Verify frame was added
                if self.frames_added <= 5 or self.frames_added % 30 == 0:
                    logger.debug(f"[CRITICAL_DEBUG] {frame_time} Buffer after add: {len(self.buffer)}/{self.buffer.maxlen}")
            except Exception as e:
                logger.error(f"[CRITICAL_DEBUG] {frame_time} ERROR adding frame to buffer: {e}")
                import traceback
                logger.error(f"[CRITICAL_DEBUG] {frame_time} Error traceback: {traceback.format_exc()}")
            
    def get_recent_frames(self, seconds):
        with self.lock:
            current_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            
            if not self.buffer:
                logger.error(f"[CRITICAL_DEBUG] {current_time} Buffer EMPTY when requesting recent frames!")
                logger.error(f"[CRITICAL_DEBUG] {current_time} Total frames ever added: {self.frames_added}")
                return []
            
            # Log buffer details
            buffer_len = len(self.buffer)
            buffer_timespan = None
            if buffer_len >= 2:
                buffer_timespan = (self.buffer[-1][1] - self.buffer[0][1]).total_seconds()
                
            logger.debug(f"[CRITICAL_DEBUG] {current_time} get_recent_frames({seconds}) called")
            logger.debug(f"[CRITICAL_DEBUG] {current_time} Current buffer size: {buffer_len}/{self.buffer.maxlen}")
            logger.debug(f"[CRITICAL_DEBUG] {current_time} Buffer timespan: {buffer_timespan}s")
            
            # Get frames from the buffer within the timespan
            try:
                cutoff_time = self.buffer[-1][1] - datetime.timedelta(seconds=seconds)
                recent_frames = [frame for frame, ts in self.buffer if ts >= cutoff_time]
                
                logger.info(f"[CRITICAL_DEBUG] {current_time} Retrieved {len(recent_frames)} recent frames from buffer (from past {seconds} seconds)")
                logger.info(f"[CRITICAL_DEBUG] {current_time} Retrieval success rate: {len(recent_frames)}/{buffer_len}")
                
                return recent_frames
                
            except Exception as e:
                logger.error(f"[CRITICAL_DEBUG] {current_time} ERROR retrieving recent frames: {e}")
                import traceback
                logger.error(f"[CRITICAL_DEBUG] {current_time} Error traceback: {traceback.format_exc()}")
                return []

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
            logger.debug(f"[MOTION_FLOW] Already recording motion - ignoring start_recording call")
            return

        start_time = datetime.datetime.now()
        event_id = self._generate_event_id()

        # Log motion event start with high visibility
        frame_time = start_time.strftime("%H:%M:%S.%f")[:-3]
        logger.critical(f"[EVENT_STARTED] {frame_time} NEW MOTION EVENT STARTED - ID: {event_id}")
        logger.critical(f"[EVENT_STARTED] Motion regions: {len(motion_regions)}, min duration threshold: {self.config.min_motion_duration_sec}s")

        # Reset optical flow analyzer for new event
        if _optical_flow_analyzer is not None:
            _optical_flow_analyzer.reset()
            logger.debug(f"[OPTICAL_FLOW] Analyzer reset for new event {event_id}")

        self.recording = True
        self.current_event = {
            "start_time": start_time,
            "frames": [],
            "regions": motion_regions,
            "id": event_id
        }
        
        # Get buffer frames from before motion started
        logger.debug(f"[EVENT_STARTED] {frame_time} Getting pre-motion frames from buffer (past {self.config.ram_buffer_seconds}s)")
        pre_frames = self.frame_buffer.get_recent_frames(
            self.config.ram_buffer_seconds
        )
        self.current_event["frames"].extend(pre_frames)
        
        # Always log the buffer details
        if hasattr(self.frame_buffer, 'buffer'):
            buffer_size = len(self.frame_buffer.buffer) if hasattr(self.frame_buffer.buffer, '__len__') else 'unknown'
            logger.critical(f"[EVENT_STARTED] {frame_time} Current frame buffer size: {buffer_size}")
        
        # Log the event creation details
        logger.critical(f"[EVENT_STARTED] {frame_time} Added {len(pre_frames)} pre-motion frames from buffer")
        logger.info(f"[MOTION_FLOW] {frame_time} Started recording motion event {event_id} - added {len(pre_frames)} pre-motion frames from buffer")
        logger.info(f"[MOTION_FLOW] Event {event_id}: initial motion regions: {len(motion_regions)}")
        
        # Log the queue status
        logger.info(f"[MOTION_FLOW] Current events queue size: {self.events_queue.qsize()}")
        logger.debug(f"[MOTION_FLOW] Storage path: {self.config.local_storage_path}, Max size: {self.config.max_disk_usage_mb}MB")
        
    def add_frame(self, frame, motion_regions):
        """Add a frame to the current motion event"""
        if not self.recording:
            return
        
        timestamp = datetime.datetime.now()
        self.current_event["frames"].append((frame, timestamp))
        self.current_event["regions"] = motion_regions
        
        # Log every 30th frame to reduce log volume
        frames_count = len(self.current_event["frames"])
        if frames_count % 30 == 0:
            frame_time = timestamp.strftime("%H:%M:%S.%f")[:-3]
            event_id = self.current_event["id"]
            duration = (timestamp - self.current_event["start_time"]).total_seconds()
            logger.debug(f"[MOTION_FLOW] {frame_time} Event {event_id}: added frame {frames_count}, current duration: {duration:.1f}s")
        
    def stop_recording(self):
        """Stop recording the current motion event"""
        if not self.recording:
            logger.debug(f"[MOTION_FLOW] Not recording - ignoring stop_recording call")
            return
            
        self.recording = False
        end_time = datetime.datetime.now()
        self.current_event["end_time"] = end_time
        
        # Check if event meets minimum duration
        duration = (self.current_event["end_time"] - 
                   self.current_event["start_time"]).total_seconds()
        
        frame_time = end_time.strftime("%H:%M:%S.%f")[:-3]
        event_id = self.current_event['id']
        frames_count = len(self.current_event["frames"])
        
        # CRITICAL DEBUG - Always log the duration check with high visibility
        logger.critical(f"[EVENT_DURATION] {frame_time} Motion event {event_id}: duration={duration:.3f}s, min={self.config.min_motion_duration_sec}s, frames={frames_count}")
        
        if duration >= self.config.min_motion_duration_sec:
            # Add to processing queue with priority based on size
            priority = frames_count
            self.events_queue.put((priority, self.current_event))
            
            logger.info(f"[MOTION_FLOW] {frame_time} Stopped recording motion event {event_id} - ")
            logger.info(f"[MOTION_FLOW] Event {event_id}: {frames_count} frames, {duration:.1f}s duration")
            logger.info(f"[MOTION_FLOW] Event {event_id}: Added to processing queue with priority {priority}")
            logger.info(f"[MOTION_FLOW] Events queue size now: {self.events_queue.qsize()}")
            
            # Also log storage path to verify where event should be saved
            logger.critical(f"[EVENT_ACCEPTED] {frame_time} Motion event {event_id} ACCEPTED - will be saved to {self.config.local_storage_path}/{event_id}")
        else:
            # Log with higher visibility for brief motion events
            logger.critical(f"[EVENT_REJECTED] {frame_time} Motion event {event_id} REJECTED - duration too short")
            logger.critical(f"[EVENT_REJECTED] Duration {duration:.3f}s < minimum {self.config.min_motion_duration_sec}s threshold")
            logger.critical(f"[EVENT_REJECTED] Frames captured: {frames_count}, Config: {self.config}")
            
            # Log more details about the motion that was detected but rejected
            if "regions" in self.current_event and self.current_event["regions"]:
                regions_count = len(self.current_event["regions"])
                logger.critical(f"[EVENT_REJECTED] Motion regions detected: {regions_count}")
                
            logger.info(f"[MOTION_FLOW] {frame_time} Discarded motion event {event_id} ")
            logger.info(f"[MOTION_FLOW] Event {event_id}: Duration too short ({duration:.1f}s < minimum {self.config.min_motion_duration_sec}s)")
        
        self.current_event = None
        
    def _process_events(self):
        """Worker thread to process recorded events"""
        while not shutdown_requested.is_set():
            try:
                try:
                    # Use a timeout to check shutdown_requested periodically
                    logger.debug(f"[MOTION_FLOW] Worker thread checking event queue. Size: {self.events_queue.qsize()}")
                    _, event = self.events_queue.get(timeout=1.0)
                    
                    event_id = event['id']
                    frame_count = len(event['frames'])
                    duration = (event['end_time'] - event['start_time']).total_seconds()
                    frame_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    
                    logger.info(f"[MOTION_FLOW] {frame_time} Processing event {event_id} ({frame_count} frames, {duration:.1f}s)")
                    self._save_event_to_disk(event)
                    self.events_queue.task_done()
                    
                    logger.info(f"[MOTION_FLOW] {frame_time} Event queue size after processing: {self.events_queue.qsize()}")
                    
                except queue.Empty:
                    # No items in queue, just continue and check shutdown flag
                    continue
            except Exception as e:
                logger.error(f"[MOTION_FLOW] Error processing motion event: {e}")
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
        event_id = event["id"]
        event_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # Create directory structure
        event_dir = os.path.join(
            self.config.local_storage_path,
            event_id
        )
        
        logger.info(f"[MOTION_FLOW] {event_time} Creating directory for event {event_id}: {event_dir}")
        os.makedirs(event_dir, exist_ok=True)
        
        # Check if directory was created successfully
        if not os.path.exists(event_dir):
            logger.error(f"[MOTION_FLOW] {event_time} Failed to create directory for event {event_id}!")
            return
        else:
            logger.info(f"[MOTION_FLOW] {event_time} Directory for event {event_id} created successfully")
            logger.info(f"[MOTION_FLOW] Directory exists check: {os.path.exists(event_dir)}")
            logger.info(f"[MOTION_FLOW] Directory permissions: {oct(os.stat(event_dir).st_mode)[-3:]}")
            logger.info(f"[MOTION_FLOW] Parent directory: {os.path.dirname(event_dir)}")
            logger.info(f"[MOTION_FLOW] Parent exists check: {os.path.exists(os.path.dirname(event_dir))}")
            
        # Check storage space
        try:
            statvfs = os.statvfs(self.config.local_storage_path)
            free_mb = (statvfs.f_frsize * statvfs.f_bavail) / (1024 * 1024)
            logger.info(f"[MOTION_FLOW] Free disk space: {free_mb:.1f} MB")  
        except Exception as e:
            logger.error(f"[MOTION_FLOW] Error checking disk space: {e}")
        
        try:
            # Use temporary file for video assembly to avoid SD card wear
            with tempfile.NamedTemporaryFile(suffix='.h264') as temp_video:
                # Use hardware encoding to create video
                event_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                event_id = event["id"]
                frame_count = len(event["frames"])
                
                logger.info(f"[MOTION_FLOW] {event_time} Encoding video for event {event_id} ({frame_count} frames)")
                self._encode_video(event["frames"], temp_video.name)
                logger.info(f"[MOTION_FLOW] {event_time} Video encoding complete for event {event_id} - temp file: {temp_video.name}")
                
                # Generate thumbnails if enabled
                thumbnails = []
                if self.config.generate_thumbnails:
                    logger.info(f"[MOTION_FLOW] {event_time} Generating thumbnails for event {event_id}")
                    thumbnails = self._generate_thumbnails(event["frames"])
                    logger.info(f"[MOTION_FLOW] {event_time} Generated {len(thumbnails)} thumbnails for event {event_id}")
                
                # Process optical flow if available
                motion_classification = None
                motion_signature = None

                if _optical_flow_analyzer is not None:
                    try:
                        logger.info(f"[OPTICAL_FLOW] Processing optical flow for event {event['id']}")

                        # Generate motion signature from accumulated flow history
                        motion_signature = _optical_flow_analyzer.generate_motion_signature()

                        if motion_signature:
                            # Classify the motion pattern
                            motion_classification = _optical_flow_analyzer.classify_motion(motion_signature)

                            logger.info(f"[OPTICAL_FLOW] Event {event['id']} classified as: "
                                      f"{motion_classification['label']} "
                                      f"(confidence: {motion_classification['confidence']:.2f})")

                            # Save to pattern database if enabled
                            if (self.config.store_optical_flow_data and
                                _motion_pattern_db is not None and
                                motion_classification['confidence'] >= self.config.min_classification_confidence):

                                _motion_pattern_db.add_pattern(
                                    event["id"],
                                    motion_signature,
                                    motion_classification,
                                    {
                                        "event_id": event["id"],
                                        "duration": (event["end_time"] - event["start_time"]).total_seconds(),
                                        "time_of_day": event["start_time"].hour,
                                        "frame_count": len(event["frames"])
                                    }
                                )
                                logger.info(f"[OPTICAL_FLOW] Pattern saved to database for event {event['id']}")

                            # Save flow visualization if enabled
                            if self.config.save_flow_visualizations:
                                try:
                                    # Get a representative frame
                                    if event["frames"] and len(_optical_flow_analyzer.flow_history) > 0:
                                        frame, _ = event["frames"][-1]
                                        last_flow = _optical_flow_analyzer.flow_history[-1]

                                        # Generate visualization
                                        flow_vis = _optical_flow_analyzer.visualize_flow(frame, last_flow)

                                        # Save to event directory
                                        flow_vis_path = os.path.join(event_dir, "flow_visualization.jpg")
                                        cv2.imwrite(flow_vis_path, flow_vis)
                                        logger.info(f"[OPTICAL_FLOW] Flow visualization saved for event {event['id']}")
                                except Exception as e:
                                    logger.error(f"[OPTICAL_FLOW] Error saving flow visualization: {e}")

                        # Reset analyzer for next event
                        _optical_flow_analyzer.reset()

                    except Exception as e:
                        logger.error(f"[OPTICAL_FLOW] Error processing optical flow: {e}")
                        import traceback
                        logger.error(traceback.format_exc())

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

                # Add motion analysis if available
                if motion_classification and motion_signature:
                    metadata["motion_analysis"] = {
                        "classification": motion_classification,
                        "motion_characteristics": motion_signature.get('statistical_features', {}),
                        "temporal_features": motion_signature.get('temporal_features', {}),
                        "signature_hash": hashlib.md5(
                            motion_signature['histogram_features'].tobytes()
                        ).hexdigest() if 'histogram_features' in motion_signature else None
                    }
                
                # Write metadata file
                metadata_path = os.path.join(event_dir, "metadata.json")
                event_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                event_id = event["id"]
                
                logger.info(f"[MOTION_FLOW] {event_time} Writing metadata for event {event_id} to {metadata_path}")
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                    
                # Verify metadata was written
                if os.path.exists(metadata_path):
                    logger.info(f"[MOTION_FLOW] {event_time} Metadata file created successfully for event {event_id}")
                    # Check file size
                    metadata_size = os.path.getsize(metadata_path)
                    logger.info(f"[MOTION_FLOW] {event_time} Metadata file size: {metadata_size} bytes")
                else:
                    logger.error(f"[MOTION_FLOW] {event_time} Failed to create metadata file for event {event_id}!")
                    
                # Save thumbnails
                if thumbnails:
                    thumbnails_dir = os.path.join(event_dir, "thumbnails")
                    os.makedirs(thumbnails_dir, exist_ok=True)
                    
                    for i, thumbnail in enumerate(thumbnails):
                        thumbnail_path = os.path.join(thumbnails_dir, f"thumb_{i}.jpg")
                        cv2.imwrite(thumbnail_path, thumbnail)
                    
                # Copy video to final location
                final_video_path = os.path.join(event_dir, "motion.mp4")
                event_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                event_id = event["id"]
                
                logger.info(f"[MOTION_FLOW] {event_time} Copying video for event {event_id} from {temp_video.name} to {final_video_path}")
                shutil.copy(temp_video.name, final_video_path)
                
                # Verify video was copied
                if os.path.exists(final_video_path):
                    video_size = os.path.getsize(final_video_path)
                    logger.info(f"[MOTION_FLOW] {event_time} Video copied successfully for event {event_id}. Size: {video_size/1024:.1f} KB")
                else:
                    logger.error(f"[MOTION_FLOW] {event_time} Failed to copy video file for event {event_id}!")
                
            # Notify transfer manager
            event_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            event_id = event["id"]
            logger.info(f"[MOTION_FLOW] {event_time} Notifying transfer manager about event {event_id}")
            transfer_manager.add_event(event_id)
            logger.info(f"[MOTION_FLOW] {event_time} Event {event_id} added to transfer manager's queue")
            
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
        event_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        if event_id in self.pending_events:
            logger.info(f"[TRANSFER_FLOW] {event_time} Event {event_id} already in pending events - skipping")
            return
            
        self.pending_events.add(event_id)
        logger.info(f"[TRANSFER_FLOW] {event_time} Added event {event_id} to pending events set. Size now: {len(self.pending_events)}")
        
        # Check if event directory exists
        event_dir = os.path.join(self.config.local_storage_path, event_id)
        if not os.path.exists(event_dir):
            logger.error(f"[TRANSFER_FLOW] {event_time} Event directory doesn't exist: {event_dir}")
            # List parent directory contents
            parent_dir = os.path.dirname(event_dir)
            if os.path.exists(parent_dir):
                logger.info(f"[TRANSFER_FLOW] {event_time} Parent directory exists. Contents: {os.listdir(parent_dir)}")
            else:
                logger.error(f"[TRANSFER_FLOW] {event_time} Parent directory doesn't exist: {parent_dir}")
            return
            
        # Check if transfer is enabled
        if self.config.upload_throttle_kbps == 0:
            logger.info(f"[TRANSFER_FLOW] {event_time} Transfers disabled (upload_throttle_kbps=0) for event {event_id}")
            return
            
        # Check if we should transfer now or during scheduled time
        current_hour = datetime.datetime.now().hour
        in_transfer_window = (self.config.transfer_schedule_start <= current_hour < 
                             self.config.transfer_schedule_end)
        
        logger.info(f"[TRANSFER_FLOW] {event_time} Transfer schedule active: {self.config.transfer_schedule_active}, ")
        logger.info(f"[TRANSFER_FLOW] {event_time} Current hour: {current_hour}, Window: {self.config.transfer_schedule_start}-{self.config.transfer_schedule_end}")
        logger.info(f"[TRANSFER_FLOW] {event_time} In transfer window: {in_transfer_window}")
                             
        if not self.config.transfer_schedule_active or in_transfer_window:
            # Immediate transfer with normal priority
            self.transfer_queue.put((50, event_id))
            logger.info(f"[TRANSFER_FLOW] {event_time} Added event {event_id} to transfer queue (immediate)")
        else:
            # Scheduled transfer with low priority
            self.transfer_queue.put((100, event_id))
            logger.info(f"[TRANSFER_FLOW] {event_time} Added event {event_id} to transfer queue (scheduled)")
            
        logger.info(f"[TRANSFER_FLOW] {event_time} Transfer queue size now: {self.transfer_queue.qsize()}")
            
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
                # Use requests library with proper resource management
                with ExitStack() as stack:
                    # Open all files in context manager
                    files = {}
                    files['video'] = stack.enter_context(open(video_path, 'rb'))

                    # Add thumbnails if available
                    thumbnails_dir = os.path.join(event_dir, "thumbnails")
                    if os.path.exists(thumbnails_dir):
                        for i, thumb_file in enumerate(os.listdir(thumbnails_dir)):
                            if thumb_file.endswith('.jpg'):
                                thumb_path = os.path.join(thumbnails_dir, thumb_file)
                                files[f'thumbnail_{i}'] = stack.enter_context(
                                    open(thumb_path, 'rb')
                                )

                    # All files guaranteed to close when exiting this block
                    logger.info(f"Uploading event {event_id}")
                    response = requests.post(
                        self.config.remote_storage_url,
                        files=files,
                        data={'metadata': json.dumps(metadata)},
                        headers={'X-API-Key': self.config.remote_api_key}
                    )

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
    
    # Log key information about the function's inputs to help diagnose connection issues
    logger.debug(f"[CRITICAL_DEBUG] modify_frame_buffer_write called with:")
    logger.debug(f"[CRITICAL_DEBUG]   - original_write_method: {original_write_method}")
    logger.debug(f"[CRITICAL_DEBUG]   - stream_buffer_instance: {stream_buffer_instance}")
    if stream_buffer_instance:
        logger.debug(f"[CRITICAL_DEBUG]   - stream_buffer attributes: {dir(stream_buffer_instance)}")
        if hasattr(stream_buffer_instance, '_original_write'):
            logger.debug(f"[CRITICAL_DEBUG]   - _original_write exists on stream_buffer")
        else:
            logger.error(f"[CRITICAL_DEBUG]   - ERROR: _original_write MISSING on stream_buffer!")
    else:
        logger.error(f"[CRITICAL_DEBUG]   - ERROR: stream_buffer_instance is None!")
    
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
        
        # Add call logging - timestamp in milliseconds for precise tracking
        call_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        logger.debug(f"[CRITICAL_DEBUG] {call_time} write_wrapper called")
        logger.debug(f"[CRITICAL_DEBUG] {call_time} self_or_frame type: {type(self_or_frame)}")
        logger.debug(f"[CRITICAL_DEBUG] {call_time} args: {args}")
        logger.debug(f"[CRITICAL_DEBUG] {call_time} stream_buffer: {stream_buffer is not None}")
        logger.debug(f"[CRITICAL_DEBUG] {call_time} motion_detected: {motion_detected}")
        
        # Detect calling pattern and adapt
        if hasattr(self_or_frame, 'raw_frame'):  # It's a direct method call
            logger.debug(f"[CRITICAL_DEBUG] {call_time} Direct method call detected")
            # This is a direct call with self as first arg
            instance = self_or_frame
            buf = args[0] if args else kwargs.get('buf')
            
            # Verify instance has _original_write
            if not hasattr(instance, '_original_write'):
                logger.error(f"[CRITICAL_DEBUG] {call_time} ERROR: instance missing _original_write attribute!")
                logger.error(f"[CRITICAL_DEBUG] {call_time} instance attributes: {dir(instance)}")
                # Fallback to original method directly
                logger.error(f"[CRITICAL_DEBUG] {call_time} Falling back to original_write_method directly")
                return original_write_method(instance, buf, *args[1:], **kwargs)
                
            # Call the original method through the instance's _original_write attribute
            logger.debug(f"[CRITICAL_DEBUG] {call_time} Calling instance._original_write")
            result = instance._original_write(buf, *args[1:], **kwargs)
        else:  # It's called from PiCamera2 FileOutput._write
            logger.debug(f"[CRITICAL_DEBUG] {call_time} PiCamera2 FileOutput call detected")
            # In this case, self_or_frame is actually the frame data
            buf = self_or_frame
            
            # Check if stream_buffer is available
            if stream_buffer is None:
                logger.error(f"[CRITICAL_DEBUG] {call_time} ERROR: stream_buffer is None!")
                logger.error(f"[CRITICAL_DEBUG] {call_time} Cannot process PiCamera2 frame - no stream buffer!")
                # We need to return something - use original method if possible
                if original_write_method:
                    logger.error(f"[CRITICAL_DEBUG] {call_time} Falling back to original_write_method")
                    # This will likely fail without proper 'self'
                    return original_write_method(buf)
                return len(buf)  # Last resort fallback
                
            # Verify stream_buffer has _original_write
            if not hasattr(stream_buffer, '_original_write'):
                logger.error(f"[CRITICAL_DEBUG] {call_time} ERROR: stream_buffer missing _original_write attribute!")
                logger.error(f"[CRITICAL_DEBUG] {call_time} stream_buffer attributes: {dir(stream_buffer)}")
                # Try original method as fallback
                logger.error(f"[CRITICAL_DEBUG] {call_time} Falling back to original method")
                if original_write_method:
                    return original_write_method(stream_buffer, buf)
                return len(buf)  # Last resort fallback
                
            # Use the saved original method from the stream buffer instance (from FastAPI)
            logger.debug(f"[CRITICAL_DEBUG] {call_time} Calling stream_buffer._original_write")
            result = stream_buffer._original_write(buf)
            
        # Add detailed frame processing logging
        call_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        try:
            # Now process the frame using the stream_buffer (the actual FastAPI FrameBuffer)
            if stream_buffer is None:
                logger.error(f"[CRITICAL_DEBUG] {call_time} Cannot process frame - stream_buffer is None")
            elif not hasattr(stream_buffer, 'raw_frame'):
                logger.error(f"[CRITICAL_DEBUG] {call_time} stream_buffer missing raw_frame attribute")
                logger.error(f"[CRITICAL_DEBUG] {call_time} stream_buffer attributes: {dir(stream_buffer)}")
            elif stream_buffer.raw_frame is None:
                logger.error(f"[CRITICAL_DEBUG] {call_time} stream_buffer.raw_frame is None")
            else:
                # Log successful frame acquisition
                logger.debug(f"[CRITICAL_DEBUG] {call_time} Got valid raw_frame from stream_buffer")
                logger.debug(f"[CRITICAL_DEBUG] {call_time} Frame shape: {stream_buffer.raw_frame.shape}")
                
                # Check frame_buffer (CircularFrameBuffer) exists
                if frame_buffer is None:
                    logger.error(f"[CRITICAL_DEBUG] {call_time} CircularFrameBuffer (frame_buffer) is None!")
                else:
                    # Log CircularFrameBuffer status before adding frame
                    if hasattr(frame_buffer, 'buffer'):
                        buffer_size = len(frame_buffer.buffer) if hasattr(frame_buffer.buffer, '__len__') else 'unknown'
                        logger.debug(f"[CRITICAL_DEBUG] {call_time} CircularFrameBuffer size before add: {buffer_size}")
                    
                    # Add the frame to our CircularFrameBuffer for motion detection
                    logger.debug(f"[CRITICAL_DEBUG] {call_time} Adding frame to CircularFrameBuffer")
                    frame_buffer.add_frame(stream_buffer.raw_frame.copy(), datetime.datetime.now())
                    logger.debug(f"[CRITICAL_DEBUG] {call_time} Added frame to CircularFrameBuffer successfully")
                    
                    # Handle motion recording if motion is detected
                    if motion_detected:
                        logger.debug(f"[CRITICAL_DEBUG] {call_time} Motion detected, regions: {len(motion_regions)}")
                        if not motion_recorder.recording:
                            logger.debug(f"[CRITICAL_DEBUG] {call_time} Starting motion recording")
                            motion_recorder.start_recording(motion_regions)
                        logger.debug(f"[CRITICAL_DEBUG] {call_time} Adding frame to active motion recording")
                        motion_recorder.add_frame(stream_buffer.raw_frame.copy(), motion_regions)
                    elif motion_recorder.recording:
                        logger.debug(f"[CRITICAL_DEBUG] {call_time} No motion detected, stopping recording")
                        motion_recorder.stop_recording()
        except Exception as e:
            logger.error(f"[CRITICAL_DEBUG] {call_time} Error in frame buffer integration: {e}")
            # Add stack trace for detailed error info
            import traceback
            logger.error(f"[CRITICAL_DEBUG] {call_time} Error traceback: {traceback.format_exc()}")
            # Also log the state of relevant objects
            logger.error(f"[CRITICAL_DEBUG] {call_time} stream_buffer exists: {stream_buffer is not None}")
            logger.error(f"[CRITICAL_DEBUG] {call_time} frame_buffer exists: {frame_buffer is not None}")
            logger.error(f"[CRITICAL_DEBUG] {call_time} motion_recorder exists: {motion_recorder is not None}")
            
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
        'shutdown': shutdown,  # Add the shutdown function to the returned resources
        'set_optical_flow_components': set_optical_flow_components  # Export optical flow setter
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

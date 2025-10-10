#!/usr/bin/python3

"""
FastAPI MJPEG Streaming Server with Motion Detection Storage

This script provides a robust implementation of an MJPEG streaming
server using FastAPI with advanced motion detection, storage and
network transfer capabilities.
"""

import argparse
import asyncio
import cv2
import datetime
import io
import json
import logging
import numpy as np
import os
import random
import signal
import sqlite3
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import partial
from threading import Condition, Thread
from typing import Dict, List, Optional, Tuple

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator

# Import centralized configuration
from config import get_config, WildlifeCameraConfig
from cli import load_config_with_cli

# Import motion storage module
from motion_storage import initialize as init_motion_storage

# Import optical flow analyzer
from optical_flow_analyzer import OpticalFlowAnalyzer, MotionPatternDatabase

# Import picamera2 with error handling
try:
    from picamera2 import Picamera2
    from picamera2.encoders import MJPEGEncoder
    from picamera2.outputs import FileOutput
except ImportError:
    print("Error: picamera2 module not found. Please install it first.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("fastapi_mjpeg_server")

# Global configuration (initialized in main)
# Note: We maintain these as separate variables for backward compatibility
# They will point to the centralized config sections
config = None  # type: WildlifeCameraConfig
camera_config = None  # Points to config.camera
storage_config = None  # Points to config.storage
active_connections: List[str] = []
active_connections_lock = threading.Lock()
camera_initialized = False
shutdown_event = asyncio.Event()

# Thread-safe motion state
class ThreadSafeMotionState:
    """Thread-safe container for motion detection state"""
    def __init__(self):
        self.lock = threading.Lock()
        self.prev_frame = None
        self.detected = False
        self.regions = []
        self.history = []

    def update_detection(self, detected: bool, regions: List[Tuple[int, int, int, int]],
                        classification=None):
        """Update motion detection state (called from camera thread)"""
        with self.lock:
            self.detected = detected
            self.regions = regions[:]  # Copy list
            if detected:
                self.history.append((datetime.datetime.now(), regions[:], classification))
                # Trim history if needed
                max_size = camera_config.motion_history_size * 2 if camera_config else 100
                if len(self.history) > max_size:
                    self.history = self.history[-max_size:]

    def update_prev_frame(self, frame):
        """Update previous frame (called from camera thread)"""
        with self.lock:
            self.prev_frame = frame

    def get_prev_frame(self):
        """Get previous frame copy (called from camera thread)"""
        with self.lock:
            return self.prev_frame

    def get_status(self):
        """Get current motion status (called from API threads)"""
        with self.lock:
            return self.detected, self.regions[:], self.history[:]

    def get_detection_state(self):
        """Get just detection flag and regions (called from camera thread)"""
        with self.lock:
            return self.detected, self.regions[:]

# Initialize thread-safe motion state
motion_state = ThreadSafeMotionState()

# Optical flow variables
optical_flow_analyzer: Optional[OpticalFlowAnalyzer] = None
motion_pattern_db: Optional[MotionPatternDatabase] = None

# Frame buffer with thread-safe access
class FrameBuffer(io.BufferedIOBase):
    def __init__(self, max_size: int = 5):
        super().__init__()
        self.frame = None
        self.condition = Condition()
        self.last_access_times: Dict[str, float] = {}
        self.max_size = max_size
        self.raw_frame = None  # Store the raw frame for processing
        self.last_frame = None  # Store last frame for optical flow analysis
        self.frame_index = 0  # Track frame number for skip logic

    def write(self, buf, *args, **kwargs):
        """
        Write a new frame to the buffer.
        This method accepts variadic arguments to handle both direct calls and calls
        from PiCamera2's FileOutput class.
        """
        with self.condition:
            # Store the original buffer
            original_buf = buf

            # Convert buffer to cv2 image for processing
            try:
                # Convert buffer to numpy array
                np_arr = np.frombuffer(buf, np.uint8)
                raw_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                self.raw_frame = raw_img.copy()

                # Process frame for motion detection if enabled
                if camera_config.motion_detection and raw_img is not None:
                    # Store previous frame for optical flow
                    prev_frame_for_flow = self.last_frame
                    self.last_frame = raw_img.copy()
                    self.frame_index += 1

                    # Detect motion with optical flow (updates motion_state internally)
                    motion_detected_local, motion_regions_local, flow_features = detect_motion(
                        raw_img, prev_frame_for_flow, self.frame_index)
                
                # Add timestamp if enabled
                if camera_config.show_timestamp:
                    img_with_timestamp = add_timestamp(raw_img)

                    # Add motion indicators if motion detected and highlighting enabled
                    if camera_config.motion_detection and camera_config.highlight_motion:
                        # Get current detection state for highlighting
                        detected, regions = motion_state.get_detection_state()
                        if detected:
                            for x, y, w, h in regions:
                                cv2.rectangle(img_with_timestamp, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            
                    # Re-encode the modified image back to JPEG
                    _, processed_buf = cv2.imencode('.jpg', img_with_timestamp)
                    buf = processed_buf.tobytes()
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                # If processing fails, use the original frame
                buf = original_buf
            
            self.frame = buf
            self.condition.notify_all()
        return len(buf)
    
    def get_frame(self, client_id: str) -> bytes:
        with self.condition:
            self.last_access_times[client_id] = time.time()
            if self.frame is None:
                self.condition.wait(timeout=5.0)
            return self.frame
    
    def register_client(self, client_id: str) -> bool:
        """Register a client and check if max clients reached"""
        with self.condition:
            if len(self.last_access_times) >= camera_config.max_clients:
                # Remove inactive clients first
                current_time = time.time()
                inactive_clients = [
                    cid for cid, last_time in self.last_access_times.items()
                    if current_time - last_time > camera_config.client_timeout
                ]
                for cid in inactive_clients:
                    del self.last_access_times[cid]
                
                # Still too many clients?
                if len(self.last_access_times) >= camera_config.max_clients:
                    return False
                    
            self.last_access_times[client_id] = time.time()
            return True
            
    def unregister_client(self, client_id: str):
        """Unregister a client"""
        with self.condition:
            if client_id in self.last_access_times:
                del self.last_access_times[client_id]
                
    # Required methods to implement BufferedIOBase interface
    def readable(self):
        return False
        
    def writable(self):
        return True
        
    def seekable(self):
        return False
        
    def flush(self):
        pass

# Initialize the frame buffer
frame_buffer = FrameBuffer()

# Camera initialization with error handling
def initialize_camera():
    global camera_initialized
    
    try:
        picam2 = Picamera2()
        
        # Configure camera with error handling
        try:
            config = picam2.create_video_configuration(
                main={"size": (camera_config.width, camera_config.height)}
            )
            picam2.configure(config)
            
            # Configure encoder
            encoder = MJPEGEncoder()
            
            # Start recording
            picam2.start_recording(encoder, FileOutput(frame_buffer))
            logger.info("Camera initialized successfully")
            camera_initialized = True
            return picam2
        except Exception as e:
            logger.error(f"Camera configuration failed: {e}")
            raise RuntimeError(f"Failed to configure camera: {e}")
            
    except Exception as e:
        logger.error(f"Camera initialization failed: {e}")
        camera_initialized = False
        raise RuntimeError(f"Failed to initialize camera: {e}")

# Graceful shutdown handler
def handle_shutdown(picam2):
    async def cleanup():
        logger.info("Shutting down camera and server...")
        
        # First stop the camera recording
        if camera_initialized:
            try:
                picam2.stop_recording()
                logger.info("Camera recording stopped")
            except Exception as e:
                logger.error(f"Error stopping camera recording: {e}")
        
        # Signal the streaming shutdown event
        shutdown_event.set()
        
        # Shutdown the motion storage module if it's initialized
        if 'motion_storage' in globals() and motion_storage is not None:
            try:
                if 'shutdown' in motion_storage:
                    motion_storage['shutdown']()
                    logger.info("Motion storage module shutdown completed")
            except Exception as e:
                logger.error(f"Error shutting down motion storage: {e}")
        
        # Force exit after everything is cleaned up
        logger.info("Shutdown completed, exiting process")
        # Use a small delay to allow logs to be written
        await asyncio.sleep(0.5)
        os._exit(0)  # Force exit the process
    
    return cleanup

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize camera when app starts
    try:
        picam2 = initialize_camera()
        
        # Initialize motion storage
        global motion_storage
        # Pass both camera_config AND storage_config to the motion_storage module
        logger.info("[MOTION_FLOW] Initializing motion_storage module")
        logger.info(f"[MOTION_FLOW] Storage config: path={storage_config.local_storage_path}, max_size={storage_config.max_disk_usage_mb}MB, upload={not storage_config.upload_throttle_kbps == 0}")
        motion_storage = init_motion_storage(app, camera_config, storage_config)
        logger.info("[MOTION_FLOW] Motion storage module initialized")

        
        # Patch the frame buffer write method
        # We need to patch it in a way that works with both direct calls and calls through PiCamera2
        original_write = frame_buffer.write

        # Store the original write method on the instance for access by the wrapper
        frame_buffer._original_write = original_write

        # Create the patched write method, passing our stream buffer instance explicitly
        patched_write = motion_storage['modify_frame_buffer_write'](original_write, frame_buffer)

        # Replace the method with our patched version
        frame_buffer.write = patched_write

        logger.info("Successfully patched frame buffer write method")

        # Initialize optical flow analyzer and pattern database
        global optical_flow_analyzer, motion_pattern_db, config
        if config.optical_flow_storage.classification_enabled and camera_config.optical_flow_enabled:
            logger.info("Initializing optical flow analyzer")

            # Create optical flow configuration
            flow_config = {
                'feature_params': {
                    'maxCorners': camera_config.optical_flow_feature_max,
                    'qualityLevel': camera_config.optical_flow_quality_level,
                    'minDistance': camera_config.optical_flow_min_distance,
                    'blockSize': 7
                },
                'grid_size': camera_config.optical_flow_grid_size,
                'direction_bins': camera_config.optical_flow_direction_bins,
                'frame_history': 10,  # Keep last 10 frames for signature generation
            }

            optical_flow_analyzer = OpticalFlowAnalyzer(config=flow_config)

            # Create signature directory if it doesn't exist
            signature_dir = os.path.join(storage_config.local_storage_path,
                                        config.optical_flow_storage.signature_dir)
            os.makedirs(signature_dir, exist_ok=True)

            # Initialize motion pattern database
            db_path = os.path.join(storage_config.local_storage_path,
                                  config.optical_flow_storage.database_path)

            motion_pattern_db = MotionPatternDatabase(db_path=db_path,
                                                     signature_dir=signature_dir)

            logger.info(f"Optical flow analyzer initialized with config: {flow_config}")
            logger.info(f"Motion pattern database at: {db_path}")

            # Pass optical flow components to motion_storage module
            motion_storage['set_optical_flow_components'](optical_flow_analyzer, motion_pattern_db)
        else:
            logger.info("Optical flow analysis disabled by configuration")

        # Register signal handlers for graceful shutdown
        for sig in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_event_loop().add_signal_handler(
                sig, lambda: asyncio.create_task(handle_shutdown(picam2)())
            )
        
        yield

        # Cleanup optical flow resources
        if optical_flow_analyzer is not None:
            logger.info("Cleaning up optical flow analyzer")
            optical_flow_analyzer.reset()

        # Cleanup resources when app shuts down
        await handle_shutdown(picam2)()

    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

# Initialize FastAPI
app = FastAPI(
    title="Raspberry Pi Camera MJPEG Server",
    description="A FastAPI-based MJPEG streaming server for Raspberry Pi cameras",
    version="1.0.0",
    lifespan=lifespan
)

# API for camera configuration updates
class CameraConfigUpdate(BaseModel):
    width: Optional[int] = Field(None, gt=0, le=3840)
    height: Optional[int] = Field(None, gt=0, le=2160)
    frame_rate: Optional[int] = Field(None, gt=0, le=120)
    rotation: Optional[int] = Field(None, ge=0, le=270)
    max_clients: Optional[int] = Field(None, gt=0, le=100)
    client_timeout: Optional[int] = Field(None, gt=0, le=600)
    show_timestamp: Optional[bool] = None
    timestamp_position: Optional[str] = None
    motion_detection: Optional[bool] = None
    motion_threshold: Optional[int] = Field(None, ge=5, le=100)
    motion_min_area: Optional[int] = Field(None, ge=100, le=10000)
    highlight_motion: Optional[bool] = None
    
    @field_validator('rotation')
    def validate_rotation(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v not in (0, 90, 180, 270):
            raise ValueError('Rotation must be 0, 90, 180, or 270 degrees')
        return v
        
    @field_validator('timestamp_position')
    def validate_timestamp_position(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ('top-left', 'top-right', 'bottom-left', 'bottom-right'):
            raise ValueError('Timestamp position must be one of: top-left, top-right, bottom-left, bottom-right')
        return v

# HTML template for video display
# HTML template for video display - using double braces to escape CSS curly braces
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Raspberry Pi Camera Stream</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            text-align: center;
            background-color: #f0f0f0;
        }}
        h1 {{
            color: #333;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }}
        img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }}
        .status {{
            margin-top: 15px;
            padding: 10px;
            background-color: #e8f4fd;
            border-radius: 4px;
            font-size: 14px;
            display: flex;
            justify-content: space-between;
        }}
        .status-left {{
            text-align: left;
        }}
        .status-right {{
            text-align: right;
        }}
        .error {{
            color: red;
            display: none;
        }}
        .motion-alert {{
            background-color: #ffe8e8;
            color: #d32f2f;
            padding: 8px;
            border-radius: 4px;
            margin-top: 10px;
            display: none;
            font-weight: bold;
            border-left: 4px solid #d32f2f;
        }}
        .motion-history {{
            margin-top: 20px;
            padding: 10px;
            background-color: #f5f5f5;
            border-radius: 4px;
            font-size: 14px;
            text-align: left;
            max-height: 150px;
            overflow-y: auto;
            border: 1px solid #ddd;
        }}
        .motion-history h3 {{
            margin-top: 0;
            margin-bottom: 10px;
            font-size: 16px;
        }}
        .motion-event {{
            padding: 5px;
            border-bottom: 1px solid #ddd;
        }}
        .classification-badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 12px;
            font-weight: bold;
            color: white;
        }}
        .classification-vehicle {{
            background-color: #2196F3;  /* Blue */
        }}
        .classification-person {{
            background-color: #4CAF50;  /* Green */
        }}
        .classification-animal {{
            background-color: #FF9800;  /* Orange */
        }}
        .classification-environment {{
            background-color: #9E9E9E;  /* Gray */
        }}
        .classification-unknown {{
            background-color: #757575;  /* Dark Gray */
        }}
        .config-panel {{
            margin-top: 20px;
            text-align: left;
            padding: 15px;
            background-color: #f5f5f5;
            border-radius: 4px;
        }}
        .config-panel h3 {{
            margin-top: 0;
        }}
        .form-group {{
            margin-bottom: 10px;
        }}
        .form-group label {{
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }}
        .form-group input {{
            padding: 5px;
            width: 100px;
        }}
        .form-row {{
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }}
        .form-row label {{
            margin-right: 10px;
            margin-bottom: 0;
        }}
        button {{
            background-color: #4CAF50;
            color: white;
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }}
        button:hover {{
            background-color: #45a049;
        }}
        .toggle-panel {{
            text-align: right;
            margin-top: 10px;
        }}
        .storage-stats {{
            margin-top: 20px;
            padding: 10px;
            background-color: #f9f9f9;
            border-radius: 4px;
            font-size: 14px;
            border: 1px solid #ddd;
            display: none;
        }}
        .storage-stats h3 {{
            margin-top: 0;
            margin-bottom: 10px;
        }}
        .progress-bar {{
            width: 100%;
            height: 20px;
            background-color: #e0e0e0;
            border-radius: 10px;
            margin: 10px 0;
        }}
        .progress-bar-fill {{
            height: 100%;
            background-color: #4CAF50;
            border-radius: 10px;
            text-align: center;
            color: white;
            font-size: 12px;
            line-height: 20px;
        }}
        .storage-events {{
            max-height: 200px;
            overflow-y: auto;
            font-size: 12px;
        }}
        .storage-event {{
            padding: 5px;
            border-bottom: 1px solid #eee;
        }}
        .storage-event-pending {{
            color: #ff9800;
        }}
        .storage-event-active {{
            color: #2196F3;
        }}
        .pattern-panel {{
            margin-top: 20px;
            padding: 15px;
            background-color: #f5f5f5;
            border-radius: 4px;
            text-align: left;
        }}
        .pattern-panel h3 {{
            margin-top: 0;
        }}
        .pattern-controls {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 15px;
        }}
        .pattern-stats {{
            padding: 10px;
            background-color: white;
            border-radius: 4px;
            margin-bottom: 15px;
            font-size: 14px;
        }}
        .pattern-list {{
            max-height: 400px;
            overflow-y: auto;
            background-color: white;
            border-radius: 4px;
            padding: 10px;
        }}
        .pattern-item {{
            padding: 12px;
            margin-bottom: 10px;
            background-color: #fafafa;
            border: 1px solid #ddd;
            border-radius: 4px;
            transition: background-color 0.2s;
        }}
        .pattern-item:hover {{
            background-color: #f0f0f0;
        }}
        .pattern-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }}
        .pattern-id {{
            font-weight: bold;
            color: #333;
            font-size: 14px;
        }}
        .pattern-actions {{
            display: flex;
            gap: 5px;
        }}
        .pattern-actions button {{
            padding: 4px 8px;
            font-size: 12px;
        }}
        .btn-relabel {{
            background-color: #2196F3;
        }}
        .btn-relabel:hover {{
            background-color: #1976D2;
        }}
        .btn-similar {{
            background-color: #FF9800;
        }}
        .btn-similar:hover {{
            background-color: #F57C00;
        }}
        .btn-delete {{
            background-color: #f44336;
        }}
        .btn-delete:hover {{
            background-color: #d32f2f;
        }}
        .pattern-details {{
            font-size: 13px;
            color: #666;
        }}
        .pattern-confidence {{
            font-weight: bold;
        }}
        .pagination {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 15px;
            margin-top: 15px;
        }}
        .pagination button:disabled {{
            background-color: #ccc;
            cursor: not-allowed;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Raspberry Pi Camera Stream</h1>
        <div>
            <img src="/stream" alt="Camera Stream">
        </div>
        <div class="status">
            <div class="status-left">
                Resolution: {width}x{height}, Frame Rate: {frame_rate}fps
            </div>
            <div class="status-right">
                <span id="connectionCount">Connections: {connection_count}</span>
            </div>
        </div>
        <div class="motion-alert" id="motionAlert">
            Motion Detected!
        </div>
        <div class="error" id="streamError">
            Stream disconnected. Attempting to reconnect...
        </div>
        
        <div class="toggle-panel">
            <button id="toggleConfig">Show Configuration</button>
            <button id="toggleHistory">Show Motion History</button>
            <button id="toggleStorage">Show Storage Stats</button>
            <button id="togglePatterns">Show Motion Patterns</button>
        </div>
        
        <div class="motion-history" id="motionHistory" style="display: none;">
            <h3>Motion Detection History</h3>
            <div id="motionEvents"></div>
        </div>
        
        <div class="storage-stats" id="storageStats">
            <h3>Storage Statistics</h3>
            <div class="progress-bar">
                <div id="storageUsage" class="progress-bar-fill" style="width: 0%">0%</div>
            </div>
            <div id="storageDetails">Loading...</div>
            <h4>Pending Transfers</h4>
            <div id="pendingEvents" class="storage-events">Loading...</div>
        </div>
        
        <div class="config-panel" id="configPanel" style="display: none;">
            <h3>Camera Configuration</h3>
            <div class="form-row">
                <label>
                    <input type="checkbox" id="showTimestamp" {timestamp_checked}>
                    Show Timestamp
                </label>
            </div>
            <div class="form-row">
                <label>
                    <input type="checkbox" id="motionDetection" {motion_checked}>
                    Motion Detection
                </label>
            </div>
            <div class="form-row">
                <label>
                    <input type="checkbox" id="highlightMotion" {highlight_checked}>
                    Highlight Motion
                </label>
            </div>
            <div class="form-row">
                <label for="motionThreshold">Motion Sensitivity:</label>
                <input type="number" id="motionThreshold" min="5" max="100" value="{motion_threshold}">
            </div>
            <div class="form-row">
                <label for="timestampPosition">Timestamp Position:</label>
                <select id="timestampPosition">
                    <option value="top-left" {ts_pos_tl}>Top Left</option>
                    <option value="top-right" {ts_pos_tr}>Top Right</option>
                    <option value="bottom-left" {ts_pos_bl}>Bottom Left</option>
                    <option value="bottom-right" {ts_pos_br}>Bottom Right</option>
                </select>
            </div>
            <div class="form-row">
                <button id="saveConfig">Save Configuration</button>
            </div>
        </div>

        <div class="pattern-panel" id="patternPanel" style="display: none;">
            <h3>Motion Pattern Database</h3>
            <div class="pattern-controls">
                <label for="patternFilter">Filter by type:</label>
                <select id="patternFilter">
                    <option value="all">All Patterns</option>
                    <option value="vehicle">Vehicle</option>
                    <option value="person">Person</option>
                    <option value="animal">Animal</option>
                    <option value="environment">Environment</option>
                    <option value="unknown">Unknown</option>
                </select>
                <button id="refreshPatterns">Refresh</button>
            </div>
            <div id="patternStats" class="pattern-stats"></div>
            <div id="patternList" class="pattern-list">Loading patterns...</div>
            <div class="pagination">
                <button id="prevPage" disabled>Previous</button>
                <span id="pageInfo">Page 1</span>
                <button id="nextPage">Next</button>
            </div>
        </div>
    </div>
    
    <script>
        const img = document.querySelector('img');
        const errorDiv = document.getElementById('streamError');
        const motionAlert = document.getElementById('motionAlert');
        const motionEvents = document.getElementById('motionEvents');
        
        // Handle stream errors
        img.onerror = function() {{
            errorDiv.style.display = 'block';
            // Try to reconnect after 3 seconds
            setTimeout(() => {{
                img.src = `/stream?cache=${{new Date().getTime()}}`;
            }}, 3000);
        }};
        
        // Hide error when stream is working
        img.onload = function() {{
            errorDiv.style.display = 'none';
        }};
        
        // Toggle panels
        document.getElementById('toggleConfig').addEventListener('click', function() {{
            const panel = document.getElementById('configPanel');
            const isVisible = panel.style.display !== 'none';
            panel.style.display = isVisible ? 'none' : 'block';
            this.textContent = isVisible ? 'Show Configuration' : 'Hide Configuration';
        }});
        
        document.getElementById('toggleHistory').addEventListener('click', function() {{
            const panel = document.getElementById('motionHistory');
            const isVisible = panel.style.display !== 'none';
            panel.style.display = isVisible ? 'none' : 'block';
            this.textContent = isVisible ? 'Show Motion History' : 'Hide Motion History';
        }});
        
        document.getElementById('toggleStorage').addEventListener('click', function() {{
            const panel = document.getElementById('storageStats');
            const isVisible = panel.style.display !== 'none';
            panel.style.display = isVisible ? 'none' : 'block';
            this.textContent = isVisible ? 'Show Storage Stats' : 'Hide Storage Stats';
            
            if (!isVisible) {{
                updateStorageStats();
            }}
        }});
        
        // Save configuration
        document.getElementById('saveConfig').addEventListener('click', function() {{
            const config = {{
                show_timestamp: document.getElementById('showTimestamp').checked,
                timestamp_position: document.getElementById('timestampPosition').value,
                motion_detection: document.getElementById('motionDetection').checked,
                highlight_motion: document.getElementById('highlightMotion').checked,
                motion_threshold: parseInt(document.getElementById('motionThreshold').value, 10)
            }};
            
            fetch('/config', {{
                method: 'POST',
                headers: {{
                    'Content-Type': 'application/json',
                }},
                body: JSON.stringify(config),
            }})
            .then(response => response.json())
            .then(data => {{
                alert('Configuration saved!');
            }})
            .catch((error) => {{
                alert('Error saving configuration');
            }});
        }});
        
        // Periodically check for motion and update status
        setInterval(() => {{
            fetch('/motion_status')
                .then(response => response.json())
                .then(data => {{
                    // Update motion alert
                    motionAlert.style.display = data.motion_detected ? 'block' : 'none';
                    
                    // Update motion history
                    motionEvents.innerHTML = '';
                    data.motion_history.forEach(event => {{
                        const eventDiv = document.createElement('div');
                        eventDiv.className = 'motion-event';

                        // Build event text with classification if available
                        let eventText = `${{event.timestamp}} - ${{event.regions.length}} regions`;
                        if (event.classification) {{
                            const confidence = (event.classification.confidence * 100).toFixed(0);
                            eventText += ` - ${{event.classification.label}} (${{confidence}}%)`;

                            // Add badge styling based on classification
                            const badge = document.createElement('span');
                            badge.className = 'classification-badge classification-' + event.classification.label.toLowerCase();
                            badge.textContent = event.classification.label;
                            eventDiv.appendChild(document.createTextNode(`${{event.timestamp}} - ${{event.regions.length}} regions - `));
                            eventDiv.appendChild(badge);
                            eventDiv.appendChild(document.createTextNode(` (${{confidence}}%)`));
                        }} else {{
                            eventDiv.textContent = eventText;
                        }}

                        motionEvents.appendChild(eventDiv);
                    }});
                    
                    // Update connection count
                    document.getElementById('connectionCount').textContent = 
                        `Connections: ${{data.active_connections}}`;
                }});
        }}, 1000);
        
        // Update storage statistics
        function updateStorageStats() {{
            fetch('/storage/status')
                .then(response => response.json())
                .then(data => {{
                    // Update storage usage bar
                    const usagePercent = data.storage.usage_percent;
                    const usageBar = document.getElementById('storageUsage');
                    usageBar.style.width = `${{usagePercent}}%`;
                    usageBar.textContent = `${{usagePercent}}%`;
                    
                    // Set color based on usage
                    if (usagePercent < 70) {{
                        usageBar.style.backgroundColor = '#4CAF50';  // Green
                    }} else if (usagePercent < 90) {{
                        usageBar.style.backgroundColor = '#ff9800';  // Orange
                    }} else {{
                        usageBar.style.backgroundColor = '#f44336';  // Red
                    }}
                    
                    // Update details
                    const details = document.getElementById('storageDetails');
                    details.innerHTML = `
                        <p>Used: ${{data.storage.size_mb}} MB / ${{data.storage.max_size_mb}} MB</p>
                        <p>Events: ${{data.storage.event_count}}</p>
                        <p>Pending Transfers: ${{data.transfer.pending_count}}</p>
                        <p>Transfer Window: ${{data.transfer.schedule_active ? 
                            data.transfer.schedule_window : 'Always'}}</p>
                        ${{data.transfer.wifi_monitoring ? 
                            `<p>WiFi Signal: ${{data.transfer.wifi_monitoring.signal_strength}} dBm</p>
                             <p>Current Throttle: ${{data.transfer.wifi_monitoring.current_throttle}} KB/s</p>` : ''}}
                    `;
                    
                    // Update pending events
                    const pendingEvents = document.getElementById('pendingEvents');
                    if (data.pending_events && data.pending_events.length > 0) {{
                        pendingEvents.innerHTML = '';
                        data.pending_events.forEach(event => {{
                            const eventDiv = document.createElement('div');
                            eventDiv.className = 'storage-event';
                            if (event.id in data.transfer.active_transfers) {{
                                eventDiv.classList.add('storage-event-active');
                                eventDiv.innerHTML = `${{event.id}} - <strong>Uploading</strong> (${{event.duration.toFixed(1)}}s, ${{event.frame_count}} frames)`;
                            }} else {{
                                eventDiv.classList.add('storage-event-pending');
                                eventDiv.innerHTML = `${{event.id}} - <strong>Pending</strong> (${{event.duration.toFixed(1)}}s, ${{event.frame_count}} frames)`;
                            }}
                            pendingEvents.appendChild(eventDiv);
                        }});
                    }} else {{
                        pendingEvents.innerHTML = '<p>No pending transfers</p>';
                    }}
                }})
                .catch(error => {{
                    console.error('Error fetching storage stats:', error);
                }});
        }}
        
        // Update storage stats every 5 seconds if visible
        setInterval(() => {{
            const storageStats = document.getElementById('storageStats');
            if (storageStats.style.display !== 'none') {{
                updateStorageStats();
            }}
        }}, 5000);

        // Pattern management functionality
        let currentPage = 0;
        let currentFilter = 'all';
        const patternsPerPage = 20;

        document.getElementById('togglePatterns').addEventListener('click', function() {{
            const panel = document.getElementById('patternPanel');
            const isVisible = panel.style.display !== 'none';
            panel.style.display = isVisible ? 'none' : 'block';
            this.textContent = isVisible ? 'Show Motion Patterns' : 'Hide Motion Patterns';

            if (!isVisible) {{
                loadPatterns();
            }}
        }});

        document.getElementById('patternFilter').addEventListener('change', function() {{
            currentFilter = this.value;
            currentPage = 0;
            loadPatterns();
        }});

        document.getElementById('refreshPatterns').addEventListener('click', function() {{
            loadPatterns();
        }});

        document.getElementById('prevPage').addEventListener('click', function() {{
            if (currentPage > 0) {{
                currentPage--;
                loadPatterns();
            }}
        }});

        document.getElementById('nextPage').addEventListener('click', function() {{
            currentPage++;
            loadPatterns();
        }});

        function loadPatterns() {{
            const offset = currentPage * patternsPerPage;

            fetch(`/patterns?limit=${{patternsPerPage}}&offset=${{offset}}`)
                .then(response => response.json())
                .then(data => {{
                    displayPatterns(data);
                    updatePagination(data);
                    updatePatternStats(data);
                }})
                .catch(error => {{
                    console.error('Error loading patterns:', error);
                    document.getElementById('patternList').innerHTML =
                        '<p style="color: red;">Error loading patterns</p>';
                }});
        }}

        function displayPatterns(data) {{
            const patternList = document.getElementById('patternList');

            if (!data.patterns || data.patterns.length === 0) {{
                patternList.innerHTML = '<p>No patterns found</p>';
                return;
            }}

            // Filter patterns if needed
            let patterns = data.patterns;
            if (currentFilter !== 'all') {{
                patterns = patterns.filter(p => p.classification === currentFilter);
            }}

            patternList.innerHTML = '';
            patterns.forEach(pattern => {{
                const item = document.createElement('div');
                item.className = 'pattern-item';

                const confidence = (pattern.confidence * 100).toFixed(0);
                const metadata = pattern.metadata || {{}};

                item.innerHTML = `
                    <div class="pattern-header">
                        <span class="pattern-id">${{pattern.pattern_id}}</span>
                        <div class="pattern-actions">
                            <button class="btn-relabel" onclick="relabelPattern('${{pattern.pattern_id}}')">Relabel</button>
                            <button class="btn-similar" onclick="findSimilar('${{pattern.pattern_id}}')">Similar</button>
                            <button class="btn-delete" onclick="deletePattern('${{pattern.pattern_id}}')">Delete</button>
                        </div>
                    </div>
                    <div class="pattern-details">
                        <span class="classification-badge classification-${{pattern.classification}}">
                            ${{pattern.classification}}
                        </span>
                        <span class="pattern-confidence">Confidence: ${{confidence}}%</span>
                        <br>
                        <small>Created: ${{new Date(pattern.created_at).toLocaleString()}}</small>
                        ${{metadata.duration ? `<br><small>Duration: ${{metadata.duration.toFixed(1)}}s, Frames: ${{metadata.frame_count || 0}}</small>` : ''}}
                    </div>
                `;

                patternList.appendChild(item);
            }});
        }}

        function updatePagination(data) {{
            const totalPages = Math.ceil(data.total / patternsPerPage);
            document.getElementById('pageInfo').textContent = `Page ${{currentPage + 1}} of ${{totalPages}}`;
            document.getElementById('prevPage').disabled = currentPage === 0;
            document.getElementById('nextPage').disabled = currentPage >= totalPages - 1;
        }}

        function updatePatternStats(data) {{
            const stats = document.getElementById('patternStats');
            stats.innerHTML = `<strong>Total patterns: ${{data.total}}</strong>`;
        }}

        function relabelPattern(patternId) {{
            const newLabel = prompt('Enter new classification (vehicle/person/animal/environment/unknown):');
            if (!newLabel) return;

            const validLabels = ['vehicle', 'person', 'animal', 'environment', 'unknown'];
            if (!validLabels.includes(newLabel.toLowerCase())) {{
                alert('Invalid classification. Must be one of: vehicle, person, animal, environment, unknown');
                return;
            }}

            fetch(`/patterns/${{patternId}}/label`, {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{ classification: newLabel.toLowerCase() }})
            }})
            .then(response => response.json())
            .then(data => {{
                alert('Pattern relabeled successfully');
                loadPatterns();
            }})
            .catch(error => {{
                alert('Error relabeling pattern');
                console.error(error);
            }});
        }}

        function findSimilar(patternId) {{
            fetch(`/patterns/similar/${{patternId}}?limit=10&threshold=0.5`)
                .then(response => response.json())
                .then(data => {{
                    if (data.similar_patterns && data.similar_patterns.length > 0) {{
                        let message = `Found ${{data.similar_patterns.length}} similar patterns:\\n\\n`;
                        data.similar_patterns.forEach(p => {{
                            message += `${{p.pattern_id}}: ${{p.classification}} (similarity: ${{(p.similarity * 100).toFixed(0)}}%)\\n`;
                        }});
                        alert(message);
                    }} else {{
                        alert('No similar patterns found');
                    }}
                }})
                .catch(error => {{
                    alert('Error finding similar patterns');
                    console.error(error);
                }});
        }}

        function deletePattern(patternId) {{
            if (!confirm(`Delete pattern ${{patternId}}?`)) return;

            fetch(`/patterns/${{patternId}}`, {{
                method: 'DELETE'
            }})
            .then(response => response.json())
            .then(data => {{
                alert('Pattern deleted successfully');
                loadPatterns();
            }})
            .catch(error => {{
                alert('Error deleting pattern');
                console.error(error);
            }});
        }}
    </script>
</body>
</html>
"""

# Routes
@app.get("/", response_class=HTMLResponse)
async def index():
    # Set checkbox states based on current config
    timestamp_checked = "checked" if camera_config.show_timestamp else ""
    motion_checked = "checked" if camera_config.motion_detection else ""
    highlight_checked = "checked" if camera_config.highlight_motion else ""
    
    # Set selected state for timestamp position dropdown
    ts_pos = {
        "ts_pos_tl": "",
        "ts_pos_tr": "",
        "ts_pos_bl": "", 
        "ts_pos_br": ""
    }
    if camera_config.timestamp_position == "top-left":
        ts_pos["ts_pos_tl"] = "selected"
    elif camera_config.timestamp_position == "top-right":
        ts_pos["ts_pos_tr"] = "selected"
    elif camera_config.timestamp_position == "bottom-left":
        ts_pos["ts_pos_bl"] = "selected"
    else:  # bottom-right is default
        ts_pos["ts_pos_br"] = "selected"

    # Get connection count thread-safely
    with active_connections_lock:
        conn_count = len(active_connections)

    return HTML_TEMPLATE.format(
        width=camera_config.width,
        height=camera_config.height,
        frame_rate=camera_config.frame_rate,
        connection_count=conn_count,
        timestamp_checked=timestamp_checked,
        motion_checked=motion_checked,
        highlight_checked=highlight_checked,
        motion_threshold=camera_config.motion_threshold,
        ts_pos_tl=ts_pos["ts_pos_tl"],
        ts_pos_tr=ts_pos["ts_pos_tr"],
        ts_pos_bl=ts_pos["ts_pos_bl"],
        ts_pos_br=ts_pos["ts_pos_br"]
    )

@app.get("/stream")
async def video_feed(request: Request):
    """Generate MJPEG stream from the camera"""
    client_id = str(hash(str(request.client)))
    
    # Check for maximum clients
    if not frame_buffer.register_client(client_id):
        logger.warning(f"Maximum clients reached. Rejecting client {client_id}")
        raise HTTPException(
            status_code=503,
            detail="Server busy. Maximum number of clients reached."
        )
    
    with active_connections_lock:
        active_connections.append(client_id)
        conn_count = len(active_connections)
    logger.info(f"Client {client_id} connected. Active connections: {conn_count}")
    
    async def generate():
        try:
            while not shutdown_event.is_set():
                # Get frame with timeout
                frame = frame_buffer.get_frame(client_id)
                if frame is None:
                    await asyncio.sleep(0.1)
                    continue
                    
                # Generate MJPEG frame
                yield b'--frame\r\n'
                yield b'Content-Type: image/jpeg\r\n'
                yield f'Content-Length: {len(frame)}\r\n\r\n'.encode()
                yield frame
                yield b'\r\n'
                
                # Prevent CPU overuse
                await asyncio.sleep(1/camera_config.frame_rate)
        except Exception as e:
            logger.error(f"Streaming error for client {client_id}: {e}")
        finally:
            # Clean up client resources
            with active_connections_lock:
                if client_id in active_connections:
                    active_connections.remove(client_id)
                conn_count = len(active_connections)
            frame_buffer.unregister_client(client_id)
            logger.info(f"Client {client_id} disconnected. Active connections: {conn_count}")
    
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

@app.get("/status")
async def get_status():
    """Get server status information"""
    # Get motion state thread-safely
    detected, _, _ = motion_state.get_status()

    # Get connection count thread-safely
    with active_connections_lock:
        conn_count = len(active_connections)

    return {
        "active_connections": conn_count,
        "camera_initialized": camera_initialized,
        "camera_config": {
            "width": camera_config.width,
            "height": camera_config.height,
            "frame_rate": camera_config.frame_rate,
            "rotation": camera_config.rotation,
            "max_clients": camera_config.max_clients,
            "client_timeout": camera_config.client_timeout,
            "show_timestamp": camera_config.show_timestamp,
            "timestamp_position": camera_config.timestamp_position,
            "motion_detection": camera_config.motion_detection,
            "motion_threshold": camera_config.motion_threshold,
            "motion_min_area": camera_config.motion_min_area,
            "highlight_motion": camera_config.highlight_motion
        },
        "motion_detected": detected
    }

@app.get("/motion_status")
async def get_motion_status():
    """Get motion detection status with optical flow classification"""
    # Get motion state thread-safely
    detected, regions, history = motion_state.get_status()

    # Process motion history with classification info
    history_with_classification = []
    for item in history[-camera_config.motion_history_size:]:
        if len(item) == 3:  # New format: (timestamp, regions, classification)
            timestamp, regions, classification = item
            entry = {
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "regions": regions
            }
            if classification:
                entry["classification"] = classification
            history_with_classification.append(entry)
        else:  # Old format: (timestamp, regions)
            timestamp, regions = item
            history_with_classification.append({
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "regions": regions
            })

    # Get connection count thread-safely
    with active_connections_lock:
        conn_count = len(active_connections)

    return {
        "motion_detected": detected,
        "motion_history": history_with_classification,
        "active_connections": conn_count,
        "optical_flow_enabled": camera_config.optical_flow_enabled and optical_flow_analyzer is not None
    }

@app.post("/config")
async def update_config(config: CameraConfigUpdate):
    """Update camera configuration"""
    global camera_config

    if config.width is not None:
        camera_config.width = config.width
    if config.height is not None:
        camera_config.height = config.height
    if config.frame_rate is not None:
        camera_config.frame_rate = config.frame_rate
    if config.rotation is not None:
        camera_config.rotation = config.rotation
    if config.max_clients is not None:
        camera_config.max_clients = config.max_clients
    if config.client_timeout is not None:
        camera_config.client_timeout = config.client_timeout
    if config.show_timestamp is not None:
        camera_config.show_timestamp = config.show_timestamp
    if config.timestamp_position is not None:
        camera_config.timestamp_position = config.timestamp_position
    if config.motion_detection is not None:
        camera_config.motion_detection = config.motion_detection
    if config.motion_threshold is not None:
        camera_config.motion_threshold = config.motion_threshold
    if config.motion_min_area is not None:
        camera_config.motion_min_area = config.motion_min_area
    if config.highlight_motion is not None:
        camera_config.highlight_motion = config.highlight_motion

    logger.info(f"Configuration updated: {camera_config}")
    return {"message": "Configuration updated", "restart_required": False}

@app.get("/patterns")
async def get_patterns(limit: int = 50, offset: int = 0):
    """Get list of motion patterns from database"""
    global motion_pattern_db

    if motion_pattern_db is None:
        return {"patterns": [], "total": 0}

    try:
        conn = sqlite3.connect(motion_pattern_db.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get total count
        cursor.execute('SELECT COUNT(*) as count FROM motion_patterns')
        total = cursor.fetchone()['count']

        # Get patterns with pagination
        cursor.execute('''
            SELECT id, pattern_id, classification, confidence, metadata, created_at
            FROM motion_patterns
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        ''', (limit, offset))

        patterns = []
        for row in cursor.fetchall():
            patterns.append({
                'id': row['id'],
                'pattern_id': row['pattern_id'],
                'classification': row['classification'],
                'confidence': row['confidence'],
                'metadata': json.loads(row['metadata']) if row['metadata'] else {},
                'created_at': row['created_at']
            })

        conn.close()

        return {
            'patterns': patterns,
            'total': total,
            'limit': limit,
            'offset': offset
        }
    except Exception as e:
        logger.error(f"Error fetching patterns: {e}")
        return {"patterns": [], "total": 0, "error": str(e)}

@app.get("/patterns/{pattern_id}")
async def get_pattern_detail(pattern_id: str):
    """Get detailed information about a specific pattern"""
    global motion_pattern_db

    if motion_pattern_db is None:
        raise HTTPException(status_code=503, detail="Pattern database not available")

    try:
        pattern = motion_pattern_db.get_pattern(pattern_id)
        if pattern is None:
            raise HTTPException(status_code=404, detail="Pattern not found")

        return pattern
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching pattern {pattern_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/patterns/{pattern_id}/label")
async def update_pattern_label(pattern_id: str, label: dict):
    """Update the label for a pattern (for training/correction)"""
    global motion_pattern_db

    if motion_pattern_db is None:
        raise HTTPException(status_code=503, detail="Pattern database not available")

    new_label = label.get('classification')
    if not new_label:
        raise HTTPException(status_code=400, detail="Missing 'classification' field")

    try:
        conn = sqlite3.connect(motion_pattern_db.db_path)
        cursor = conn.cursor()

        # Update classification
        cursor.execute('''
            UPDATE motion_patterns
            SET classification = ?, confidence = 1.0
            WHERE pattern_id = ?
        ''', (new_label, pattern_id))

        if cursor.rowcount == 0:
            conn.close()
            raise HTTPException(status_code=404, detail="Pattern not found")

        conn.commit()
        conn.close()

        logger.info(f"Updated pattern {pattern_id} label to {new_label}")
        return {"message": "Label updated", "pattern_id": pattern_id, "new_label": new_label}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating pattern label: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/patterns/{pattern_id}")
async def delete_pattern(pattern_id: str):
    """Delete a pattern from the database"""
    global motion_pattern_db

    if motion_pattern_db is None:
        raise HTTPException(status_code=503, detail="Pattern database not available")

    try:
        conn = sqlite3.connect(motion_pattern_db.db_path)
        cursor = conn.cursor()

        # Get signature path before deleting
        cursor.execute('SELECT signature_path FROM motion_patterns WHERE pattern_id = ?', (pattern_id,))
        row = cursor.fetchone()

        if row is None:
            conn.close()
            raise HTTPException(status_code=404, detail="Pattern not found")

        signature_path = row[0]

        # Delete from database
        cursor.execute('DELETE FROM motion_patterns WHERE pattern_id = ?', (pattern_id,))
        conn.commit()
        conn.close()

        # Delete signature file if it exists
        if signature_path and os.path.exists(signature_path):
            os.remove(signature_path)

        logger.info(f"Deleted pattern {pattern_id}")
        return {"message": "Pattern deleted", "pattern_id": pattern_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting pattern: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/patterns/similar/{pattern_id}")
async def find_similar(pattern_id: str, limit: int = 10, threshold: float = 0.7):
    """Find patterns similar to the given pattern"""
    global motion_pattern_db

    if motion_pattern_db is None:
        raise HTTPException(status_code=503, detail="Pattern database not available")

    try:
        # Get the pattern
        pattern = motion_pattern_db.get_pattern(pattern_id)
        if pattern is None:
            raise HTTPException(status_code=404, detail="Pattern not found")

        # Find similar patterns
        similar = motion_pattern_db.find_similar_patterns(
            pattern['signature'],
            limit=limit,
            similarity_threshold=threshold
        )

        return {
            'pattern_id': pattern_id,
            'similar_patterns': similar,
            'count': len(similar)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding similar patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions for frame processing
def add_timestamp(frame):
    """Add timestamp to frame"""
    if frame is None or not camera_config.show_timestamp:
        return frame
        
    try:
        # Create a copy to avoid modifying the original frame
        img = frame.copy()
        
        # Get current time
        now = datetime.datetime.now()
        timestamp_text = now.strftime("%Y-%m-%d %H:%M:%S")
        
        # Get frame dimensions
        height, width = img.shape[:2]
        
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = camera_config.timestamp_size
        font_thickness = 1 if height < 720 else 2
        color = camera_config.timestamp_color
        
        # Calculate text size
        text_size, _ = cv2.getTextSize(timestamp_text, font, font_scale, font_thickness)
        text_width, text_height = text_size
        
        # Position based on configuration
        if camera_config.timestamp_position == "top-left":
            position = (10, text_height + 10)
        elif camera_config.timestamp_position == "top-right":
            position = (width - text_width - 10, text_height + 10)
        elif camera_config.timestamp_position == "bottom-left":
            position = (10, height - 10)
        else:  # bottom-right is default
            position = (width - text_width - 10, height - 10)
        
        # Draw black background for better readability
        cv2.putText(img, timestamp_text, position, font, font_scale, (0, 0, 0), font_thickness + 1)
        # Draw text in specified color
        cv2.putText(img, timestamp_text, position, font, font_scale, color, font_thickness)
        
        return img
    except Exception as e:
        logger.error(f"Error adding timestamp: {e}")
        return frame

def detect_motion(frame, prev_color_frame=None, frame_index=0):
    """
    Detect motion in frame and return motion regions with optical flow features.

    Args:
        frame: Current frame (BGR)
        prev_color_frame: Previous frame for optical flow (BGR, optional)
        frame_index: Frame counter for frame skipping (optional)

    Returns:
        Tuple of (motion_detected, regions, flow_features)
    """
    global optical_flow_analyzer

    if frame is None or not camera_config.motion_detection:
        return False, [], None

    logger.debug("[MOTION_FLOW] Processing frame for motion detection")
    frame_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]

    try:
        # Convert to grayscale for motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply gaussian blur
        blur_kernel = camera_config.motion_blur if camera_config.motion_blur % 2 == 1 else camera_config.motion_blur + 1
        gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

        # Get previous frame from thread-safe state
        prev_frame = motion_state.get_prev_frame()

        # If first frame or prev_frame is None, initialize it
        if prev_frame is None:
            logger.info(f"[MOTION_FLOW] {frame_time} Initializing first frame for motion detection")
            motion_state.update_prev_frame(gray)
            return False, [], None

        # Calculate absolute difference between current and previous frame
        frame_delta = cv2.absdiff(prev_frame, gray)
        
        # Apply threshold to highlight differences
        thresh = cv2.threshold(frame_delta, camera_config.motion_threshold, 255, cv2.THRESH_BINARY)[1]
        
        # Dilate the thresholded image to fill in holes
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours on thresholded image
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Initialize motion regions list
        regions = []
        
        # Check if any contour is big enough to be considered motion
        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) < camera_config.motion_min_area:
                continue
                
            # Get bounding box coordinates
            (x, y, w, h) = cv2.boundingRect(contour)
            regions.append((x, y, w, h))
            motion_detected = True
        
        # Update previous frame in thread-safe state
        motion_state.update_prev_frame(gray)

        # Initialize flow features
        flow_features = None

        # If motion is detected and optical flow is enabled, calculate optical flow
        # Skip frames for performance (process every Nth frame)
        should_process_flow = (
            motion_detected and
            camera_config.optical_flow_enabled and
            optical_flow_analyzer is not None and
            prev_color_frame is not None and
            (frame_index % camera_config.optical_flow_frame_skip == 0)
        )

        if should_process_flow:
            try:
                # Optionally downscale for performance
                max_w, max_h = config.optical_flow.max_resolution
                h, w = frame.shape[:2]
                if w > max_w or h > max_h:
                    scale = min(max_w / w, max_h / h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    frame_scaled = cv2.resize(frame, (new_w, new_h))
                    prev_frame_scaled = cv2.resize(prev_color_frame, (new_w, new_h))

                    # Scale regions to match downscaled frame
                    regions_scaled = [(int(x*scale), int(y*scale), int(w*scale), int(h*scale))
                                     for x, y, w, h in regions]

                    flow_features = optical_flow_analyzer.extract_flow(
                        prev_frame_scaled, frame_scaled, regions_scaled)
                else:
                    flow_features = optical_flow_analyzer.extract_flow(prev_color_frame, frame, regions)

            except Exception as e:
                logger.error(f"Error extracting optical flow: {e}")
                flow_features = None

        # If motion is detected, update motion state
        if motion_detected:
            # Generate real-time classification (optional, only if visualization enabled)
            classification = None
            if (flow_features and optical_flow_analyzer and
                camera_config.optical_flow_visualization):
                try:
                    # Only generate signature if we have enough history
                    if len(optical_flow_analyzer.flow_history) >= 3:
                        signature = optical_flow_analyzer.generate_motion_signature()
                        if signature:
                            classification = optical_flow_analyzer.classify_motion(signature)
                except Exception as e:
                    logger.error(f"Error generating real-time classification: {e}")

            # Update thread-safe motion state (handles history trimming internally)
            motion_state.update_detection(motion_detected, regions, classification)

            # Log with high visibility to track motion detection precisely
            contour_areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) >= camera_config.motion_min_area]
            logger.critical(f"[MOTION_DETECTED] {frame_time} Motion detected! Regions: {len(regions)}")
            logger.critical(f"[MOTION_DETECTED] Contour areas: {contour_areas}")
            logger.critical(f"[MOTION_DETECTED] Motion threshold: {camera_config.motion_threshold}, Min area: {camera_config.motion_min_area}")

            if classification:
                logger.info(f"[MOTION_FLOW] {frame_time} Motion detected! Regions: {len(regions)}, "
                           f"Classification: {classification['label']} ({classification['confidence']:.2f})")
            else:
                logger.info(f"[MOTION_FLOW] {frame_time} Motion detected! Regions: {len(regions)}, Contour areas: {contour_areas}")

        if not motion_detected and random.random() < 0.01:  # Log about 1% of non-motion frames to avoid excessive logging
            logger.debug(f"[MOTION_FLOW] {frame_time} No motion detected")

        return motion_detected, regions, flow_features
    except Exception as e:
        logger.error(f"Error detecting motion: {e}")
        return False, [], None

# Parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="FastAPI MJPEG Streaming Server with Motion Storage")
    
    # Camera settings
    parser.add_argument("--width", type=int, default=640, help="Camera width in pixels (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Camera height in pixels (default: 480)")
    parser.add_argument("--fps", type=int, default=30, help="Camera frame rate (default: 30)")
    parser.add_argument("--rotation", type=int, default=0, choices=[0, 90, 180, 270], 
                        help="Camera rotation in degrees (default: 0)")
    
    # Server settings
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--max-clients", type=int, default=10, help="Maximum number of clients (default: 10)")
    parser.add_argument("--client-timeout", type=int, default=30, 
                        help="Client timeout in seconds (default: 30)")
    
    # Timestamp settings
    parser.add_argument("--no-timestamp", action="store_true", help="Disable timestamp display")
    parser.add_argument("--timestamp-position", type=str, default="bottom-right", 
                        choices=["top-left", "top-right", "bottom-left", "bottom-right"], 
                        help="Timestamp position (default: bottom-right)")
    
    # Motion detection settings
    parser.add_argument("--no-motion", action="store_true", help="Disable motion detection")
    parser.add_argument("--motion-threshold", type=int, default=25, 
                        help="Motion detection threshold (5-100, lower is more sensitive, default: 25)")
    parser.add_argument("--motion-min-area", type=int, default=500, 
                        help="Minimum pixel area to consider as motion (default: 500)")
    parser.add_argument("--no-highlight", action="store_true", 
                        help="Disable motion highlighting in video")
    
    # Storage settings
    parser.add_argument("--storage-path", type=str, default="/tmp/motion_events",
                       help="Path to store motion events (default: /tmp/motion_events)")
    parser.add_argument("--max-storage", type=int, default=1000,
                       help="Maximum storage usage in MB (default: 1000)")
    parser.add_argument("--remote-url", type=str, default="http://192.168.1.100:8080/storage",
                       help="URL of remote storage server")
    parser.add_argument("--api-key", type=str, default="your_api_key_here",
                       help="API key for remote storage server")
    parser.add_argument("--no-upload", action="store_true",
                       help="Disable uploading to remote server")
    parser.add_argument("--upload-throttle", type=int, default=500,
                       help="Upload throttle in KB/s (default: 500)")
    parser.add_argument("--no-wifi-monitoring", action="store_true",
                       help="Disable WiFi signal monitoring")
    parser.add_argument("--no-thumbnails", action="store_true",
                       help="Disable thumbnail generation")
    
    return parser.parse_args()

# Main entry point
if __name__ == "__main__":
    try:
        # Load configuration using centralized system (handles CLI args, env vars, config file)
        # Parse with old argument parser for backward compatibility
        # But we'll use centralized config as primary source
        args = parse_arguments()

        # Load centralized configuration
        config_file = None  # Could add --config arg to parse_arguments()
        config = get_config(config_file=config_file)

        # Apply command-line argument overrides to centralized config
        # Camera settings
        if args.width != 640:  # Not default
            config.camera.width = args.width
        if args.height != 480:
            config.camera.height = args.height
        if args.fps != 30:
            config.camera.frame_rate = args.fps
        if args.rotation != 0:
            config.camera.rotation = args.rotation
        if args.max_clients != 10:
            config.camera.max_clients = args.max_clients
        if args.client_timeout != 30:
            config.camera.client_timeout = args.client_timeout
        config.camera.show_timestamp = not args.no_timestamp
        if args.timestamp_position != "bottom-right":
            config.camera.timestamp_position = args.timestamp_position

        # Motion detection settings
        config.motion_detection.enabled = not args.no_motion
        if args.motion_threshold != 25:
            config.motion_detection.threshold = args.motion_threshold
        if args.motion_min_area != 500:
            config.motion_detection.min_area = args.motion_min_area
        config.motion_detection.highlight_motion = not args.no_highlight

        # Storage settings
        if args.storage_path != "/tmp/motion_events":
            config.storage.local_storage_path = args.storage_path
        if args.max_storage != 1000:
            config.storage.max_disk_usage_mb = args.max_storage
        if args.remote_url != "http://192.168.1.100:8080/storage":
            config.storage.remote_storage_url = args.remote_url
        if args.api_key != "your_api_key_here":
            config.storage.remote_api_key = args.api_key
        if args.no_upload:
            config.storage.upload_throttle_kbps = 0
        elif args.upload_throttle != 500:
            config.storage.upload_throttle_kbps = args.upload_throttle
        config.storage.wifi_monitoring = not args.no_wifi_monitoring
        config.storage.generate_thumbnails = not args.no_thumbnails

        # Create a flat camera_config that merges camera + motion_detection for backward compat
        from dataclasses import dataclass
        @dataclass
        class FlatCameraConfig:
            pass

        camera_config = FlatCameraConfig()
        # Copy camera settings
        for attr in ['width', 'height', 'frame_rate', 'rotation', 'max_clients', 'client_timeout',
                     'show_timestamp', 'timestamp_position', 'timestamp_color', 'timestamp_size']:
            setattr(camera_config, attr, getattr(config.camera, attr))

        # Copy motion detection settings with motion_ prefix
        camera_config.motion_detection = config.motion_detection.enabled
        camera_config.motion_threshold = config.motion_detection.threshold
        camera_config.motion_min_area = config.motion_detection.min_area
        camera_config.motion_blur = config.motion_detection.blur_kernel_size
        camera_config.highlight_motion = config.motion_detection.highlight_motion
        camera_config.motion_history_size = config.motion_detection.history_size

        # Copy optical flow settings
        camera_config.optical_flow_enabled = config.optical_flow.enabled
        camera_config.optical_flow_feature_max = config.optical_flow.feature_max
        camera_config.optical_flow_min_distance = config.optical_flow.min_distance
        camera_config.optical_flow_quality_level = config.optical_flow.quality_level
        camera_config.optical_flow_grid_size = config.optical_flow.grid_size
        camera_config.optical_flow_direction_bins = config.optical_flow.direction_bins
        camera_config.optical_flow_visualization = config.optical_flow.visualization
        camera_config.optical_flow_frame_skip = config.optical_flow.frame_skip

        # Set storage_config to point to centralized config
        storage_config = config.storage

        # Log configuration
        logger.info(f"Starting server with configuration:")
        logger.info(f"  Camera: {camera_config.width}x{camera_config.height} @ {camera_config.frame_rate}fps")
        logger.info(f"  Server: {args.host}:{args.port}")
        logger.info(f"  Timestamp: {'Enabled' if camera_config.show_timestamp else 'Disabled'}")
        logger.info(f"  Motion detection: {'Enabled' if camera_config.motion_detection else 'Disabled'}")
        logger.info(f"  Storage: {storage_config.local_storage_path} (max: {storage_config.max_disk_usage_mb} MB)")
        logger.info(f"  Remote storage: {storage_config.remote_storage_url}")
        logger.info(f"  Upload throttle: {storage_config.upload_throttle_kbps} KB/s")
        logger.info(f"  WiFi monitoring: {'Enabled' if storage_config.wifi_monitoring else 'Disabled'}")
        logger.info(f"  Thumbnail generation: {'Enabled' if storage_config.generate_thumbnails else 'Disabled'}")

        # Run the server with a timeout for graceful shutdown
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info",
            reload=False,
            timeout_graceful_shutdown=5  # 5 seconds timeout for graceful shutdown
        )
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        sys.exit(1)

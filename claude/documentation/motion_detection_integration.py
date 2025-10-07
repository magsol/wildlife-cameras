"""
Integration code for optical flow analysis with the existing motion detection system.
This file demonstrates how to integrate the optical flow analyzer with the existing
fastapi_mjpeg_server_with_storage.py and motion_storage.py files.

This is not a standalone file but rather shows the key sections that would need
to be modified in the existing code.

IMPORTANT NOTES:
1. The actual FrameBuffer class inherits from io.BufferedIOBase for PiCamera2 compatibility
2. The actual MotionEventRecorder uses threading and a queue-based event processing system
3. Flow features need to be passed from FrameBuffer.write() to MotionEventRecorder
   - Option A: Use a module-level variable (simple but not ideal for threading)
   - Option B: Add flow features to the event queue (better threading model)
   - Option C: Pass via circular buffer alongside frames (cleanest design)
4. Performance optimizations are critical for Raspberry Pi:
   - Frame skipping (process every Nth frame)
   - Resolution downscaling for flow computation
   - Disable real-time classification during recording (do it at end instead)
5. The metadata.json structure needs extension to include motion_analysis section
"""

# Import section additions
import cv2
import numpy as np
from datetime import datetime
import logging
import json
import os
from pathlib import Path

# Add this import
from optical_flow_analyzer import OpticalFlowAnalyzer, MotionPatternDatabase

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Initialize globals section addition
# Add these global variables
optical_flow_analyzer = None
motion_pattern_db = None

# Configuration class updates
# NOTE: These parameters need to be added to the existing CameraConfig and StorageConfig classes

# Add to CameraConfig in fastapi_mjpeg_server_with_storage.py:
"""
class CameraConfig:
    # ... existing parameters ...

    # New parameters for optical flow analysis
    optical_flow_enabled: bool = True
    optical_flow_feature_max: int = 100
    optical_flow_min_distance: int = 7
    optical_flow_quality_level: float = 0.3
    optical_flow_grid_size: Tuple[int, int] = (8, 8)
    optical_flow_direction_bins: int = 8
    optical_flow_visualization: bool = False  # Expensive, disable by default
    optical_flow_frame_skip: int = 2  # Process every Nth frame for performance
"""

# Add to StorageConfig in motion_storage.py:
"""
class StorageConfig:
    # ... existing parameters ...

    # New parameters for optical flow storage
    store_optical_flow_data: bool = True
    optical_flow_signature_dir: str = "flow_signatures"
    optical_flow_database_path: str = "motion_patterns.db"

    # Motion classification parameters
    motion_classification_enabled: bool = True
    min_classification_confidence: float = 0.5
    save_flow_visualizations: bool = True

    # Performance optimization
    optical_flow_max_resolution: Tuple[int, int] = (320, 240)  # Downscale for flow computation
"""

# Lifespan function update

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources when the application starts and stops."""
    global optical_flow_analyzer, motion_pattern_db

    try:
        # Existing initialization code
        # ...

        # Initialize optical flow analyzer and database
        # NOTE: Get storage_config from motion_storage module after initialization
        if hasattr(storage_config, 'motion_classification_enabled') and storage_config.motion_classification_enabled:
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
                                        storage_config.optical_flow_signature_dir)
            os.makedirs(signature_dir, exist_ok=True)

            # Initialize motion pattern database
            db_path = os.path.join(storage_config.local_storage_path,
                                  storage_config.optical_flow_database_path)

            motion_pattern_db = MotionPatternDatabase(db_path=db_path,
                                                     signature_dir=signature_dir)

            logger.info(f"Optical flow analyzer initialized with config: {flow_config}")
            logger.info(f"Motion pattern database at: {db_path}")

        yield

        # Cleanup optical flow resources
        if optical_flow_analyzer is not None:
            logger.info("Cleaning up optical flow analyzer")
            optical_flow_analyzer.reset()

        # Existing cleanup code
        # ...

    except Exception as e:
        logger.error(f"Error in lifespan: {e}")
        raise
    finally:
        # Cleanup resources
        pass

# Modified detect_motion function

def detect_motion(frame, prev_frame=None, frame_index=0):
    """
    Detect motion in a frame using background subtraction and optical flow.

    Args:
        frame: Current frame (BGR or grayscale)
        prev_frame: Previous frame for optical flow (BGR, optional)
        frame_index: Frame counter for frame skipping (optional)

    Returns:
        Tuple of (motion_detected, regions, flow_features)
    """
    global prev_motion_frame, optical_flow_analyzer

    # Convert frame to grayscale for motion detection
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    # Apply gaussian blur as per existing implementation
    blur_kernel = camera_config.motion_blur if camera_config.motion_blur % 2 == 1 else camera_config.motion_blur + 1
    gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    # Initialize previous frame if needed
    if prev_motion_frame is None:
        prev_motion_frame = gray.copy()
        return False, [], None

    # Compute absolute difference between current and previous frame
    frame_delta = cv2.absdiff(prev_motion_frame, gray)

    # Apply threshold to highlight differences
    thresh = cv2.threshold(frame_delta, camera_config.motion_threshold, 255, cv2.THRESH_BINARY)[1]

    # Dilate thresholded image to fill in holes
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours on thresholded image
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize motion regions
    regions = []
    motion_detected = False

    # Process contours
    for c in contours:
        # Filter out small contours
        if cv2.contourArea(c) < camera_config.motion_min_area:
            continue

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(c)
        regions.append((x, y, w, h))
        motion_detected = True

    # Update the previous frame
    prev_motion_frame = gray.copy()

    # Initialize flow features
    flow_features = None

    # If motion is detected and optical flow is enabled, calculate optical flow
    # Skip frames for performance (process every Nth frame)
    should_process_flow = (
        motion_detected and
        camera_config.optical_flow_enabled and
        optical_flow_analyzer is not None and
        prev_frame is not None and
        (frame_index % camera_config.optical_flow_frame_skip == 0)
    )

    if should_process_flow:
        try:
            # Optionally downscale for performance
            if hasattr(storage_config, 'optical_flow_max_resolution'):
                max_w, max_h = storage_config.optical_flow_max_resolution
                h, w = frame.shape[:2]
                if w > max_w or h > max_h:
                    scale = min(max_w / w, max_h / h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    frame_scaled = cv2.resize(frame, (new_w, new_h))
                    prev_frame_scaled = cv2.resize(prev_frame, (new_w, new_h))

                    # Scale regions to match downscaled frame
                    regions_scaled = [(int(x*scale), int(y*scale), int(w*scale), int(h*scale))
                                     for x, y, w, h in regions]

                    flow_features = optical_flow_analyzer.extract_flow(
                        prev_frame_scaled, frame_scaled, regions_scaled)
                else:
                    flow_features = optical_flow_analyzer.extract_flow(prev_frame, frame, regions)
            else:
                flow_features = optical_flow_analyzer.extract_flow(prev_frame, frame, regions)

        except Exception as e:
            logger.error(f"Error extracting optical flow: {e}")
            flow_features = None

    return motion_detected, regions, flow_features

# Add optical flow support to the FrameBuffer class
# NOTE: The actual FrameBuffer class inherits from io.BufferedIOBase and has a different structure
# This is a simplified illustration showing the key changes needed

"""
In fastapi_mjpeg_server_with_storage.py, modify FrameBuffer.write() method:

class FrameBuffer(io.BufferedIOBase):
    def __init__(self, max_size: int = 5):
        super().__init__()
        # ... existing initialization ...
        self.last_frame = None  # Store last frame for optical flow analysis
        self.frame_index = 0  # Track frame number for skip logic

    def write(self, buf, *args, **kwargs):
        global prev_frame, motion_detected, motion_regions, optical_flow_analyzer

        # ... existing JPEG encoding code ...

        # Decode JPEG to numpy array for processing
        if camera_config.motion_detection and raw_img is not None:
            # Store previous frame for optical flow
            prev_frame_for_flow = self.last_frame
            self.last_frame = raw_img.copy()
            self.frame_index += 1

            # Detect motion with optical flow
            motion_detected, motion_regions, flow_features = detect_motion(
                raw_img, prev_frame_for_flow, self.frame_index)

            # If motion detected, add to history
            if motion_detected:
                motion_time = datetime.datetime.now()

                # Generate real-time classification (optional, expensive)
                # Only do this if we have enough flow history
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

                # Add motion event with optional classification
                motion_history.append((motion_time, motion_regions, classification))

                # Trim history if needed
                if len(motion_history) > camera_config.motion_history_size * 2:
                    motion_history = motion_history[-camera_config.motion_history_size * 2:]

                # Log motion detection
                if classification:
                    logger.info(f"Motion detected! Regions: {len(motion_regions)}, "
                               f"Classification: {classification['label']} "
                               f"({classification['confidence']:.2f})")
                else:
                    logger.info(f"Motion detected! Regions: {len(motion_regions)}")

                # Store flow features in motion_storage module for recorder
                # This will be picked up by MotionEventRecorder
                if flow_features:
                    # Store in a global or pass to recorder
                    # Details depend on the threading model

        # ... rest of existing write() method ...
        return len(buf)
"""

# Update MotionEventRecorder in motion_storage.py
# NOTE: The actual MotionEventRecorder has a different signature and uses threading
# Here's what needs to be added:

"""
In motion_storage.py, modify MotionEventRecorder class:

class MotionEventRecorder:
    def __init__(self, frame_buffer, config):
        # ... existing initialization ...
        self.flow_features_list = []  # NEW: Store flow features during recording

    def start_recording(self, motion_regions):
        # Note: Current signature doesn't take flow_features parameter
        # Flow features will be collected during add_frame() calls

        if self.recording:
            logger.debug(f"Already recording motion - ignoring start_recording call")
            return

        # ... existing event initialization ...

        # NEW: Reset flow features for this event
        self.flow_features_list = []

        # Reset optical flow analyzer history for new event
        if optical_flow_analyzer is not None:
            optical_flow_analyzer.reset()

        logger.info(f"Started recording motion event {event_id}")
"""
    
"""
    def add_frame(self, frame, timestamp):
        # Note: Current signature only takes frame and timestamp
        # Flow features need to be collected differently

        # ... existing frame recording logic ...

        # NEW: Collect flow features if available
        # Option: Store flow features alongside frames in a parallel structure
        # Or: Access global flow features that were computed during motion detection
"""
    
"""
    def stop_recording(self):
        # ... existing stop recording logic ...

        # NEW: Process optical flow data if available
        motion_classification = None
        motion_signature = None

        if optical_flow_analyzer is not None:
            try:
                # Generate motion signature from accumulated flow history
                # The analyzer has been collecting flow features during the event
                motion_signature = optical_flow_analyzer.generate_motion_signature()

                if motion_signature:
                    # Classify the motion pattern
                    motion_classification = optical_flow_analyzer.classify_motion(motion_signature)

                    # Save to pattern database if enabled
                    if (storage_config.store_optical_flow_data and
                        motion_pattern_db is not None and
                        motion_classification['confidence'] >= storage_config.min_classification_confidence):

                        motion_pattern_db.add_pattern(
                            self.current_event["id"],
                            motion_signature,
                            motion_classification,
                            {
                                "event_id": self.current_event["id"],
                                "duration": duration,
                                "time_of_day": start_time.hour,
                                "frame_count": len(self.current_event["frames"])
                            }
                        )

                    # Save flow visualization if enabled
                    if storage_config.save_flow_visualizations:
                        # Get a representative frame from the buffer
                        recent_frames = self.frame_buffer.get_recent_frames(seconds=1)
                        if recent_frames and len(optical_flow_analyzer.flow_history) > 0:
                            frame, _ = recent_frames[-1]
                            last_flow = optical_flow_analyzer.flow_history[-1]

                            # Generate visualization
                            flow_vis = optical_flow_analyzer.visualize_flow(frame, last_flow)

                            # Save to event directory
                            flow_vis_path = os.path.join(event_dir, "flow_visualization.jpg")
                            cv2.imwrite(flow_vis_path, flow_vis)

                    # Log classification
                    logger.info(f"Motion event {event_id} classified as: "
                              f"{motion_classification['label']} "
                              f"(confidence: {motion_classification['confidence']:.2f})")

            except Exception as e:
                logger.error(f"Error processing optical flow data: {e}")
                import traceback
                logger.error(traceback.format_exc())

        # NEW: Add motion_analysis section to metadata
        if motion_classification:
            self.current_event["motion_analysis"] = {
                "classification": motion_classification,
                "motion_characteristics": motion_signature.get('statistical_features', {}),
                "temporal_features": motion_signature.get('temporal_features', {}),
                "signature_hash": hashlib.md5(
                    motion_signature['histogram_features'].tobytes()
                ).hexdigest() if motion_signature else None
            }

        # ... existing metadata saving and cleanup ...
"""

# Update motion status API endpoint

@app.get("/motion_status")
async def get_motion_status():
    """Get current motion status."""
    global motion_history
    
    # Clean up old motion entries
    current_time = datetime.datetime.now()
    cutoff_time = current_time - datetime.timedelta(seconds=60)
    recent_motion = [(t, r, c) for t, r, c in motion_history if t >= cutoff_time]
    
    # Get latest motion detection
    latest_motion = None
    latest_classification = None
    
    if recent_motion:
        latest_time, latest_regions, latest_classification = recent_motion[-1]
        latest_motion = {
            "time": latest_time.isoformat(),
            "regions": latest_regions,
        }
        
        if latest_classification:
            latest_motion["classification"] = latest_classification
    
    # Return motion status
    return {
        "motion_detected": len(recent_motion) > 0,
        "latest_motion": latest_motion,
        "motion_history": [
            {
                "time": t.isoformat(),
                "regions": r,
                "classification": c
            } for t, r, c in recent_motion
        ]
    }

# Update HTML template to include classification display

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Wildlife Camera</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f0f0f0;
        }}
        h1 {{
            color: #333;
        }}
        .container {{
            display: flex;
            flex-direction: column;
            max-width: 1200px;
            margin: 0 auto;
        }}
        .video-container {{
            position: relative;
        }}
        .stream {{
            max-width: 100%;
            border: 1px solid #ccc;
        }}
        .motion-alert {{
            position: absolute;
            top: 10px;
            left: 10px;
            background-color: rgba(255, 0, 0, 0.7);
            color: white;
            padding: 5px 10px;
            border-radius: 5px;
            display: none;
        }}
        .controls {{
            margin-top: 20px;
            padding: 10px;
            background-color: #fff;
            border: 1px solid #ccc;
            border-radius: 5px;
        }}
        .history {{
            margin-top: 20px;
            padding: 10px;
            background-color: #fff;
            border: 1px solid #ccc;
            border-radius: 5px;
        }}
        .history h2 {{
            margin-top: 0;
        }}
        .motion-event {{
            padding: 10px;
            margin-bottom: 10px;
            background-color: #f9f9f9;
            border-left: 4px solid #ff6b6b;
            border-radius: 3px;
        }}
        .motion-timestamp {{
            font-weight: bold;
            color: #333;
        }}
        .motion-classification {{
            display: inline-block;
            margin-left: 10px;
            padding: 3px 8px;
            border-radius: 10px;
            font-size: 0.8em;
        }}
        .high-confidence {{
            background-color: #4CAF50;
            color: white;
        }}
        .low-confidence {{
            background-color: #ff9800;
            color: white;
        }}
        .unknown {{
            background-color: #9e9e9e;
            color: white;
        }}
        .confidence-bar {{
            margin-top: 5px;
            height: 10px;
            background-color: #e0e0e0;
            border-radius: 5px;
            overflow: hidden;
        }}
        .confidence-bar-fill {{
            height: 100%;
            background-color: #4CAF50;
            border-radius: 5px;
            transition: width 0.3s ease;
        }}
        .thumbnail-container {{
            display: flex;
            margin-top: 5px;
        }}
        .motion-thumbnail {{
            width: 120px;
            height: 90px;
            object-fit: cover;
            margin-right: 10px;
            border: 1px solid #ddd;
        }}
        .motion-flow {{
            width: 120px;
            height: 90px;
            object-fit: cover;
            border: 1px solid #ddd;
        }}
        .tabs {{
            margin-top: 20px;
        }}
        .tab {{
            display: inline-block;
            padding: 10px 15px;
            background-color: #ddd;
            cursor: pointer;
            border-radius: 5px 5px 0 0;
        }}
        .tab.active {{
            background-color: #fff;
            border: 1px solid #ccc;
            border-bottom: none;
        }}
        .tab-content {{
            display: none;
            padding: 20px;
            background-color: #fff;
            border: 1px solid #ccc;
            border-radius: 0 0 5px 5px;
        }}
        .tab-content.active {{
            display: block;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Wildlife Camera</h1>
        
        <div class="video-container">
            <img class="stream" src="/video_feed" alt="Video Stream" />
            <div class="motion-alert" id="motionAlert">
                Motion Detected: <span id="motionClassification">Unknown</span>
                <div class="confidence-bar">
                    <div id="confidenceLevel" class="confidence-bar-fill" style="width: 0%"></div>
                </div>
            </div>
        </div>
        
        <div class="tabs">
            <div class="tab active" data-tab="history">Motion History</div>
            <div class="tab" data-tab="patterns">Motion Patterns</div>
            <div class="tab" data-tab="settings">Settings</div>
        </div>
        
        <div id="historyTab" class="tab-content active">
            <div class="history">
                <h2>Recent Motion Events</h2>
                <div id="motionHistory"></div>
            </div>
        </div>
        
        <div id="patternsTab" class="tab-content">
            <h2>Motion Patterns</h2>
            <div id="patternCategories">
                <div class="pattern-category">
                    <h3>Vehicles</h3>
                    <div class="pattern-samples"></div>
                </div>
                <div class="pattern-category">
                    <h3>People</h3>
                    <div class="pattern-samples"></div>
                </div>
                <div class="pattern-category">
                    <h3>Animals</h3>
                    <div class="pattern-samples"></div>
                </div>
                <div class="pattern-category">
                    <h3>Environment (Wind/Rain)</h3>
                    <div class="pattern-samples"></div>
                </div>
                <div class="pattern-category">
                    <h3>Unknown</h3>
                    <div class="pattern-samples"></div>
                </div>
            </div>
        </div>
        
        <div id="settingsTab" class="tab-content">
            <h2>Settings</h2>
            <form id="settingsForm">
                <fieldset>
                    <legend>Motion Detection</legend>
                    <div>
                        <label>
                            <input type="checkbox" id="motionDetection" checked />
                            Enable motion detection
                        </label>
                    </div>
                    <div>
                        <label>
                            <input type="checkbox" id="opticalFlowEnabled" checked />
                            Enable optical flow analysis
                        </label>
                    </div>
                    <div>
                        <label for="motionThreshold">Motion threshold:</label>
                        <input type="range" id="motionThreshold" min="10" max="50" value="25" />
                        <span id="motionThresholdValue">25</span>
                    </div>
                    <div>
                        <label for="motionMinArea">Minimum motion area:</label>
                        <input type="range" id="motionMinArea" min="100" max="2000" value="500" />
                        <span id="motionMinAreaValue">500</span>
                    </div>
                    <div>
                        <label for="minMotionDuration">Minimum motion duration (seconds):</label>
                        <input type="range" id="minMotionDuration" min="1" max="10" value="3" />
                        <span id="minMotionDurationValue">3</span>
                    </div>
                </fieldset>
                <button type="submit">Save Settings</button>
            </form>
        </div>
    </div>
    
    <script>
        // Tab functionality
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                
                tab.classList.add('active');
                document.getElementById(tab.dataset.tab + 'Tab').classList.add('active');
            });
        });
        
        // Motion detection status
        const motionAlert = document.getElementById('motionAlert');
        const motionClassification = document.getElementById('motionClassification');
        const confidenceLevel = document.getElementById('confidenceLevel');
        const motionHistory = document.getElementById('motionHistory');
        
        // Update motion detection status
        function updateMotionStatus() {
            fetch('/motion_status')
                .then(response => response.json())
                .then(data => {
                    // Update motion alert
                    motionAlert.style.display = data.motion_detected ? 'block' : 'none';
                    
                    if (data.motion_detected && data.latest_motion && data.latest_motion.classification) {
                        // Show classification
                        const classification = data.latest_motion.classification;
                        motionClassification.textContent = classification.label;
                        
                        // Update confidence bar
                        const confidence = classification.confidence * 100;
                        confidenceLevel.style.width = `${confidence}%`;
                        
                        // Set color based on confidence
                        if (confidence > 80) {
                            confidenceLevel.style.backgroundColor = '#4CAF50';  // Green
                        } else if (confidence > 50) {
                            confidenceLevel.style.backgroundColor = '#ff9800';  // Orange
                        } else {
                            confidenceLevel.style.backgroundColor = '#f44336';  // Red
                        }
                    } else {
                        motionClassification.textContent = 'Unknown';
                        confidenceLevel.style.width = '0%';
                    }
                    
                    // Update motion history
                    updateMotionHistory(data.motion_history);
                });
        }
        
        // Update motion history
        function updateMotionHistory(history) {
            motionHistory.innerHTML = '';
            
            if (history.length === 0) {
                motionHistory.innerHTML = '<p>No recent motion events</p>';
                return;
            }
            
            // Group events by minute to avoid too many entries
            const groupedEvents = {};
            history.forEach(event => {
                const time = new Date(event.time);
                const key = `${time.getHours()}:${time.getMinutes()}`;
                
                if (!groupedEvents[key]) {
                    groupedEvents[key] = [];
                }
                
                groupedEvents[key].push(event);
            });
            
            // Create history entries
            Object.keys(groupedEvents).forEach(key => {
                const events = groupedEvents[key];
                const latestEvent = events[events.length - 1];
                const time = new Date(latestEvent.time);
                
                const eventElement = document.createElement('div');
                eventElement.className = 'motion-event';
                
                // Format time
                const timeString = time.toLocaleTimeString();
                
                // Create header with classification badge
                let headerHTML = `<div class="motion-event-header">
                    <span class="motion-timestamp">${timeString}</span>`;
                
                // Add classification badge if available
                if (latestEvent.classification) {
                    const classification = latestEvent.classification;
                    const confidence = classification.confidence;
                    const confidenceClass = confidence > 0.7 ? 'high-confidence' : 'low-confidence';
                    
                    headerHTML += `<span class="motion-classification ${confidenceClass}">
                        ${classification.label}
                    </span>`;
                } else {
                    headerHTML += `<span class="motion-classification unknown">
                        unknown
                    </span>`;
                }
                
                headerHTML += '</div>';
                
                // Add details
                let detailsHTML = `<div class="motion-event-details">
                    <span>${events.length} detection${events.length > 1 ? 's' : ''}</span>
                    <span>${latestEvent.regions ? latestEvent.regions.length : 0} region${latestEvent.regions && latestEvent.regions.length !== 1 ? 's' : ''}</span>
                </div>`;
                
                eventElement.innerHTML = headerHTML + detailsHTML;
                motionHistory.appendChild(eventElement);
            });
        }
        
        // Update every second
        setInterval(updateMotionStatus, 1000);
        
        // Initialize
        updateMotionStatus();
        
        // Settings form
        const settingsForm = document.getElementById('settingsForm');
        settingsForm.addEventListener('submit', (e) => {
            e.preventDefault();
            
            // Get form values
            const settings = {
                motion_detection: document.getElementById('motionDetection').checked,
                optical_flow_enabled: document.getElementById('opticalFlowEnabled').checked,
                motion_threshold: parseInt(document.getElementById('motionThreshold').value),
                motion_min_area: parseInt(document.getElementById('motionMinArea').value),
                min_motion_duration_sec: parseInt(document.getElementById('minMotionDuration').value)
            };
            
            // Send settings to server
            fetch('/update_settings', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(settings)
            })
            .then(response => response.json())
            .then(data => {
                alert('Settings updated successfully');
            })
            .catch(error => {
                alert('Error updating settings: ' + error);
            });
        });
        
        // Update range input values
        document.getElementById('motionThreshold').addEventListener('input', (e) => {
            document.getElementById('motionThresholdValue').textContent = e.target.value;
        });
        
        document.getElementById('motionMinArea').addEventListener('input', (e) => {
            document.getElementById('motionMinAreaValue').textContent = e.target.value;
        });
        
        document.getElementById('minMotionDuration').addEventListener('input', (e) => {
            document.getElementById('minMotionDurationValue').textContent = e.target.value;
        });
    </script>
</body>
</html>
"""

# ============================================================================
# INTEGRATION SUMMARY
# ============================================================================
"""
To integrate optical flow motion classification, the following changes are needed:

1. CONFIGURATION CHANGES:
   - Add optical flow parameters to CameraConfig dataclass
   - Add optical flow storage parameters to StorageConfig dataclass

2. INITIALIZATION CHANGES (in lifespan function):
   - Create OpticalFlowAnalyzer instance with configuration
   - Create MotionPatternDatabase instance
   - Initialize storage directories

3. MOTION DETECTION CHANGES:
   - Update detect_motion() to accept prev_frame and frame_index
   - Add optical flow extraction when motion is detected
   - Implement frame skipping for performance
   - Implement resolution downscaling for flow computation

4. FRAME BUFFER CHANGES:
   - Add last_frame and frame_index tracking
   - Pass previous frame to detect_motion()
   - Optionally generate real-time classification for display

5. MOTION EVENT RECORDER CHANGES:
   - Add flow_features_list to track flow during recording
   - Reset analyzer when starting new recording
   - Generate signature and classify at end of recording
   - Save signature to database
   - Save flow visualization image
   - Add motion_analysis to metadata.json

6. API ENDPOINT CHANGES:
   - Update /motion_status to include classification
   - Add endpoints for pattern browsing (Phase 3)
   - Add endpoints for user feedback (Phase 3)

7. UI CHANGES:
   - Display classification labels in real-time
   - Show confidence bars
   - Display flow visualizations in event history
   - Add pattern management tab (Phase 3)

PERFORMANCE CONSIDERATIONS:
- Process optical flow every 2-3 frames, not every frame
- Downscale to 320x240 for flow computation
- Disable real-time classification unless needed for display
- Do full signature generation and classification only at event end
- Use threading to avoid blocking camera pipeline

THREADING MODEL:
- FrameBuffer.write() runs in camera thread
- Optical flow extraction should be fast (< 10ms target)
- Heavy processing (signature, classification) should be in recorder thread
- Pattern database has its own locking for thread safety
"""
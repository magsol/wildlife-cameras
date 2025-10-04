"""
Integration code for optical flow analysis with the existing motion detection system.
This file demonstrates how to integrate the optical flow analyzer with the existing
fastapi_mjpeg_server_with_storage.py file.

This is not a standalone file but rather shows the key sections that would need
to be modified in the existing code.
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

class MotionConfig(BaseModel):
    # Existing parameters
    motion_detection: bool = True
    motion_threshold: int = 25
    motion_min_area: int = 500
    motion_mask_enabled: bool = False
    motion_mask_points: List[List[int]] = []
    motion_history_size: int = 10
    motion_cooldown_seconds: int = 2
    
    # New parameters for optical flow analysis
    optical_flow_enabled: bool = True
    optical_flow_feature_max: int = 100
    optical_flow_min_distance: int = 7
    optical_flow_grid_size: Tuple[int, int] = (8, 8)
    optical_flow_direction_bins: int = 8
    optical_flow_visualization: bool = True

class StorageConfig(BaseModel):
    # Existing parameters
    local_storage_enabled: bool = True
    local_storage_path: str = "storage"
    max_event_duration_sec: int = 30
    min_motion_duration_sec: int = 3
    pre_motion_sec: int = 2
    post_motion_sec: int = 2
    
    # New parameters for optical flow storage
    store_optical_flow_data: bool = True
    optical_flow_signature_dir: str = "flow_signatures"
    optical_flow_database_path: str = "motion_patterns.db"
    
    # Motion classification parameters
    motion_classification_enabled: bool = True
    min_classification_confidence: float = 0.5
    save_flow_visualizations: bool = True

# Lifespan function update

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources when the application starts and stops."""
    global optical_flow_analyzer, motion_pattern_db
    
    try:
        # Existing initialization code
        # ...
        
        # Initialize optical flow analyzer and database
        if storage_config.motion_classification_enabled:
            logger.info("Initializing optical flow analyzer")
            
            # Create optical flow configuration
            flow_config = {
                'feature_params': {
                    'maxCorners': camera_config.optical_flow_feature_max,
                    'qualityLevel': 0.3,
                    'minDistance': camera_config.optical_flow_min_distance,
                    'blockSize': 7
                },
                'grid_size': camera_config.optical_flow_grid_size,
                'direction_bins': camera_config.optical_flow_direction_bins
            }
            
            optical_flow_analyzer = OpticalFlowAnalyzer(config=flow_config)
            
            # Create signature directory if it doesn't exist
            os.makedirs(os.path.join(storage_config.local_storage_path, 
                                     storage_config.optical_flow_signature_dir), 
                        exist_ok=True)
            
            # Initialize motion pattern database
            db_path = os.path.join(storage_config.local_storage_path, 
                                  storage_config.optical_flow_database_path)
            signature_dir = os.path.join(storage_config.local_storage_path, 
                                        storage_config.optical_flow_signature_dir)
            
            motion_pattern_db = MotionPatternDatabase(db_path=db_path, 
                                                     signature_dir=signature_dir)
            
            logger.info("Optical flow analyzer and database initialized")
        
        yield
        
        # Existing cleanup code
        # ...
        
    except Exception as e:
        logger.error(f"Error in lifespan: {e}")
        raise
    finally:
        # Cleanup resources
        # ...
        pass

# Modified detect_motion function

def detect_motion(frame, prev_frame=None):
    """
    Detect motion in a frame using background subtraction and optical flow.
    
    Args:
        frame: Current frame
        prev_frame: Previous frame (optional)
        
    Returns:
        Tuple of (motion_detected, regions, flow_features)
    """
    global prev_motion_frame, optical_flow_analyzer
    
    # Convert frame to grayscale if it's not already
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
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
    
    # Apply motion mask if enabled
    if camera_config.motion_mask_enabled and len(camera_config.motion_mask_points) > 0:
        # Create mask from points
        mask = np.zeros_like(thresh)
        points = np.array(camera_config.motion_mask_points)
        cv2.fillPoly(mask, [points], 255)
        # Apply mask
        thresh = cv2.bitwise_and(thresh, thresh, mask=mask)
    
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
    if motion_detected and camera_config.optical_flow_enabled and optical_flow_analyzer is not None:
        if prev_frame is not None:
            flow_features = optical_flow_analyzer.extract_flow(prev_frame, frame, regions)
    
    return motion_detected, regions, flow_features

# Add optical flow support to the FrameBuffer class

class FrameBuffer:
    """Buffer for frame data using a temporary file."""
    
    def __init__(self, app_state):
        """Initialize the frame buffer."""
        self.buffer = tempfile.NamedTemporaryFile(suffix='.jpg')
        self.lock = threading.Lock()
        self.app_state = app_state
        self.frame_count = 0
        self.last_frame = None  # Store last frame for optical flow analysis
    
    def write(self, buf, *args, **kwargs):
        """Write frame data to buffer."""
        with self.lock:
            # Existing code
            self.buffer.seek(0)
            self.buffer.write(buf)
            self.buffer.truncate()
            self.buffer.flush()
            
            # Update frame in app state
            self.buffer.seek(0)
            raw_img = read_image_data(self.buffer.read())
            self.app_state["last_frame"] = raw_img
            self.buffer.seek(0)
            
            # Process frame for motion detection if enabled
            if camera_config.motion_detection and raw_img is not None:
                # Store previous frame for optical flow
                prev_frame = self.last_frame
                self.last_frame = raw_img.copy()
                
                # Detect motion with optical flow
                motion_detected, motion_regions, flow_features = detect_motion(raw_img, prev_frame)
                
                # If motion detected, add to app state
                if motion_detected:
                    motion_time = datetime.datetime.now()
                    
                    # If we have flow features, add classification
                    classification = None
                    if flow_features and optical_flow_analyzer:
                        # Generate signature from current flow features
                        signature = optical_flow_analyzer.generate_motion_signature([flow_features])
                        if signature:
                            # Get classification
                            classification = optical_flow_analyzer.classify_motion(signature)
                    
                    # Add motion event with classification
                    motion_history.append((motion_time, motion_regions, classification))
                    
                    # Trim history if needed
                    if len(motion_history) > camera_config.motion_history_size * 2:
                        motion_history = motion_history[-camera_config.motion_history_size * 2:]
                    
                    # Log motion detection with classification
                    contour_areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) >= camera_config.motion_min_area]
                    if classification:
                        logger.info(f"Motion detected! Regions: {len(motion_regions)}, "
                                   f"Classification: {classification['label']} "
                                   f"({classification['confidence']:.2f})")
                    else:
                        logger.info(f"Motion detected! Regions: {len(motion_regions)}")
            
            # Return number of bytes written
            return len(buf)

# Update MotionEventRecorder in motion_storage.py

class MotionEventRecorder:
    """Records motion events to disk."""
    
    def __init__(self, config, circular_buffer):
        self.config = config
        self.buffer = circular_buffer
        self.current_event = None
        self.recording = False
        self.lock = threading.Lock()
        self.flow_features = []  # Store flow features during recording
    
    def start_recording(self, motion_regions, flow_features=None):
        """Start recording a new motion event."""
        with self.lock:
            if self.recording:
                return
                
            # Generate event ID
            event_time = datetime.now()
            event_id = f"motion_{event_time.strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
            
            # Create event directory
            event_dir = os.path.join(self.config.local_storage_path, event_id)
            os.makedirs(event_dir, exist_ok=True)
            
            # Create thumbnails directory
            thumb_dir = os.path.join(event_dir, "thumbnails")
            os.makedirs(thumb_dir, exist_ok=True)
            
            # Initialize event metadata
            self.current_event = {
                "id": event_id,
                "start_time": event_time.isoformat(),
                "regions": motion_regions,
                "frames": [],
                "path": event_dir,
                "thumbnails_path": thumb_dir,
                "flow_features": []  # Store flow features
            }
            
            # Reset flow features
            self.flow_features = []
            if flow_features:
                self.flow_features.append(flow_features)
            
            # Set recording flag
            self.recording = True
            
            logger.info(f"Started recording motion event {event_id}")
            
            # Create video writer
            self._create_video_writer(event_dir, event_id)
            
            # Save first frame as thumbnail
            self._save_thumbnail(0)
    
    def add_frame(self, frame, motion_regions=None, flow_features=None):
        """Add a frame to the current recording."""
        with self.lock:
            if not self.recording or self.current_event is None:
                return
                
            # Add frame to video
            if self.writer is not None and frame is not None:
                self.writer.write(frame)
                
            # Add flow features if provided
            if flow_features:
                self.flow_features.append(flow_features)
                
            # Update regions if provided
            if motion_regions:
                self.current_event["regions"] = motion_regions
                
            # Add frame to event
            frame_time = datetime.now()
            self.current_event["frames"].append({
                "timestamp": frame_time.isoformat(),
                "has_motion": motion_regions is not None and len(motion_regions) > 0
            })
            
            # Save thumbnail periodically
            if len(self.current_event["frames"]) % 30 == 0:
                self._save_thumbnail(len(self.current_event["frames"]) // 30)
    
    def stop_recording(self):
        """Stop the current recording."""
        with self.lock:
            if not self.recording or self.current_event is None:
                return
                
            # Release video writer
            if self.writer is not None:
                self.writer.release()
                self.writer = None
                
            # Calculate duration
            end_time = datetime.now()
            start_time = datetime.fromisoformat(self.current_event["start_time"])
            duration = (end_time - start_time).total_seconds()
            
            # Update event metadata
            self.current_event["end_time"] = end_time.isoformat()
            self.current_event["duration"] = duration
            
            # Process optical flow data if available
            motion_classification = None
            if self.flow_features and len(self.flow_features) > 0 and optical_flow_analyzer is not None:
                try:
                    # Generate motion signature
                    motion_signature = optical_flow_analyzer.generate_motion_signature(self.flow_features)
                    
                    if motion_signature:
                        # Classify motion
                        motion_classification = optical_flow_analyzer.classify_motion(motion_signature)
                        
                        # Save flow signature
                        if storage_config.store_optical_flow_data and motion_pattern_db is not None:
                            # Save to database
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
                        if storage_config.save_flow_visualizations and len(self.flow_features) > 0:
                            # Create a visualization of the motion
                            last_frame = self.buffer.get_frame(0)  # Get latest frame
                            if last_frame is not None and self.flow_features[-1] is not None:
                                flow_vis = optical_flow_analyzer.visualize_flow(
                                    last_frame, self.flow_features[-1])
                                
                                # Save visualization
                                flow_vis_path = os.path.join(self.current_event["path"], "flow.jpg")
                                cv2.imwrite(flow_vis_path, flow_vis)
                except Exception as e:
                    logger.error(f"Error processing optical flow data: {e}")
            
            # Add motion classification to metadata
            if motion_classification:
                self.current_event["motion_classification"] = motion_classification
                
                # Log classification result
                label = motion_classification["label"]
                confidence = motion_classification["confidence"]
                logger.info(f"Motion event {self.current_event['id']} classified as: "
                          f"{label} ({confidence:.2f})")
            
            # Save metadata
            metadata_path = os.path.join(self.current_event["path"], "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(self.current_event, f, indent=2)
                
            # Reset state
            event_id = self.current_event["id"]
            self.current_event = None
            self.recording = False
            self.flow_features = []
            
            logger.info(f"Stopped recording motion event {event_id}")

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
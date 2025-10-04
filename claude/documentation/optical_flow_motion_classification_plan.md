# Optical Flow-Based Motion Classification Plan

## 1. Optical Flow Background & Selection

There are several optical flow approaches that could work for this application:

1. **Dense Optical Flow** (Farneback method): Computes flow vectors for all pixels
   - Advantages: Rich motion information, good for general pattern extraction
   - Disadvantages: Computationally expensive

2. **Sparse Optical Flow** (Lucas-Kanade method): Computes flow for selected points
   - Advantages: More efficient, works well for tracking specific features
   - Disadvantages: Less comprehensive motion information

3. **FlowNet/DeepFlow**: Deep learning based optical flow
   - Advantages: Very accurate
   - Disadvantages: Too computationally heavy for Raspberry Pi

For this application, a hybrid approach using Lucas-Kanade method with careful feature point selection would balance performance and accuracy.

## 2. Motion Pattern Recognition Architecture

### 2.1 Current System Integration Points

The current system has several key components we can leverage:
- `detect_motion()` in fastapi_mjpeg_server_with_storage.py - identifies motion regions
- FrameBuffer class - handles frame processing and storage
- CircularFrameBuffer in motion_storage.py - maintains a history of recent frames
- MotionEventRecorder - handles motion event recording

Our optical flow analysis would integrate at these points:
1. After initial motion detection to analyze motion patterns
2. Store motion signature with event metadata
3. Classify motion patterns during or after recording

## 3. Motion Pattern Extraction and Representation

### 3.1 Feature Selection and Tracking

1. **Feature Selection:**
   - Use Shi-Tomasi corner detection (cv2.goodFeaturesToTrack)
   - Focus on corners/features within detected motion regions
   - Track 50-100 feature points per motion region

2. **Flow Calculation:**
   - Apply Lucas-Kanade optical flow (cv2.calcOpticalFlowPyrLK)
   - Track features across 5-10 frame sequences
   - Calculate displacement vectors, velocity, and acceleration

3. **Motion Signature Extraction:**
   - **Spatial distribution:** Grid-based histogram of flow vectors (8x8 grid)
   - **Temporal patterns:** 
     - Average velocity and acceleration over time
     - Direction changes frequency
     - Flow consistency (how uniform is the motion)
   - **Motion shape:** Convex hull of motion trajectory

### 3.2 Motion Signature Representation

Create a compact numerical representation that captures motion characteristics:
- **Flow histogram:** 8x8 grid x 8 direction bins = 512-dimensional vector
- **Temporal features:** 20-30 statistical measures of flow over time
- **Dimensionality reduction:** Apply PCA to reduce to 50-100 dimensions

## 4. Motion Pattern Classification

### 4.1 Classification Approach

1. **Initial Classification Strategy:**
   - Unsupervised clustering (K-Means, DBSCAN) to identify natural groups
   - Start with no labels, just cluster IDs (e.g., "Motion Type 1", "Motion Type 2")
   - Allow user to label clusters over time through UI

2. **Advanced Classification:**
   - Semi-supervised learning approach
   - Train a simple classifier (SVM or Random Forest) on labeled examples
   - Use confidence scores to identify uncertain classifications
   - Update model periodically as more labeled data becomes available

3. **Classification Features:**
   - Use motion signatures as feature vectors
   - Extract additional meta-features:
     - Time of day
     - Duration of motion
     - Size of moving object
     - Speed of movement

### 4.2 Similarity Metrics for Comparing Motion Patterns

- **Cosine similarity:** For comparing flow histogram vectors
- **Dynamic Time Warping:** For comparing temporal sequences
- **Mahalanobis distance:** For comparing statistical features with covariance

## 5. Motion Pattern Database

### 5.1 Database Structure

Create a lightweight database to store and retrieve motion patterns:

```python
class MotionPattern:
    id: str                    # Unique identifier
    timestamp: datetime        # When pattern was recorded
    feature_vector: np.array   # Compact motion signature
    raw_flow_data: dict        # Optional raw flow data (for retraining)
    classification: str        # Current classification (e.g., "car", "person")
    confidence: float          # Classification confidence
    metadata: dict             # Additional information (time of day, etc.)
```

### 5.2 Storage and Indexing

1. **Immediate storage:**
   - SQLite database for structured data
   - NumPy arrays stored as binary blobs or JSON
   - Option to store only signatures, not raw flow data

2. **Indexing for fast retrieval:**
   - Simple index on classification field
   - Consider approximate nearest neighbor indexing (Annoy, FAISS) for similarity search
   - Time-based indexing for temporal queries

### 5.3 Pattern Evolution and Learning

1. **Incremental learning:**
   - Start with small clusters that grow over time
   - Periodically recalculate centroids and boundaries
   - Track pattern changes over time (e.g., seasonal variations)

2. **Feedback mechanism:**
   - Allow users to confirm or correct classifications
   - Weighted recent examples more heavily
   - Implement active learning to request labels for uncertain cases

## 6. Integration with Existing Code

### 6.1 New Module Structure

Create a new module called `optical_flow_analyzer.py` with these components:

1. **OpticalFlowAnalyzer class:**
   - Methods for feature detection, tracking and flow calculation
   - Motion signature extraction
   - Classification interface

2. **MotionPatternDatabase class:**
   - Storage and retrieval of motion patterns
   - Pattern matching and similarity search
   - Model training and updating

### 6.2 Integration Points

1. **Main integration with detect_motion:**
```python
def detect_motion(frame):
    # Existing motion detection code
    # ...
    
    if motion_detected:
        # Extract optical flow data
        flow_features = optical_flow_analyzer.extract_flow(prev_frame, frame, motion_regions)
        
        # Get real-time classification if possible
        classification = optical_flow_analyzer.classify_motion(flow_features)
        
        # Enhanced motion regions with classification
        enhanced_regions = [(x, y, w, h, classification) for (x, y, w, h) in motion_regions]
        
        return motion_detected, enhanced_regions, flow_features
    # ...
```

2. **Integration with FrameBuffer.write:**
```python
def write(self, buf, *args, **kwargs):
    # Existing code
    # ...
    
    # Process frame for motion detection if enabled
    if camera_config.motion_detection and raw_img is not None:
        motion_detected, motion_regions, flow_features = detect_motion(raw_img)
        
        # Store flow features temporarily
        if motion_detected:
            self.current_flow_features = flow_features
    # ...
```

3. **Integration with motion_recorder:**
```python
def start_recording(self, motion_regions, flow_features=None):
    # Existing code
    # ...
    
    self.current_event["flow_features"] = flow_features
    
    # Start flow tracking for this event
    self.flow_tracker.start_tracking(self.current_event["id"])
    # ...

def add_frame(self, frame, motion_regions, flow_features=None):
    # Existing code
    # ...
    
    if flow_features:
        self.flow_tracker.add_flow_data(self.current_event["id"], flow_features)
    # ...
    
def stop_recording(self):
    # Existing code
    # ...
    
    # Complete flow analysis
    if hasattr(self, 'flow_tracker'):
        flow_signature = self.flow_tracker.generate_signature(self.current_event["id"])
        classification = self.flow_tracker.classify_motion(flow_signature)
        self.current_event["motion_signature"] = flow_signature
        self.current_event["motion_classification"] = classification
    # ...
```

## 7. Storage Structure for Optical Flow Metadata

### 7.1 Event Metadata Extension

Extend the existing metadata.json structure to include optical flow information:

```json
{
  "id": "motion-20250923-164201-123",
  "start_time": "2025-09-23T16:42:01.123",
  "end_time": "2025-09-23T16:42:10.456",
  "duration": 9.333,
  "regions": [...],
  "frame_count": 280,
  "resolution": {"width": 640, "height": 480},
  "has_thumbnails": true,
  "processed": true,
  
  "motion_analysis": {
    "classification": {
      "label": "car",
      "confidence": 0.92,
      "alternatives": [
        {"label": "truck", "confidence": 0.05},
        {"label": "person", "confidence": 0.02}
      ]
    },
    "motion_characteristics": {
      "avg_velocity": 45.6,
      "direction": "left-to-right",
      "path_complexity": 0.12,
      "flow_consistency": 0.89
    },
    "signature_hash": "f8a2e3b7d6c5a4e1b2c3d4e5f6a7b8c9"
  }
}
```

### 7.2 Optical Flow Data Storage

Create separate files for detailed flow data:

1. **Compact signature file** (`motion-[ID]-signature.npz`):
   - Numpy compressed format
   - Contains feature vector for quick classification
   - Small enough to be included in database

2. **Optional full flow data** (`motion-[ID]-flow.npz`):
   - Complete flow vectors
   - Only kept for training examples or uncertain cases
   - Can be purged to save space

3. **Flow visualization** (`motion-[ID]-flow.jpg`):
   - Color-coded visualization of flow patterns
   - Similar to OpenCV flow visualization
   - Used for UI display and human verification

## 8. UI Enhancements for Motion Classification

### 8.1 Real-Time Display

Add real-time motion classification information to the existing UI:

1. **Motion Alert Enhancement:**
```html
<div class="motion-alert" id="motionAlert">
    Motion Detected: <span id="motionClassification">Unknown</span>
    <div class="confidence-bar">
        <div id="confidenceLevel" class="confidence-bar-fill" style="width: 0%"></div>
    </div>
</div>
```

2. **Motion History Enhancement:**
```html
<div class="motion-event">
    <div class="motion-event-header">
        <span class="motion-timestamp">{{event.timestamp}}</span>
        <span class="motion-classification {{event.classification_confidence > 0.7 ? 'high-confidence' : 'low-confidence'}}">
            {{event.classification}}
        </span>
    </div>
    <div class="motion-event-details">
        <span>{{event.regions.length}} regions</span>
        <span>{{event.duration.toFixed(1)}}s</span>
        <div class="motion-thumbnail-container">
            <img class="motion-thumbnail" src="/storage/events/{{event.id}}/thumbnails/thumb_0.jpg">
            <img class="motion-flow" src="/storage/events/{{event.id}}/flow.jpg">
        </div>
    </div>
</div>
```

### 8.2 Motion Classification UI

Create a new tab in the UI for motion classification management:

1. **Classification Dashboard:**
   - Display of known motion types with sample images
   - Statistics on frequency of each type
   - Distribution by time of day/week

2. **Pattern Management:**
   - Ability to rename/merge motion types
   - Mark false positives
   - View similar events

3. **Filtering Options:**
   - Filter motion events by classification
   - Set recording thresholds by type
   - Configure alerts for specific motion types

### 8.3 JavaScript Updates

Add JavaScript to handle motion classification:

```javascript
// Update motion classification in real-time
setInterval(() => {
    fetch('/motion_status')
        .then(response => response.json())
        .then(data => {
            // Update motion alert
            motionAlert.style.display = data.motion_detected ? 'block' : 'none';
            
            if (data.motion_detected && data.classification) {
                // Show classification
                document.getElementById('motionClassification').textContent = data.classification.label;
                
                // Update confidence bar
                const confidenceBar = document.getElementById('confidenceLevel');
                const confidence = data.classification.confidence * 100;
                confidenceBar.style.width = `${confidence}%`;
                confidenceBar.textContent = `${confidence.toFixed(0)}%`;
                
                // Set color based on confidence
                if (confidence > 80) {
                    confidenceBar.style.backgroundColor = '#4CAF50';  // Green
                } else if (confidence > 50) {
                    confidenceBar.style.backgroundColor = '#ff9800';  // Orange
                } else {
                    confidenceBar.style.backgroundColor = '#f44336';  // Red
                }
            }
            
            // Update motion history with classifications
            updateMotionHistoryWithClassifications(data.motion_history);
        });
}, 1000);
```

## 9. Performance Optimization Techniques

### 9.1 Computation Efficiency

1. **Selective Processing:**
   - Only calculate optical flow when motion is detected
   - Process flow on a subset of frames (e.g., every 3rd frame)
   - Use lower resolution for flow calculation (e.g., 320x240)

2. **Feature Point Optimization:**
   - Limit maximum number of feature points (50-100 per motion region)
   - Distribute points evenly using grid-based selection
   - Filter out low-quality features before tracking

3. **Motion Signature Caching:**
   - Cache intermediate calculations during an event
   - Reuse flow calculations between frames when possible
   - Implement early signature estimation for faster classification

### 9.2 Threading and Process Management

1. **Background Processing:**
   - Move intensive calculations to background thread
   - Queue motion events for analysis
   - Process when CPU usage is lower

2. **Tiered Processing:**
   - Real-time tier: Quick classification using simplified features
   - Background tier: Detailed signature extraction
   - Batch tier: Model updating and database maintenance

3. **Resource Management:**
   - Monitor CPU and memory usage
   - Dynamically adjust processing based on system load
   - Implement backpressure mechanisms for high motion scenarios

### 9.3 Storage Optimization

1. **Data Pruning:**
   - Store full flow data only for uncertain or novel patterns
   - Automatically purge detailed flow data for common patterns
   - Keep only compact signatures for typical events

2. **Incremental Updates:**
   - Use incremental PCA for feature updating
   - Periodically rebuild models with consolidated data
   - Schedule heavy processing during quiet periods

## 10. Evaluation and Testing Approach

### 10.1 Performance Benchmarking

1. **Computational Overhead Measurement:**
   - Baseline: Measure system performance without optical flow analysis
   - Incremental: Add optical flow and measure resource impact
   - Profile code to identify bottlenecks

2. **Timing Analysis:**
   - Measure processing time for each component:
     - Feature detection
     - Flow calculation
     - Signature extraction
     - Classification
   - Set performance budgets for each stage

3. **Hardware Testing:**
   - Test on different Raspberry Pi models (3B+, 4, etc.)
   - Verify memory usage stays within constraints
   - Measure power consumption impact

### 10.2 Classification Accuracy Evaluation

1. **Ground Truth Dataset:**
   - Create labeled dataset of common motion types
   - Include diverse examples of each class
   - Cover different lighting conditions and times of day

2. **Cross-Validation:**
   - Implement k-fold cross-validation
   - Measure precision, recall, and F1-score for each class
   - Create confusion matrix to identify misclassifications

3. **Comparison Metrics:**
   - Compare with simple motion detection (baseline)
   - Benchmark against lightweight YOLO if feasible
   - Evaluate false positive/negative rates

### 10.3 Long-term Robustness Testing

1. **Continuous Operation Test:**
   - Run system continuously for extended periods (days/weeks)
   - Monitor for memory leaks, crashes, or degradation
   - Verify consistent classification over time

2. **Seasonal and Environmental Variations:**
   - Test during day/night transitions
   - Test during different weather conditions
   - Verify adaptation to seasonal changes in lighting

3. **Edge Case Handling:**
   - Test with unusual motion patterns
   - Test with multiple simultaneous objects
   - Test with very brief or very extended motion

### 10.4 User Experience Evaluation

1. **Classification Usefulness:**
   - Measure how often classifications match user expectations
   - Track manual corrections to evaluate learning effectiveness
   - Survey users on system utility and accuracy

2. **System Responsiveness:**
   - Measure end-to-end latency from motion to classification
   - Verify UI updates happen in near real-time
   - Check for any perceived system slowdowns

## 11. Implementation Roadmap

### Phase 1: Core Optical Flow Integration
1. Create optical_flow_analyzer.py module with basic functionality
2. Integrate with existing motion detection
3. Implement basic flow visualization
4. Develop test harness for performance evaluation

### Phase 2: Motion Signature and Storage
1. Implement feature extraction and signature generation
2. Create storage structures for flow metadata
3. Design and implement database component
4. Add flow visualization to web UI

### Phase 3: Classification Engine
1. Implement unsupervised clustering for initial patterns
2. Develop similarity metrics and matching algorithms
3. Add classification results to motion events
4. Build classification management UI components

### Phase 4: Refinement and Optimization
1. Optimize performance for Raspberry Pi
2. Implement feedback mechanisms for corrections
3. Add learning capabilities to improve over time
4. Create administrative tools for pattern management

### Phase 5: Extended Features
1. Add notification filtering by motion type
2. Implement advanced visualization tools
3. Develop statistical reporting on motion patterns
4. Add scheduling options based on motion types

## 12. Conclusion

The optical flow-based motion classification approach offers several advantages:

1. **Hardware Efficiency:**
   - More computationally efficient than deep learning approaches
   - Works well on resource-constrained devices like Raspberry Pi
   - Minimal additional hardware requirements

2. **Self-Learning Capabilities:**
   - Adapts to the specific camera environment
   - Improves classification over time
   - Can learn new motion patterns without explicit training

3. **Contextual Awareness:**
   - Leverages spatial patterns specific to the camera location
   - Can distinguish similar objects by how they move in the scene
   - Works well with fixed camera positions

4. **Extensibility:**
   - Can be combined with simple object detection for hybrid approach
   - Allows for user customization and labeling
   - Provides foundation for more advanced features

This approach represents a balanced solution between simple motion detection and resource-intensive deep learning models, particularly well-suited for edge devices with fixed camera positions.
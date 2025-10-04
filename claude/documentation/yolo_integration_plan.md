# Plan for Replacing Motion Detection with YOLO Object Detection

## 1. YOLO Model Selection for Raspberry Pi

For a Raspberry Pi environment, we need to consider:
- YOLOv5-nano or YOLOv8-nano - optimized for edge devices
- ONNX runtime for optimized inference
- Consider TensorRT for acceleration if using Pi 4 with 4GB+ RAM
- Alternative: YOLOv4-tiny with OpenCV DNN module

## 2. Integration Architecture

### 2.1 Current Architecture Analysis
The existing motion detection occurs in `detect_motion()` function in fastapi_mjpeg_server_with_storage.py and produces:
- Boolean motion detection status
- Regions (bounding boxes) of detected motion

### 2.2 Core Integration Points
1. Replace `detect_motion()` with `detect_objects()`
2. Modify frame processing in FrameBuffer.write()
3. Extend motion_regions to include object class and confidence
4. Update CircularFrameBuffer in motion_storage.py to store object metadata
5. Modify motion event recording to include detected objects

## 3. Code Modifications Required

### 3.1 Dependencies and Initialization
```
pip install ultralytics opencv-python onnxruntime
```

### 3.2 Configuration Updates
- Add YOLO-specific parameters to CameraConfig:
  - Model selection (nano, small, medium)
  - Confidence threshold (0.25-0.5)
  - IOU threshold (0.45-0.7)
  - Classes to detect (people, vehicles, animals)
  - Device (CPU/GPU)

### 3.3 Model Loading
- Load at startup in FastAPI lifespan
- Handle model loading errors gracefully
- Store model in app state

### 3.4 Frame Processing Pipeline
- Process every Nth frame (e.g., 5-10) to maintain performance
- Run inference on resized frames (320x320, 416x416, or 640x640)
- Filter detections by confidence and specified classes

### 3.5 Storage Modifications
- Extend metadata.json to include:
  - Object counts by class
  - Object bounding boxes with confidence
  - Highest confidence detection timestamp

### 3.6 UI Enhancements
- Update HTML/JS to show detected object classes
- Add object filters in UI
- Display confidence scores
- Implement object highlighting with class labels

## 4. Performance Considerations

### 4.1 Optimization Techniques
- Run inference at lower resolution (320x320 or 416x416)
- Implement frame skipping (process 1 in N frames)
- Use quantized models (INT8)
- Consider background thread for inference
- Buffer inference results between real-time frames

### 4.2 Hardware Recommendations
- Raspberry Pi 4 with 4GB+ RAM minimum
- Consider Coral USB Accelerator
- Optional: GPU acceleration via NVIDIA Jetson Nano

## 5. Implementation Strategy

1. Create separate object_detection.py module
2. Implement staged rollout:
   - First pass: Basic YOLO integration with simple object detection
   - Second pass: Add object filtering and metadata storage
   - Third pass: UI enhancements and performance optimizations
3. Add feature flags to toggle between:
   - Traditional motion detection
   - Object detection
   - Hybrid approach

## 6. Testing Strategy

1. Benchmark performance on target hardware
2. Test with various lighting conditions
3. Validate detection accuracy against ground truth
4. Test system stability over extended periods
5. Compare storage requirements with original implementation

This implementation will transform the system from simple motion detection to intelligent object recognition, enabling more precise video capture based on specific entities of interest.
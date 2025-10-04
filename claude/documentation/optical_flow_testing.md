# Optical Flow Analysis Testing Guide

This guide explains how to run tests for the Optical Flow Analyzer module, which implements motion-based object classification.

## Unit Testing

The unit test suite provides comprehensive testing of the `OpticalFlowAnalyzer` and `MotionPatternDatabase` classes. It verifies the functionality of flow calculation, signature generation, classification, and database operations.

### Running Unit Tests

To run the unit tests, use the following pixi commands:

```bash
# Run all tests for the optical flow analyzer
pixi run test-optical-flow

# Run tests with coverage report
pixi run test-optical-flow-cov
```

The unit tests use synthetic frames and mocked data to validate the optical flow analysis functionality without requiring actual video input.

## Performance Testing

The `test_optical_flow.py` script provides tools for evaluating the performance of the optical flow analysis system with real video files or camera input.

### Testing with Video Files

To test the optical flow analyzer with a video file:

```bash
# Test with a specific video file
pixi run test-video path/to/video.mp4
```

The script will process the video, extract optical flow features, generate motion signatures, and classify the motion patterns. The results will be displayed in a visualization window and saved to the `test_results` directory.

### Testing with Camera Input

To test with a live camera feed:

```bash
# Test with the default camera (index 0) for 60 seconds
pixi run test-camera 60
```

This will capture video from the camera, process it for optical flow, and display the results in real-time.

### Benchmarking

To evaluate the system's performance on a set of labeled videos:

```bash
# Benchmark with videos in a directory
pixi run benchmark test_videos/
```

The benchmark mode requires a set of labeled videos, either:
- A directory with subdirectories named by category (e.g., `test_videos/car/`, `test_videos/person/`), or
- A directory with a `labels.json` file mapping video files to labels

## Test Environment

The tests use a dedicated `optical_flow_test` environment defined in `pixi.toml`, which includes additional dependencies like matplotlib and scikit-learn for visualization and data analysis.

To activate the environment:

```bash
pixi shell --feature optical_flow_test
```

## Creating Test Data

For benchmarking and evaluation, you should create a test dataset with videos of different motion types:

1. Create a `test_videos` directory
2. Organize videos by category (e.g., `car`, `person`, `animal`, `environment`)
3. Or create a `labels.json` file with the format:
   ```json
   {
     "video1.mp4": "car",
     "video2.mp4": "person",
     "video3.mp4": "animal"
   }
   ```

## Interpreting Results

The test results include:

- Processing speed (frames per second)
- Flow calculation time
- Classification time
- Classification accuracy (if ground truth labels are available)
- Visualization of optical flow patterns
- Classification results

For benchmark results, a summary will be displayed and detailed results will be saved to JSON files in the `test_results` directory.
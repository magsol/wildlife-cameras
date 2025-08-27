# Raspberry Pi Camera Motion Storage System Tests

This directory contains tests for the Raspberry Pi camera motion storage system. The tests cover all major components of the system:

1. **Motion Storage Module**: Core functionality for motion detection, storage, and transfer
2. **Storage Server**: Remote server for receiving and storing videos
3. **MJPEG Server with Storage**: Main server with integrated motion storage

## Test Structure

### Unit Tests for Motion Storage Module

The `test_motion_storage.py` file contains tests for:
- `StorageConfig` class: Configuration settings
- `CircularFrameBuffer` class: RAM buffering of frames
- `MotionEventRecorder` class: Recording and saving motion events
- `WiFiMonitor` class: WiFi signal monitoring and throttling
- `TransferManager` class: Network transfer management

### Unit Tests for Storage Server

The `test_storage_server.py` file contains tests for:
- Server info endpoint
- Authentication and authorization
- Chunked upload workflow (init, upload chunks, finalize)
- Complete file upload
- Storage statistics
- Event listing and management

### Unit Tests for MJPEG Server with Storage

The `test_fastapi_mjpeg_server_with_storage.py` file contains tests for:
- Web interface endpoints
- Status and configuration endpoints
- Motion detection functions
- Frame buffer functionality
- Command line argument parsing

## Running the Tests

To run all tests:

```bash
cd /path/to/picamera2/examples
pytest -v tests/
```

To run specific test files:

```bash
pytest -v tests/test_motion_storage.py
pytest -v tests/test_storage_server.py
pytest -v tests/test_fastapi_mjpeg_server_with_storage.py
```

To run a specific test class or test:

```bash
pytest -v tests/test_motion_storage.py::TestMotionEventRecorder
pytest -v tests/test_motion_storage.py::TestMotionEventRecorder::test_start_recording
```

## Test Coverage

The tests aim to cover all critical functionality including:
- Normal operation paths
- Error handling
- Edge cases
- Configuration validation

Key areas tested include:
- Motion detection and event recording
- Frame processing and buffering
- Network transfer with throttling and chunking
- WiFi signal strength monitoring
- Storage management and cleanup
- API endpoints and authentication

## Mocking

The tests use extensive mocking to avoid dependencies on:
- Physical camera hardware
- Network connectivity
- File system operations
- System time

This allows the tests to run in any environment without special hardware or configuration.

## Adding New Tests

When adding new functionality, corresponding tests should be added to maintain coverage. Follow these guidelines:

1. Organize tests by component and class
2. Use descriptive test method names
3. Cover both success and failure cases
4. Mock external dependencies
5. Clean up any temporary resources

## Known Limitations

- Some tests mock hardware that would normally be present on a Raspberry Pi
- Network transfer tests rely on mocked responses rather than actual network calls
- The streaming nature of some responses makes them difficult to test fully
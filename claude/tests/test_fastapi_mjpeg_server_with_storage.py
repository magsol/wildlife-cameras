import os
import sys
import unittest
import tempfile
import shutil
import time
import datetime
import json
from unittest.mock import MagicMock, patch, mock_open
import numpy as np
import cv2

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock the picamera2 module before importing
sys.modules['picamera2'] = MagicMock()
sys.modules['picamera2.encoders'] = MagicMock()
sys.modules['picamera2.outputs'] = MagicMock()

# Import module to test
import fastapi_mjpeg_server_with_storage as server
from fastapi_mjpeg_server_with_storage import app


class MockPicamera2:
    """Mock class for Picamera2"""
    def __init__(self):
        self.configured = False
        self.recording = False
        
    def create_video_configuration(self, **kwargs):
        return {"config": "test"}
        
    def configure(self, config):
        self.configured = True
        
    def start_recording(self, encoder, output):
        self.recording = True
        
    def stop_recording(self):
        self.recording = False


class TestMJPEGServer(unittest.TestCase):
    """Test cases for FastAPI MJPEG Server with Storage"""

    def setUp(self):
        """Set up test environment"""
        # Create temp directory for test
        self.test_dir = tempfile.mkdtemp()
        
        # Patch Picamera2 before creating client
        self.picam_patcher = patch('fastapi_mjpeg_server_with_storage.Picamera2', 
                                  return_value=MockPicamera2())
        self.mock_picam = self.picam_patcher.start()
        
        # Patch initialize_camera
        self.init_camera_patcher = patch('fastapi_mjpeg_server_with_storage.initialize_camera',
                                        return_value=MockPicamera2())
        self.mock_init_camera = self.init_camera_patcher.start()
        
        # Patch motion storage initialize function
        self.init_storage_patcher = patch('fastapi_mjpeg_server_with_storage.init_motion_storage')
        self.mock_init_storage = self.init_storage_patcher.start()
        self.mock_init_storage.return_value = {
            'frame_buffer': MagicMock(),
            'motion_recorder': MagicMock(),
            'transfer_manager': MagicMock(),
            'wifi_monitor': MagicMock(),
            'storage_config': MagicMock(),
            'modify_frame_buffer_write': lambda x: x
        }
        
        # Set camera_initialized to True for testing
        server.camera_initialized = True
        
        # Create test client
        self.client = TestClient(app)

    def tearDown(self):
        """Clean up after test"""
        # Stop patchers
        self.picam_patcher.stop()
        self.init_camera_patcher.stop()
        self.init_storage_patcher.stop()
        
        # Remove temp directory
        shutil.rmtree(self.test_dir)

    def test_index_route(self):
        """Test index route returns HTML"""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/html; charset=utf-8")
        self.assertIn(b"Raspberry Pi Camera Stream", response.content)

    def test_status_endpoint(self):
        """Test status endpoint returns correct data"""
        response = self.client.get("/status")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("camera_initialized", data)
        self.assertIn("camera_config", data)
        self.assertTrue(data["camera_initialized"])
        self.assertEqual(data["camera_config"]["width"], 640)
        self.assertEqual(data["camera_config"]["height"], 480)
        self.assertEqual(data["camera_config"]["frame_rate"], 30)

    def test_motion_status_endpoint(self):
        """Test motion status endpoint"""
        # Add some test motion history
        server.motion_history = [
            (datetime.datetime.now(), [(10, 10, 30, 30)]),
            (datetime.datetime.now(), [(20, 20, 40, 40)])
        ]
        
        response = self.client.get("/motion_status")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("motion_detected", data)
        self.assertIn("motion_history", data)
        self.assertEqual(len(data["motion_history"]), len(server.motion_history))

    def test_config_update_endpoint(self):
        """Test configuration update endpoint"""
        # Original config values
        original_timestamp = server.camera_config.show_timestamp
        original_motion_detection = server.camera_config.motion_detection
        
        # New config values
        new_config = {
            "show_timestamp": not original_timestamp,
            "motion_detection": not original_motion_detection,
            "motion_threshold": 35,
            "timestamp_position": "top-left"
        }
        
        response = self.client.post(
            "/config",
            json=new_config
        )
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(server.camera_config.show_timestamp, new_config["show_timestamp"])
        self.assertEqual(server.camera_config.motion_detection, new_config["motion_detection"])
        self.assertEqual(server.camera_config.motion_threshold, new_config["motion_threshold"])
        self.assertEqual(server.camera_config.timestamp_position, new_config["timestamp_position"])

    def test_config_invalid_value(self):
        """Test configuration update with invalid value"""
        # Try to set invalid rotation value
        invalid_config = {
            "rotation": 45  # Only 0, 90, 180, 270 are valid
        }
        
        response = self.client.post(
            "/config",
            json=invalid_config
        )
        
        self.assertEqual(response.status_code, 422)  # Validation error

    def test_add_timestamp(self):
        """Test add_timestamp function"""
        # Create test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test with timestamp enabled
        server.camera_config.show_timestamp = True
        result = server.add_timestamp(frame)
        
        # Result should be a different object than the input
        self.assertIsNot(result, frame)
        
        # Test with timestamp disabled
        server.camera_config.show_timestamp = False
        result = server.add_timestamp(frame)
        
        # Result should be the same as input
        self.assertIs(result, frame)
        
        # Test with different positions
        server.camera_config.show_timestamp = True
        for position in ["top-left", "top-right", "bottom-left", "bottom-right"]:
            server.camera_config.timestamp_position = position
            result = server.add_timestamp(frame)
            self.assertIsNotNone(result)

    def test_detect_motion(self):
        """Test detect_motion function"""
        # Create previous frame
        prev_frame = np.zeros((480, 640), dtype=np.uint8)
        server.prev_frame = prev_frame
        
        # Create current frame with motion
        current_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a rectangle that will be detected as motion
        cv2.rectangle(current_frame, (100, 100), (200, 200), (255, 255, 255), -1)
        
        # Set motion detection parameters
        server.camera_config.motion_detection = True
        server.camera_config.motion_threshold = 20
        server.camera_config.motion_min_area = 100
        
        # Test with motion
        motion_detected, regions = server.detect_motion(current_frame)
        
        self.assertTrue(motion_detected)
        self.assertTrue(len(regions) > 0)
        
        # Test with motion detection disabled
        server.camera_config.motion_detection = False
        motion_detected, regions = server.detect_motion(current_frame)
        
        self.assertFalse(motion_detected)
        self.assertEqual(len(regions), 0)

    def test_first_motion_frame(self):
        """Test detect_motion with first frame"""
        # Reset prev_frame to None
        server.prev_frame = None
        
        # Create frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Set motion detection parameters
        server.camera_config.motion_detection = True
        
        # First frame should not detect motion
        motion_detected, regions = server.detect_motion(frame)
        
        self.assertFalse(motion_detected)
        self.assertEqual(len(regions), 0)
        
        # prev_frame should now be initialized
        self.assertIsNotNone(server.prev_frame)

    @patch('fastapi_mjpeg_server_with_storage.frame_buffer')
    def test_stream_endpoint_max_clients(self, mock_frame_buffer):
        """Test stream endpoint with maximum clients reached"""
        # Mock register_client to return False (max clients)
        mock_frame_buffer.register_client.return_value = False
        
        response = self.client.get("/stream")
        self.assertEqual(response.status_code, 503)  # Service unavailable

    @patch('fastapi_mjpeg_server_with_storage.frame_buffer')
    @patch('asyncio.sleep', side_effect=lambda x: None)  # Don't actually sleep
    def test_stream_endpoint(self, mock_sleep, mock_frame_buffer):
        """Test stream endpoint"""
        # Mock register_client to return True
        mock_frame_buffer.register_client.return_value = True
        
        # Mock get_frame to return a test JPEG
        test_frame = b'--frame\r\nContent-Type: image/jpeg\r\n\r\ntest jpeg content\r\n'
        mock_frame_buffer.get_frame.return_value = test_frame
        
        # Need to use a context manager due to streaming response
        with self.client.stream("GET", "/stream") as response:
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["content-type"], 
                           "multipart/x-mixed-replace; boundary=frame")
            
            # Can't easily test the streaming content


class TestParseArguments(unittest.TestCase):
    """Test cases for command line argument parsing"""

    def test_default_arguments(self):
        """Test parsing with default arguments"""
        with patch('sys.argv', ['fastapi_mjpeg_server_with_storage.py']):
            args = server.parse_arguments()
            
            self.assertEqual(args.width, 640)
            self.assertEqual(args.height, 480)
            self.assertEqual(args.fps, 30)
            self.assertEqual(args.rotation, 0)
            self.assertEqual(args.host, "0.0.0.0")
            self.assertEqual(args.port, 8000)
            self.assertEqual(args.max_clients, 10)
            self.assertFalse(args.no_timestamp)
            self.assertFalse(args.no_motion)

    def test_custom_arguments(self):
        """Test parsing with custom arguments"""
        with patch('sys.argv', [
            'fastapi_mjpeg_server_with_storage.py',
            '--width', '1280',
            '--height', '720',
            '--fps', '15',
            '--rotation', '180',
            '--port', '8080',
            '--no-timestamp',
            '--no-motion',
            '--storage-path', '/custom/path',
            '--max-storage', '2000'
        ]):
            args = server.parse_arguments()
            
            self.assertEqual(args.width, 1280)
            self.assertEqual(args.height, 720)
            self.assertEqual(args.fps, 15)
            self.assertEqual(args.rotation, 180)
            self.assertEqual(args.port, 8080)
            self.assertTrue(args.no_timestamp)
            self.assertTrue(args.no_motion)
            self.assertEqual(args.storage_path, '/custom/path')
            self.assertEqual(args.max_storage, 2000)


class TestFrameBuffer(unittest.TestCase):
    """Test cases for FrameBuffer class"""

    def setUp(self):
        """Set up test environment"""
        self.buffer = server.FrameBuffer()

    def test_initialization(self):
        """Test FrameBuffer initialization"""
        self.assertIsNone(self.buffer.frame)
        self.assertIsNone(self.buffer.raw_frame)
        self.assertEqual(self.buffer.last_access_times, {})
        self.assertEqual(self.buffer.max_size, 5)

    def test_register_client(self):
        """Test client registration"""
        # First client should be accepted
        self.assertTrue(self.buffer.register_client("client1"))
        self.assertIn("client1", self.buffer.last_access_times)
        
        # Fill up to max_clients
        server.camera_config.max_clients = 2
        self.assertTrue(self.buffer.register_client("client2"))
        
        # Next client should be rejected
        self.assertFalse(self.buffer.register_client("client3"))
        
        # After timeout, inactive clients should be removed
        server.camera_config.client_timeout = -1  # Force timeout
        self.assertTrue(self.buffer.register_client("client3"))
        
    @patch('cv2.imencode')
    @patch('cv2.imdecode')
    @patch('numpy.frombuffer')
    def test_write_method(self, mock_frombuffer, mock_imdecode, mock_imencode):
        """Test write method with motion detection and timestamp"""
        # Mock motion detection
        server.motion_detected = False
        server.motion_regions = []
        
        # Mock motion detection and timestamp functions
        original_detect_motion = server.detect_motion
        original_add_timestamp = server.add_timestamp
        
        server.detect_motion = MagicMock(return_value=(True, [(10, 10, 30, 30)]))
        server.add_timestamp = MagicMock(return_value=np.zeros((480, 640, 3)))
        
        # Mock numpy and OpenCV functions
        mock_frombuffer.return_value = np.zeros(1)
        mock_imdecode.return_value = np.zeros((480, 640, 3))
        mock_imencode.return_value = (True, b'encoded jpeg')
        
        # Test buffer write
        buf = b'test buffer'
        self.buffer.write(buf)
        
        # Check that motion detection was called
        server.detect_motion.assert_called_once()
        
        # Check that timestamp was added
        server.add_timestamp.assert_called_once()
        
        # Restore original functions
        server.detect_motion = original_detect_motion
        server.add_timestamp = original_add_timestamp

    def test_get_frame(self):
        """Test getting frame from buffer"""
        # Set a test frame
        test_frame = b'test frame'
        self.buffer.frame = test_frame
        
        # Get frame for a client
        frame = self.buffer.get_frame("test_client")
        
        # Should return the test frame
        self.assertEqual(frame, test_frame)
        
        # Client should be registered
        self.assertIn("test_client", self.buffer.last_access_times)


if __name__ == '__main__':
    unittest.main()
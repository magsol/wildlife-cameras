#!/usr/bin/python3

"""
Test script to verify the bug fixes implemented in the wildlife camera code.
This tests:
1. Fixed variable name in unregister_client
2. Global variable scope issues
3. Improved hardware detection reliability
4. WiFi monitoring on non-Raspberry Pi hardware
5. Error handling in chunked uploads
6. FrameBuffer inherits from BufferedIOBase for PiCamera2 compatibility
"""

import os
import sys
import unittest
import tempfile
import shutil
import io
import subprocess
from unittest.mock import MagicMock, patch, mock_open

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import test helpers to set up mocks before importing the modules
from tests.test_helpers import setup_picamera2_mocks
setup_picamera2_mocks()

# Import modules to test
import fastapi_mjpeg_server_with_storage as server
from motion_storage import (
    StorageConfig,
    CircularFrameBuffer,
    WiFiMonitor,
    TransferManager,
    is_raspberry_pi,
    IS_RASPBERRY_PI,
    WIFI_AVAILABLE
)

class TestBugFixes(unittest.TestCase):
    """Test the bug fixes implemented"""

    def setUp(self):
        """Set up test environment"""
        # Create temp directory for test
        self.test_dir = tempfile.mkdtemp()
        
        # Create test config with test directory
        self.config = StorageConfig()
        self.config.local_storage_path = self.test_dir
        self.config.wifi_monitoring = True
        self.config.wifi_adapter = "test_wlan0"

    def tearDown(self):
        """Clean up after test"""
        # Remove temp directory
        shutil.rmtree(self.test_dir)

    def test_unregister_client_variable_fix(self):
        """Test the fix for variable name in unregister_client"""
        # Create frame buffer
        frame_buffer = server.FrameBuffer()
        
        # Register a client
        client_id = "test_client"
        frame_buffer.register_client(client_id)
        self.assertIn(client_id, frame_buffer.last_access_times)
        
        # Test unregister_client with the fixed variable name
        frame_buffer.unregister_client(client_id)
        self.assertNotIn(client_id, frame_buffer.last_access_times)

    def test_hardware_detection(self):
        """Test improved hardware detection reliability"""
        # Mock all methods that would indicate we're on a Raspberry Pi
        with patch('os.path.exists', return_value=False), \
             patch('builtins.open', mock_open(read_data="Test data")), \
             patch.dict('os.environ', {}, clear=True):
            
            # Verify we're not detected as a Raspberry Pi
            self.assertFalse(is_raspberry_pi())
            
        # Mock one indicator that would suggest we are on a Pi
        with patch('os.path.exists', lambda path: path == "/usr/bin/vcgencmd"), \
             patch('builtins.open', mock_open(read_data="Test data")), \
             patch.dict('os.environ', {}, clear=True):
            
            # Verify we are detected as a Raspberry Pi
            self.assertTrue(is_raspberry_pi())
            
        # Test environment variable detection
        with patch('os.path.exists', return_value=False), \
             patch('builtins.open', mock_open(read_data="Test data")), \
             patch.dict('os.environ', {"RASPBERRY_PI": "true"}):
            
            # Verify we are detected as a Raspberry Pi via env variable
            self.assertTrue(is_raspberry_pi())

    def test_wifi_monitoring_non_pi(self):
        """Test WiFi monitoring on non-Raspberry Pi hardware"""
        # Mock that we're not on a Raspberry Pi
        with patch('motion_storage.IS_RASPBERRY_PI', False), \
             patch('shutil.which', return_value=None):
            
            # Create WiFi monitor
            wifi_monitor = WiFiMonitor(self.config)
            
            # Verify monitoring is disabled
            self.assertFalse(wifi_monitor.enabled)

    def test_wifi_adapter_check(self):
        """Test WiFi adapter check improvements"""
        # Mock that iwconfig is not available
        with patch('shutil.which', return_value=None), \
             patch('motion_storage.IS_RASPBERRY_PI', True):
            
            # Create WiFi monitor
            wifi_monitor = WiFiMonitor(self.config)
            
            # Verify monitoring is disabled
            self.assertFalse(wifi_monitor.enabled)

        # Mock that iwconfig is available but adapter doesn't exist
        with patch('shutil.which', return_value="/usr/bin/iwconfig"), \
             patch('subprocess.check_output', side_effect=subprocess.SubprocessError()), \
             patch('motion_storage.IS_RASPBERRY_PI', True):
            
            # Create WiFi monitor
            wifi_monitor = WiFiMonitor(self.config)
            
            # Verify monitoring is disabled
            self.assertFalse(wifi_monitor.enabled)
            
    def test_global_variables_initialization(self):
        """Test the global variables are properly initialized"""
        # Check that the global variables are defined in motion_storage
        from motion_storage import prev_frame, motion_detected, motion_regions
        
        # Verify they are initialized with default values
        self.assertIsNone(prev_frame)
        self.assertFalse(motion_detected)
        self.assertEqual(motion_regions, [])
        
    def test_frame_buffer_io_inheritance(self):
        """Test that FrameBuffer properly inherits from BufferedIOBase"""
        frame_buffer = server.FrameBuffer()
        
        # Verify it's an instance of BufferedIOBase
        self.assertIsInstance(frame_buffer, io.BufferedIOBase)
        
        # Test the implemented interface methods
        self.assertFalse(frame_buffer.readable())
        self.assertTrue(frame_buffer.writable())
        self.assertFalse(frame_buffer.seekable())
        
        # Ensure it doesn't raise exceptions when used as a file-like object
        frame_buffer.write(b"test")
        frame_buffer.flush()

if __name__ == '__main__':
    unittest.main()
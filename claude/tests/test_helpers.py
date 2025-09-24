#!/usr/bin/python3

"""
Test helpers for the Wildlife Camera project.
This module provides mock implementations and utilities for testing,
particularly for handling platform-specific dependencies like picamera2.
"""

import os
import sys
from unittest.mock import MagicMock

# Detect if this is a Raspberry Pi
def is_raspberry_pi():
    """Detect if running on a Raspberry Pi"""
    # Method 1: Check for Pi-specific files
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
        
    # Method 3: Check for environment variable
    if "RASPBERRY_PI" in os.environ:
        return True
        
    return False

# Check if picamera2 is available
def is_picamera2_available():
    """Check if picamera2 module is available"""
    # If SKIP_PICAMERA environment variable is set, always return False
    if os.environ.get("SKIP_PICAMERA", "").lower() in ("1", "true", "yes"):
        return False
        
    try:
        import picamera2
        return True
    except ImportError:
        return False

# Mock picamera2 modules if not on Raspberry Pi or picamera2 is not available
def setup_picamera2_mocks():
    """Set up mocks for picamera2 modules if not available"""
    if not is_raspberry_pi() or not is_picamera2_available():
        print("picamera2 not available, using mock implementation")
        
        # Create mock modules
        sys.modules['picamera2'] = MagicMock()
        sys.modules['picamera2.encoders'] = MagicMock()
        sys.modules['picamera2.outputs'] = MagicMock()
        
        # Create mock classes
        sys.modules['picamera2'].Picamera2 = MockPicamera2
        sys.modules['picamera2.encoders'].MJPEGEncoder = MockMJPEGEncoder
        sys.modules['picamera2.outputs'].FileOutput = MockFileOutput

# Mock implementation of Picamera2
class MockPicamera2:
    """Mock implementation of Picamera2 class"""
    
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

# Mock implementation of MJPEGEncoder
class MockMJPEGEncoder:
    """Mock implementation of MJPEGEncoder class"""
    
    def __init__(self, **kwargs):
        self.quality = kwargs.get('quality', 80)
        
# Mock implementation of FileOutput
class MockFileOutput:
    """Mock implementation of FileOutput class"""
    
    def __init__(self, file_like_obj):
        self.file = file_like_obj
        
    def start(self):
        pass
        
    def stop(self):
        pass

# Run the mock setup when this module is imported
setup_picamera2_mocks()
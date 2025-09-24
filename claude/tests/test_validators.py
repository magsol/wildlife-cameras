#!/usr/bin/python3

"""
Simple test script to verify the updated Pydantic V2 style field validators.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Set environment variable to skip picamera2
os.environ["SKIP_PICAMERA"] = "1"

# Import test helpers first to set up mocks
from tests.test_helpers import setup_picamera2_mocks
setup_picamera2_mocks()

# Now import the class to test
from fastapi_mjpeg_server_with_storage import CameraConfigUpdate

def test_rotation_validator():
    print("Testing rotation validation...")
    
    # Valid values
    for rotation in [0, 90, 180, 270]:
        try:
            config = CameraConfigUpdate(rotation=rotation)
            print(f"  ✓ Rotation {rotation} - Valid as expected")
        except Exception as e:
            print(f"  ✗ Rotation {rotation} - ERROR: {e}")
    
    # Invalid values
    for rotation in [45, 100, 360]:
        try:
            config = CameraConfigUpdate(rotation=rotation)
            print(f"  ✗ Rotation {rotation} - Unexpectedly accepted invalid value")
        except ValueError as e:
            print(f"  ✓ Rotation {rotation} - Validation failed as expected: {e}")
        except Exception as e:
            print(f"  ✗ Rotation {rotation} - ERROR: {e}")

def test_timestamp_position_validator():
    print("\nTesting timestamp position validation...")
    
    # Valid values
    for position in ["top-left", "top-right", "bottom-left", "bottom-right"]:
        try:
            config = CameraConfigUpdate(timestamp_position=position)
            print(f"  ✓ Position {position} - Valid as expected")
        except Exception as e:
            print(f"  ✗ Position {position} - ERROR: {e}")
    
    # Invalid values
    for position in ["center", "invalid"]:
        try:
            config = CameraConfigUpdate(timestamp_position=position)
            print(f"  ✗ Position {position} - Unexpectedly accepted invalid value")
        except ValueError as e:
            print(f"  ✓ Position {position} - Validation failed as expected: {e}")
        except Exception as e:
            print(f"  ✗ Position {position} - ERROR: {e}")

if __name__ == "__main__":
    test_rotation_validator()
    test_timestamp_position_validator()
    print("\nAll tests completed!")
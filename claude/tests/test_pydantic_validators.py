#!/usr/bin/python3

"""
Test script to verify the updated Pydantic V2 style field validators.
"""

import sys
from fastapi.testclient import TestClient
from fastapi_mjpeg_server_with_storage import app

client = TestClient(app)

# Test rotation validation
print("Testing rotation validation...")
test_cases = [
    # Valid cases
    {"rotation": 0, "expected_status": 200},
    {"rotation": 90, "expected_status": 200},
    {"rotation": 180, "expected_status": 200},
    {"rotation": 270, "expected_status": 200},
    # Invalid cases
    {"rotation": 45, "expected_status": 422},
    {"rotation": 360, "expected_status": 422},
]

for i, test in enumerate(test_cases):
    response = client.post("/config", json={"rotation": test["rotation"]})
    success = response.status_code == test["expected_status"]
    print(f"  Test {i+1}: Rotation {test['rotation']} - {'✓' if success else '✗'} (expected {test['expected_status']}, got {response.status_code})")
    if not success and response.status_code == 422:
        print(f"    Error details: {response.json()}")

# Test timestamp position validation
print("\nTesting timestamp position validation...")
test_cases = [
    # Valid cases
    {"timestamp_position": "top-left", "expected_status": 200},
    {"timestamp_position": "top-right", "expected_status": 200},
    {"timestamp_position": "bottom-left", "expected_status": 200},
    {"timestamp_position": "bottom-right", "expected_status": 200},
    # Invalid cases
    {"timestamp_position": "center", "expected_status": 422},
    {"timestamp_position": "invalid", "expected_status": 422},
]

for i, test in enumerate(test_cases):
    response = client.post("/config", json={"timestamp_position": test["timestamp_position"]})
    success = response.status_code == test["expected_status"]
    print(f"  Test {i+1}: Position {test['timestamp_position']} - {'✓' if success else '✗'} (expected {test['expected_status']}, got {response.status_code})")
    if not success and response.status_code == 422:
        print(f"    Error details: {response.json()}")

print("\nAll tests completed!")
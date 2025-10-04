#!/usr/bin/env python3
"""
Configuration and fixtures for pytest.
"""

import os
import sys
import tempfile
import pytest
import numpy as np
import cv2
from datetime import datetime

# Add parent directory to path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def test_frames():
    """
    Create synthetic test frames with known motion patterns.
    
    Returns:
        Dictionary containing:
        - frame1: First frame with a white rectangle
        - frame2: Second frame with the rectangle moved
        - frame1_gray: Grayscale version of frame1
        - frame2_gray: Grayscale version of frame2
        - motion_region: Region encompassing the motion
    """
    # Create a black background frame
    frame_size = (320, 240)
    frame1 = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
    frame2 = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
    
    # Add a rectangle in frame1
    x1, y1, w1, h1 = 50, 50, 40, 30
    frame1[y1:y1+h1, x1:x1+w1] = (255, 255, 255)
    
    # Add a rectangle in frame2, moved right and down (simulating motion)
    x2, y2 = x1 + 10, y1 + 5  # Move 10px right, 5px down
    frame2[y2:y2+h1, x2:x2+w1] = (255, 255, 255)
    
    # Define motion region
    motion_region = [(min(x1, x2), min(y1, y2), 
                      max(x1+w1, x2+w1) - min(x1, x2), 
                      max(y1+h1, y2+h1) - min(y1, y2))]
    
    # Convert to grayscale
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    return {
        'frame1': frame1,
        'frame2': frame2,
        'frame1_gray': frame1_gray,
        'frame2_gray': frame2_gray,
        'motion_region': motion_region,
        'frame_size': frame_size
    }


@pytest.fixture
def sample_flow_vectors():
    """
    Create sample flow vectors with different directions.
    
    Returns:
        List of flow vectors (old_x, old_y, new_x, new_y, magnitude, angle)
    """
    return [
        (50, 50, 60, 50, 10.0, 0.0),       # Horizontal right
        (100, 50, 100, 60, 10.0, np.pi/2), # Vertical down
        (150, 50, 140, 50, 10.0, np.pi),   # Horizontal left
        (200, 50, 200, 40, 10.0, -np.pi/2) # Vertical up
    ]


@pytest.fixture
def sample_flow_features():
    """
    Create sample flow features dictionary.
    
    Returns:
        Dictionary with flow features
    """
    return {
        'frame_idx': 1,
        'timestamp': datetime.now().timestamp(),
        'flow_vectors': [
            (50, 50, 60, 55, 11.18, 0.464),
            (100, 100, 110, 105, 11.18, 0.464)
        ],
        'frame_shape': (240, 320),
        'histogram': np.random.random((8, 8, 8)),
        'motion_regions': [(40, 40, 30, 30)],
        'feature_tracks': {
            0: [(50, 50), (60, 55)],
            1: [(100, 100), (110, 105)]
        },
        'stats': {
            'mean_magnitude': 11.18,
            'std_magnitude': 0.0,
            'mean_angle': 0.464,
            'angular_dispersion': 0.0,
            'max_magnitude': 11.18,
            'flow_complexity': 0.125,
            'dominant_direction': 0.5
        }
    }


@pytest.fixture
def sample_flow_history():
    """
    Create sample flow history for testing signature generation.
    
    Returns:
        List of flow features dictionaries
    """
    history = []
    for i in range(3):
        history.append({
            'frame_idx': i,
            'timestamp': datetime.now().timestamp() + i,
            'histogram': np.random.random((8, 8, 8)),
            'motion_regions': [(40, 40, 30, 30)],
            'stats': {
                'mean_magnitude': 10.0 + i * 2.0,
                'std_magnitude': 1.0 + i * 0.5,
                'mean_angle': 0.1 * i,
                'angular_dispersion': 0.1 + i * 0.05,
                'max_magnitude': 15.0 + i * 3.0,
                'flow_complexity': 0.2 + i * 0.1,
                'dominant_direction': 0.1 * i
            }
        })
    return history


@pytest.fixture
def temp_db_dir():
    """
    Create temporary directory for database testing.
    
    Returns:
        Path to temporary directory
    """
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
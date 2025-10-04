#!/usr/bin/env python3
"""
Unit tests for optical_flow_analyzer.py module.
"""

import os
import sys
import unittest
import pytest
import numpy as np
import cv2
import tempfile
import shutil
import json
from datetime import datetime
from unittest.mock import MagicMock, patch

# Add parent directory to path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from optical_flow_analyzer import OpticalFlowAnalyzer, MotionPatternDatabase


class TestOpticalFlowAnalyzer(unittest.TestCase):
    """Test cases for OpticalFlowAnalyzer class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self.analyzer = OpticalFlowAnalyzer()
        
        # Create test frames
        self.frame_size = (320, 240)
        self.create_test_frames()
    
    def create_test_frames(self):
        """Create synthetic test frames with known motion patterns."""
        # Create a black background frame
        self.frame1 = np.zeros((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8)
        self.frame2 = np.zeros((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8)
        
        # Add a rectangle in frame1
        x1, y1, w1, h1 = 50, 50, 40, 30
        self.frame1[y1:y1+h1, x1:x1+w1] = (255, 255, 255)
        
        # Add a rectangle in frame2, moved right and down (simulating motion)
        x2, y2 = x1 + 10, y1 + 5  # Move 10px right, 5px down
        self.frame2[y2:y2+h1, x2:x2+w1] = (255, 255, 255)
        
        # Define motion region
        self.motion_region = [(min(x1, x2), min(y1, y2), 
                              max(x1+w1, x2+w1) - min(x1, x2), 
                              max(y1+h1, y2+h1) - min(y1, y2))]
        
        # Convert to grayscale
        self.frame1_gray = cv2.cvtColor(self.frame1, cv2.COLOR_BGR2GRAY)
        self.frame2_gray = cv2.cvtColor(self.frame2, cv2.COLOR_BGR2GRAY)
    
    def test_initialization(self):
        """Test initialization of OpticalFlowAnalyzer."""
        # Check default config
        self.assertIsNotNone(self.analyzer.config)
        self.assertIn('feature_params', self.analyzer.config)
        self.assertIn('lk_params', self.analyzer.config)
        
        # Check custom config
        custom_config = {
            'feature_params': {
                'maxCorners': 50,
                'qualityLevel': 0.2
            }
        }
        analyzer = OpticalFlowAnalyzer(config=custom_config)
        self.assertEqual(analyzer.config['feature_params']['maxCorners'], 50)
        self.assertEqual(analyzer.config['feature_params']['qualityLevel'], 0.2)
    
    def test_reset(self):
        """Test reset function."""
        # Set some state variables
        self.analyzer.prev_frame = self.frame1
        self.analyzer.prev_gray = self.frame1_gray
        self.analyzer.prev_features = np.array([[[10, 10]]], dtype=np.float32)
        self.analyzer.feature_tracks = {0: [(10, 10), (11, 11)]}
        self.analyzer.frame_count = 10
        
        # Reset analyzer
        self.analyzer.reset()
        
        # Check that state was reset
        self.assertIsNone(self.analyzer.prev_frame)
        self.assertIsNone(self.analyzer.prev_gray)
        self.assertIsNone(self.analyzer.prev_features)
        self.assertEqual(self.analyzer.feature_tracks, {})
        self.assertEqual(self.analyzer.frame_count, 0)
    
    def test_detect_features(self):
        """Test feature detection."""
        # Detect features in frame1
        features = self.analyzer._detect_features(self.frame1_gray)
        
        # Check that we got features
        self.assertIsNotNone(features)
        self.assertIsInstance(features, np.ndarray)
        
        # Try with a mask
        mask = np.zeros_like(self.frame1_gray)
        mask[50:100, 50:100] = 255  # Only detect in this region
        masked_features = self.analyzer._detect_features(self.frame1_gray, mask)
        
        # Check that masked features are found only in the masked region
        if masked_features is not None:
            for feature in masked_features:
                x, y = feature.ravel()
                self.assertTrue(50 <= x < 100)
                self.assertTrue(50 <= y < 100)
    
    def test_extract_flow_initialization(self):
        """Test the initialization case of extract_flow."""
        # First call should initialize and return None
        result = self.analyzer.extract_flow(self.frame1, self.frame2)
        self.assertIsNone(result)
        
        # Check that state was initialized
        self.assertIsNotNone(self.analyzer.prev_gray)
        self.assertIsNotNone(self.analyzer.prev_features)
    
    @patch('cv2.calcOpticalFlowPyrLK')
    def test_extract_flow_calculation(self, mock_calcOpticalFlowPyrLK):
        """Test optical flow calculation."""
        # Mock the optical flow calculation to return known values
        # Create synthetic feature points and status
        old_points = np.array([[[50, 50]], [[60, 60]], [[70, 70]]], dtype=np.float32)
        new_points = np.array([[[60, 55]], [[70, 65]], [[80, 75]]], dtype=np.float32)
        status = np.array([1, 1, 1], dtype=np.int32)
        err = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        
        mock_calcOpticalFlowPyrLK.return_value = (new_points, status, err)
        
        # Initialize analyzer state for test
        self.analyzer.prev_gray = self.frame1_gray
        self.analyzer.prev_features = old_points
        self.analyzer.frame_count = 1
        self.analyzer.feature_tracks = {
            0: [(50, 50)],
            1: [(60, 60)],
            2: [(70, 70)]
        }
        
        # Call extract_flow with motion regions
        flow_features = self.analyzer.extract_flow(
            self.frame1, self.frame2, self.motion_region)
        
        # Verify calcOpticalFlowPyrLK was called
        mock_calcOpticalFlowPyrLK.assert_called_once()
        
        # Check that flow_features contains expected data
        self.assertIsNotNone(flow_features)
        self.assertEqual(flow_features['frame_idx'], 1)
        self.assertEqual(len(flow_features['flow_vectors']), 3)
        
        # Check that flow vectors have expected structure
        for i, vector in enumerate(flow_features['flow_vectors']):
            old_x, old_y = old_points[i][0]
            new_x, new_y = new_points[i][0]
            self.assertEqual(vector[0], old_x)
            self.assertEqual(vector[1], old_y)
            self.assertEqual(vector[2], new_x)
            self.assertEqual(vector[3], new_y)
            
            # Check magnitude and angle
            dx, dy = new_x - old_x, new_y - old_y
            expected_mag = np.sqrt(dx*dx + dy*dy)
            expected_angle = np.arctan2(dy, dx)
            self.assertAlmostEqual(vector[4], expected_mag)
            self.assertAlmostEqual(vector[5], expected_angle)
        
        # Check that feature tracks were updated
        self.assertEqual(len(self.analyzer.feature_tracks), 3)
        for i, track in enumerate(self.analyzer.feature_tracks.values()):
            self.assertEqual(len(track), 2)  # Original + new point
            self.assertEqual(track[1], (int(new_points[i][0][0]), int(new_points[i][0][1])))
    
    def test_compute_flow_histogram(self):
        """Test flow histogram computation."""
        # Create synthetic flow vectors
        flow_vectors = [
            (50, 50, 60, 50, 10.0, 0.0),      # Horizontal right
            (100, 50, 100, 60, 10.0, np.pi/2), # Vertical down
            (150, 50, 140, 50, 10.0, np.pi),   # Horizontal left
            (200, 50, 200, 40, 10.0, -np.pi/2) # Vertical up
        ]
        
        # Compute histogram
        histogram = self.analyzer._compute_flow_histogram(
            flow_vectors, self.frame_size)
        
        # Check histogram shape
        grid_h, grid_w = self.analyzer.config['grid_size']
        direction_bins = self.analyzer.config['direction_bins']
        self.assertEqual(histogram.shape, (grid_h, grid_w, direction_bins))
        
        # Check that histogram sums to 1 (normalized)
        self.assertAlmostEqual(histogram.sum(), 1.0)
        
        # Check that histogram has non-zero values in expected locations
        # This is approximate since we're binning flow vectors
        self.assertTrue(np.any(histogram > 0))
    
    def test_compute_flow_statistics(self):
        """Test flow statistics computation."""
        # Create synthetic flow vectors with known statistics
        flow_vectors = [
            (50, 50, 60, 50, 10.0, 0.0),      # Horizontal right
            (100, 50, 100, 60, 10.0, np.pi/2), # Vertical down
            (150, 50, 140, 50, 10.0, np.pi),   # Horizontal left
            (200, 50, 200, 40, 10.0, -np.pi/2) # Vertical up
        ]
        
        # Compute statistics
        stats = self.analyzer._compute_flow_statistics(flow_vectors)
        
        # Check that stats contains expected keys
        expected_keys = [
            'mean_magnitude', 'std_magnitude', 'mean_angle',
            'angular_dispersion', 'max_magnitude',
            'flow_complexity', 'dominant_direction'
        ]
        for key in expected_keys:
            self.assertIn(key, stats)
        
        # Check specific values
        self.assertEqual(stats['mean_magnitude'], 10.0)
        self.assertEqual(stats['std_magnitude'], 0.0)
        self.assertEqual(stats['max_magnitude'], 10.0)
        
        # Angular dispersion should be high since angles are in all directions
        self.assertGreater(stats['angular_dispersion'], 0.5)
        
        # Flow complexity should be high since we have 4 different directions
        self.assertGreater(stats['flow_complexity'], 0.0)
    
    def test_visualize_flow(self):
        """Test flow visualization."""
        # Create synthetic flow features
        flow_features = {
            'frame_idx': 1,
            'flow_vectors': [
                (50, 50, 60, 55, 10.0, 0.1),
                (100, 100, 110, 105, 10.0, 0.1)
            ],
            'motion_regions': [(40, 40, 30, 30)],
            'feature_tracks': {
                0: [(50, 50), (60, 55)],
                1: [(100, 100), (110, 105)]
            }
        }
        
        # Visualize flow
        vis_frame = self.analyzer.visualize_flow(self.frame1, flow_features)
        
        # Check that result is an image
        self.assertIsInstance(vis_frame, np.ndarray)
        self.assertEqual(vis_frame.shape, self.frame1.shape)
        
        # Visualize without flow_features should return original frame
        vis_frame2 = self.analyzer.visualize_flow(self.frame1, None)
        np.testing.assert_array_equal(vis_frame2, self.frame1)
    
    def test_generate_motion_signature(self):
        """Test motion signature generation."""
        # Create synthetic flow history
        flow_history = [
            {
                'histogram': np.random.random((8, 8, 8)),
                'stats': {
                    'mean_magnitude': 10.0,
                    'std_magnitude': 2.0,
                    'mean_angle': 0.5,
                    'angular_dispersion': 0.3,
                    'max_magnitude': 15.0,
                    'flow_complexity': 0.4,
                    'dominant_direction': 0.0
                }
            },
            {
                'histogram': np.random.random((8, 8, 8)),
                'stats': {
                    'mean_magnitude': 12.0,
                    'std_magnitude': 2.5,
                    'mean_angle': 0.6,
                    'angular_dispersion': 0.35,
                    'max_magnitude': 18.0,
                    'flow_complexity': 0.45,
                    'dominant_direction': 0.1
                }
            }
        ]
        
        # Set flow history in analyzer
        self.analyzer.flow_history = flow_history
        
        # Generate signature
        signature = self.analyzer.generate_motion_signature()
        
        # Check that signature contains expected keys
        expected_keys = [
            'histogram_features', 'statistical_features',
            'temporal_features', 'timestamp', 'frame_count'
        ]
        for key in expected_keys:
            self.assertIn(key, signature)
        
        # Check specific values
        self.assertEqual(signature['frame_count'], 2)
        self.assertEqual(signature['statistical_features']['mean_magnitude'], 11.0)
        
        # Check with explicit flow_history
        signature2 = self.analyzer.generate_motion_signature(flow_history)
        self.assertEqual(signature2['frame_count'], 2)
    
    def test_extract_temporal_features(self):
        """Test temporal feature extraction."""
        # Create synthetic flow history
        flow_history = [
            {
                'stats': {
                    'mean_magnitude': 10.0,
                    'mean_angle': 0.0
                }
            },
            {
                'stats': {
                    'mean_magnitude': 15.0,
                    'mean_angle': 0.1
                }
            },
            {
                'stats': {
                    'mean_magnitude': 20.0,
                    'mean_angle': 0.2
                }
            }
        ]
        
        # Extract temporal features
        temporal_features = self.analyzer._extract_temporal_features(flow_history)
        
        # Check that temporal_features contains expected keys
        expected_keys = [
            'magnitude_trend', 'magnitude_variability',
            'acceleration_mean', 'acceleration_std',
            'direction_changes', 'direction_stability'
        ]
        for key in expected_keys:
            self.assertIn(key, temporal_features)
        
        # Check specific values
        self.assertAlmostEqual(temporal_features['magnitude_trend'], 5.0)
        self.assertGreater(temporal_features['direction_stability'], 0.0)
    
    def test_classify_motion(self):
        """Test motion classification."""
        # Create a synthetic motion signature
        motion_signature = {
            'histogram_features': np.random.random(512),
            'statistical_features': {
                'mean_magnitude': 15.0,
                'std_magnitude': 2.0,
                'mean_angle': 0.5,
                'angular_dispersion': 0.3,
                'max_magnitude': 20.0,
                'flow_complexity': 0.2,
                'dominant_direction': 0.0
            },
            'temporal_features': {
                'magnitude_trend': 5.0,
                'magnitude_variability': 3.0,
                'acceleration_mean': 2.0,
                'acceleration_std': 1.0,
                'direction_changes': 0.1,
                'direction_stability': 0.9
            }
        }
        
        # Classify motion
        classification = self.analyzer.classify_motion(motion_signature)
        
        # Check that classification contains expected keys
        expected_keys = ['label', 'confidence', 'alternatives']
        for key in expected_keys:
            self.assertIn(key, classification)
        
        # Check that label is one of the expected classes
        self.assertIn(classification['label'], ['vehicle', 'person', 'animal', 'environment', 'unknown'])
        
        # Check confidence is between 0 and 1
        self.assertTrue(0.0 <= classification['confidence'] <= 1.0)


class TestMotionPatternDatabase(unittest.TestCase):
    """Test cases for MotionPatternDatabase class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create temporary directory for database and signatures
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, 'test_db.db')
        self.signature_dir = os.path.join(self.test_dir, 'signatures')
        
        # Create database
        self.db = MotionPatternDatabase(
            db_path=self.db_path,
            signature_dir=self.signature_dir
        )
        
        # Create sample motion signature
        self.pattern_id = 'test_pattern_001'
        self.motion_signature = {
            'histogram_features': np.random.random(512),
            'statistical_features': {
                'mean_magnitude': 10.0,
                'std_magnitude': 2.0,
                'mean_angle': 0.5,
                'angular_dispersion': 0.3,
                'max_magnitude': 15.0,
                'flow_complexity': 0.4,
                'dominant_direction': 0.0
            },
            'temporal_features': {
                'magnitude_trend': 2.0,
                'magnitude_variability': 1.0,
                'acceleration_mean': 0.5,
                'acceleration_std': 0.2,
                'direction_changes': 0.1,
                'direction_stability': 0.9
            },
            'timestamp': datetime.now().isoformat(),
            'frame_count': 30
        }
        
        self.classification = {
            'label': 'vehicle',
            'confidence': 0.8,
            'alternatives': [
                {'label': 'person', 'confidence': 0.1},
                {'label': 'other', 'confidence': 0.1}
            ]
        }
        
        self.metadata = {
            'event_id': 'event_001',
            'duration': 5.2,
            'time_of_day': 14,
            'weather': 'sunny'
        }
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test database initialization."""
        # Check that database and signature directory were created
        self.assertTrue(os.path.exists(self.db_path))
        self.assertTrue(os.path.exists(self.signature_dir))
        
        # Check that database has required tables
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Check motion_patterns table
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='motion_patterns'")
        self.assertIsNotNone(c.fetchone())
        
        # Check labels table
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='labels'")
        self.assertIsNotNone(c.fetchone())
        
        conn.close()
    
    def test_add_pattern(self):
        """Test adding a pattern to the database."""
        # Add pattern
        result = self.db.add_pattern(
            self.pattern_id,
            self.motion_signature,
            self.classification,
            self.metadata
        )
        
        # Check that add was successful
        self.assertTrue(result)
        
        # Check that signature file was created
        signature_path = os.path.join(self.signature_dir, f"{self.pattern_id}.npz")
        self.assertTrue(os.path.exists(signature_path))
        
        # Check that database record was created
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("SELECT id, classification, confidence FROM motion_patterns WHERE id=?",
                 (self.pattern_id,))
        row = c.fetchone()
        conn.close()
        
        self.assertIsNotNone(row)
        self.assertEqual(row[0], self.pattern_id)
        self.assertEqual(row[1], self.classification['label'])
        self.assertEqual(row[2], self.classification['confidence'])
    
    def test_get_pattern(self):
        """Test retrieving a pattern from the database."""
        # Add pattern first
        self.db.add_pattern(
            self.pattern_id,
            self.motion_signature,
            self.classification,
            self.metadata
        )
        
        # Get pattern
        pattern = self.db.get_pattern(self.pattern_id)
        
        # Check that pattern was retrieved
        self.assertIsNotNone(pattern)
        self.assertEqual(pattern['id'], self.pattern_id)
        self.assertEqual(pattern['classification']['label'], self.classification['label'])
        self.assertEqual(pattern['classification']['confidence'], self.classification['confidence'])
        self.assertEqual(pattern['metadata'], self.metadata)
        
        # Check that signature was loaded
        self.assertIn('signature', pattern)
        self.assertIn('histogram_features', pattern['signature'])
        self.assertIn('statistical_features', pattern['signature'])
        self.assertIn('temporal_features', pattern['signature'])
    
    def test_update_classification(self):
        """Test updating classification for a pattern."""
        # Add pattern first
        self.db.add_pattern(
            self.pattern_id,
            self.motion_signature,
            self.classification,
            self.metadata
        )
        
        # New classification
        new_classification = {
            'label': 'person',
            'confidence': 0.9
        }
        
        # Update classification
        result = self.db.update_classification(
            self.pattern_id,
            new_classification
        )
        
        # Check that update was successful
        self.assertTrue(result)
        
        # Check that database record was updated
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("SELECT classification, confidence FROM motion_patterns WHERE id=?",
                 (self.pattern_id,))
        row = c.fetchone()
        conn.close()
        
        self.assertIsNotNone(row)
        self.assertEqual(row[0], new_classification['label'])
        self.assertEqual(row[1], new_classification['confidence'])
        
        # Get pattern and check classification
        pattern = self.db.get_pattern(self.pattern_id)
        self.assertEqual(pattern['classification']['label'], new_classification['label'])
        self.assertEqual(pattern['classification']['confidence'], new_classification['confidence'])


if __name__ == "__main__":
    unittest.main()
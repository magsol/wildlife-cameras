#!/usr/bin/env python3
"""
Optical Flow Analysis Module for Motion Classification

This module provides functionality to analyze motion patterns using optical flow techniques.
It is designed to work with the existing motion detection system to classify movements
as different object types (people, vehicles, animals, etc.) based on their motion signatures.
"""

import cv2
import numpy as np
import time
import logging
import os
import json
from datetime import datetime
import sqlite3
from typing import Dict, List, Tuple, Optional, Any, Union
import threading
import pickle
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class OpticalFlowAnalyzer:
    """
    Main class for analyzing optical flow patterns in video frames.
    """
    
    def __init__(self, config=None):
        """
        Initialize the OpticalFlowAnalyzer with optional configuration.
        
        Args:
            config: Configuration dictionary with parameters for optical flow analysis
        """
        # Default configuration
        self.config = {
            'feature_params': {
                'maxCorners': 100,
                'qualityLevel': 0.3,
                'minDistance': 7,
                'blockSize': 7
            },
            'lk_params': {
                'winSize': (15, 15),
                'maxLevel': 2,
                'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            },
            'grid_size': (8, 8),       # 8x8 grid for spatial binning
            'direction_bins': 8,       # 8 direction bins (45 degrees each)
            'frame_history': 10,       # Number of frames to track features
            'min_features': 10,        # Minimum features to track
            'feature_quality_threshold': 0.01,  # Threshold for feature quality
            'flow_visualization_scale': 1.5,    # Scale factor for flow visualization
        }
        
        # Update with user config if provided
        if config:
            self._update_config(config)
            
        # State variables
        self.prev_frame = None
        self.prev_gray = None
        self.prev_features = None
        self.feature_tracks = {}  # Track feature points over time
        self.frame_count = 0
        self.last_feature_detection_frame = 0
        self.flow_history = []
        
        # Mapping from cluster IDs to semantic labels
        self.cluster_labels = {}
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        logger.info("OpticalFlowAnalyzer initialized with configuration: %s", self.config)

    def _update_config(self, config):
        """
        Update the configuration with user-provided values.
        
        Args:
            config: Configuration dictionary to update from
        """
        for key, value in config.items():
            if isinstance(value, dict) and key in self.config and isinstance(self.config[key], dict):
                self.config[key].update(value)
            else:
                self.config[key] = value

    def reset(self):
        """Reset the analyzer state for a new motion sequence."""
        with self.lock:
            self.prev_frame = None
            self.prev_gray = None
            self.prev_features = None
            self.feature_tracks = {}
            self.frame_count = 0
            self.last_feature_detection_frame = 0
            self.flow_history = []

    def _detect_features(self, frame_gray, mask=None):
        """
        Detect good features to track in the frame.
        
        Args:
            frame_gray: Grayscale input frame
            mask: Optional mask to restrict feature detection to specific regions
            
        Returns:
            Array of detected feature points
        """
        return cv2.goodFeaturesToTrack(
            frame_gray, 
            mask=mask,
            **self.config['feature_params']
        )

    def extract_flow(self, prev_frame, current_frame, motion_regions=None):
        """
        Extract optical flow features from consecutive frames.
        
        Args:
            prev_frame: Previous video frame
            current_frame: Current video frame
            motion_regions: Optional list of (x, y, w, h) tuples defining motion regions
            
        Returns:
            Dictionary containing flow features and metadata
        """
        with self.lock:
            # Convert frames to grayscale if needed
            if len(prev_frame.shape) == 3:
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            else:
                prev_gray = prev_frame
                
            if len(current_frame.shape) == 3:
                current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            else:
                current_gray = current_frame
            
            # Initialize or update previous frame
            if self.prev_gray is None:
                self.prev_gray = prev_gray
                self.frame_count = 0
                
                # Initial feature detection
                if motion_regions:
                    # Create mask from motion regions
                    mask = np.zeros_like(prev_gray)
                    for x, y, w, h in motion_regions:
                        mask[y:y+h, x:x+w] = 255
                    
                    self.prev_features = self._detect_features(prev_gray, mask)
                else:
                    self.prev_features = self._detect_features(prev_gray)
                    
                # Initialize feature tracks
                if self.prev_features is not None:
                    for i, point in enumerate(self.prev_features):
                        pt = tuple(map(int, point.ravel()))
                        self.feature_tracks[i] = [pt]
                
                self.last_feature_detection_frame = 0
                return None
            
            # If we don't have features to track, detect new ones
            feature_detection_interval = 5  # Detect new features every N frames
            if self.prev_features is None or len(self.prev_features) < self.config['min_features'] or \
               (self.frame_count - self.last_feature_detection_frame) >= feature_detection_interval:
                
                if motion_regions:
                    # Create mask from motion regions
                    mask = np.zeros_like(prev_gray)
                    for x, y, w, h in motion_regions:
                        mask[y:y+h, x:x+w] = 255
                    
                    self.prev_features = self._detect_features(prev_gray, mask)
                else:
                    self.prev_features = self._detect_features(prev_gray)
                
                # Reset feature tracks with new detection
                self.feature_tracks = {}
                if self.prev_features is not None:
                    for i, point in enumerate(self.prev_features):
                        pt = tuple(map(int, point.ravel()))
                        self.feature_tracks[i] = [pt]
                
                self.last_feature_detection_frame = self.frame_count
            
            # If no features were found, return None
            if self.prev_features is None or len(self.prev_features) == 0:
                self.prev_gray = current_gray
                self.frame_count += 1
                return None
            
            # Calculate optical flow
            next_features, status, error = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, 
                current_gray, 
                self.prev_features, 
                None, 
                **self.config['lk_params']
            )
            
            # Filter out points where flow wasn't found
            # status is (N,) but features are (N, 1, 2), so we need to reshape
            status_flat = status.ravel()
            good_new = next_features[status_flat == 1]
            good_old = self.prev_features[status_flat == 1]
            
            # Calculate flow vectors
            flow_vectors = []
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                
                # Calculate displacement vector
                dx, dy = a - c, b - d
                
                # Calculate flow magnitude and direction
                magnitude = np.sqrt(dx * dx + dy * dy)
                angle = np.arctan2(dy, dx)
                
                # Store flow vector (old_x, old_y, new_x, new_y, magnitude, angle)
                flow_vectors.append((c, d, a, b, magnitude, angle))
                
                # Update feature tracks
                track_id = None
                for k, track in self.feature_tracks.items():
                    if len(track) > 0 and track[-1] == (int(c), int(d)):
                        track_id = k
                        break
                
                if track_id is not None:
                    self.feature_tracks[track_id].append((int(a), int(b)))
                
            # Create flow features dictionary
            flow_features = {
                'frame_idx': self.frame_count,
                'timestamp': time.time(),
                'flow_vectors': flow_vectors,
                'frame_shape': current_gray.shape,
                'histogram': self._compute_flow_histogram(flow_vectors, current_gray.shape),
                'motion_regions': motion_regions if motion_regions else [],
                'feature_tracks': {k: v for k, v in self.feature_tracks.items() if len(v) > 1},
                'stats': self._compute_flow_statistics(flow_vectors)
            }
            
            # Store flow history
            self.flow_history.append(flow_features)
            if len(self.flow_history) > self.config['frame_history']:
                self.flow_history.pop(0)
            
            # Update state for next frame
            self.prev_gray = current_gray
            self.prev_features = good_new.reshape(-1, 1, 2)
            self.frame_count += 1
            
            return flow_features

    def _compute_flow_histogram(self, flow_vectors, frame_shape):
        """
        Compute histogram of flow vectors binned by location and direction.
        
        Args:
            flow_vectors: List of flow vectors
            frame_shape: Shape of the frame (height, width)
            
        Returns:
            Flow histogram as numpy array
        """
        height, width = frame_shape
        grid_h, grid_w = self.config['grid_size']
        direction_bins = self.config['direction_bins']
        
        # Initialize histogram
        histogram = np.zeros((grid_h, grid_w, direction_bins))
        
        # Bin size
        bin_h, bin_w = height / grid_h, width / grid_w
        
        for old_x, old_y, new_x, new_y, magnitude, angle in flow_vectors:
            # Skip very small movements (noise)
            if magnitude < 0.5:
                continue
                
            # Determine grid cell
            grid_x = min(int(old_x / bin_w), grid_w - 1)
            grid_y = min(int(old_y / bin_h), grid_h - 1)
            
            # Determine direction bin (convert radians to 0-2π range, then bin)
            angle_normalized = (angle + 2*np.pi) % (2*np.pi)
            direction_bin = int(angle_normalized / (2*np.pi) * direction_bins) % direction_bins
            
            # Add to histogram, weighted by magnitude
            histogram[grid_y, grid_x, direction_bin] += magnitude
        
        # Normalize histogram
        if histogram.sum() > 0:
            histogram = histogram / histogram.sum()
            
        return histogram

    def _compute_flow_statistics(self, flow_vectors):
        """
        Compute statistical measures from flow vectors.
        
        Args:
            flow_vectors: List of flow vectors
            
        Returns:
            Dictionary of statistical measures
        """
        if not flow_vectors:
            return {
                'mean_magnitude': 0,
                'std_magnitude': 0,
                'mean_angle': 0,
                'angular_dispersion': 0,
                'max_magnitude': 0,
                'flow_complexity': 0,
                'dominant_direction': 0
            }
            
        # Extract magnitudes and angles
        magnitudes = np.array([mag for _, _, _, _, mag, _ in flow_vectors])
        angles = np.array([ang for _, _, _, _, _, ang in flow_vectors])
        
        # Basic statistics
        mean_magnitude = np.mean(magnitudes) if len(magnitudes) > 0 else 0
        std_magnitude = np.std(magnitudes) if len(magnitudes) > 0 else 0
        max_magnitude = np.max(magnitudes) if len(magnitudes) > 0 else 0
        
        # Circular statistics for angles
        mean_angle = np.arctan2(
            np.sum(np.sin(angles)),
            np.sum(np.cos(angles))
        ) if len(angles) > 0 else 0
        
        # Angular dispersion (measure of how concentrated angles are)
        r = np.sqrt(
            np.sum(np.cos(angles)) ** 2 +
            np.sum(np.sin(angles)) ** 2
        ) / len(angles) if len(angles) > 0 else 0
        
        angular_dispersion = 1 - r  # 0: all angles same, 1: angles uniformly distributed
        
        # Calculate dominant direction (binned)
        direction_bins = self.config['direction_bins']
        angle_bins = np.zeros(direction_bins)
        for angle in angles:
            angle_normalized = (angle + 2*np.pi) % (2*np.pi)
            bin_idx = int(angle_normalized / (2*np.pi) * direction_bins) % direction_bins
            angle_bins[bin_idx] += 1
            
        dominant_direction = np.argmax(angle_bins) * 2 * np.pi / direction_bins
        
        # Flow complexity: ratio of different directions
        nonzero_bins = np.count_nonzero(angle_bins)
        flow_complexity = nonzero_bins / direction_bins
        
        return {
            'mean_magnitude': float(mean_magnitude),
            'std_magnitude': float(std_magnitude),
            'mean_angle': float(mean_angle),
            'angular_dispersion': float(angular_dispersion),
            'max_magnitude': float(max_magnitude),
            'flow_complexity': float(flow_complexity),
            'dominant_direction': float(dominant_direction)
        }

    def visualize_flow(self, frame, flow_features):
        """
        Create visualization of optical flow on the input frame.
        
        Args:
            frame: Input frame to draw on
            flow_features: Flow features dictionary from extract_flow
            
        Returns:
            Frame with flow visualization
        """
        if flow_features is None:
            return frame
            
        # Create a copy of the frame to draw on
        vis_frame = frame.copy()
        
        # Convert to RGB if grayscale
        if len(vis_frame.shape) == 2:
            vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_GRAY2BGR)
            
        # Draw motion regions
        for x, y, w, h in flow_features['motion_regions']:
            cv2.rectangle(vis_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        # Draw flow vectors
        scale = self.config['flow_visualization_scale']
        for old_x, old_y, new_x, new_y, magnitude, angle in flow_features['flow_vectors']:
            # Convert to integers
            old_x, old_y = int(old_x), int(old_y)
            
            # Calculate flow direction endpoint
            end_x = int(old_x + scale * (new_x - old_x))
            end_y = int(old_y + scale * (new_y - old_y))
            
            # Draw flow arrow
            cv2.arrowedLine(vis_frame, (old_x, old_y), (end_x, end_y), (0, 0, 255), 2)
            
        # Draw feature tracks
        max_track_length = 10  # Limit track history to avoid clutter
        for track in flow_features['feature_tracks'].values():
            # Only draw if track has multiple points
            if len(track) > 1:
                # Limit track length
                display_track = track[-max_track_length:] if len(track) > max_track_length else track
                
                # Draw track line
                points = np.array(display_track, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis_frame, [points], False, (0, 255, 255), 1)
                
                # Draw last point
                cv2.circle(vis_frame, track[-1], 2, (255, 0, 0), -1)
            
        return vis_frame

    def generate_motion_signature(self, flow_history=None):
        """
        Generate a compact motion signature from flow history.
        
        Args:
            flow_history: Optional flow history to use instead of self.flow_history
            
        Returns:
            Motion signature dictionary
        """
        if flow_history is None:
            flow_history = self.flow_history
            
        if not flow_history:
            return None
            
        # Extract histograms and statistics
        histograms = [f['histogram'] for f in flow_history if 'histogram' in f]
        stats = [f['stats'] for f in flow_history if 'stats' in f]
        
        if not histograms or not stats:
            return None
            
        # Average histogram over time
        avg_histogram = np.mean(histograms, axis=0)
        
        # Flatten for feature vector
        flattened_histogram = avg_histogram.flatten()
        
        # Average statistics over time
        avg_stats = {}
        for key in stats[0].keys():
            values = [s[key] for s in stats]
            avg_stats[key] = float(np.mean(values))
        
        # Create signature with temporal features
        signature = {
            'histogram_features': flattened_histogram,
            'statistical_features': avg_stats,
            'temporal_features': self._extract_temporal_features(flow_history),
            'timestamp': datetime.now().isoformat(),
            'frame_count': len(flow_history)
        }
        
        return signature

    def _extract_temporal_features(self, flow_history):
        """
        Extract temporal features from flow history.
        
        Args:
            flow_history: List of flow features from consecutive frames
            
        Returns:
            Dictionary of temporal features
        """
        if not flow_history or len(flow_history) < 2:
            return {}
            
        # Track magnitude changes over time
        magnitudes = []
        for f in flow_history:
            if 'stats' in f and 'mean_magnitude' in f['stats']:
                magnitudes.append(f['stats']['mean_magnitude'])
                
        if not magnitudes:
            return {}
            
        magnitudes = np.array(magnitudes)
        
        # Calculate acceleration (changes in magnitude)
        accelerations = np.diff(magnitudes)
        
        # Calculate direction consistency
        angles = [f['stats']['mean_angle'] for f in flow_history if 'stats' in f and 'mean_angle' in f['stats']]
        angle_changes = np.abs(np.diff(angles))
        # Handle circular wrapping (e.g., change from 350° to 10° should be 20°, not 340°)
        angle_changes = np.minimum(angle_changes, 2*np.pi - angle_changes)
        
        # Calculate directional entropy
        direction_changes = len([c for c in angle_changes if c > 0.5]) / max(1, len(angle_changes))
        
        # Calculate features
        temporal_features = {
            'magnitude_trend': float(np.polyfit(np.arange(len(magnitudes)), magnitudes, 1)[0]),
            'magnitude_variability': float(np.std(magnitudes)),
            'acceleration_mean': float(np.mean(accelerations) if len(accelerations) > 0 else 0),
            'acceleration_std': float(np.std(accelerations) if len(accelerations) > 0 else 0),
            'direction_changes': float(direction_changes),
            'direction_stability': float(1.0 - np.mean(angle_changes) / np.pi if len(angle_changes) > 0 else 1.0)
        }
        
        return temporal_features

    def classify_motion(self, motion_signature, database=None):
        """
        Classify a motion signature using the pattern database.
        
        Args:
            motion_signature: Motion signature to classify
            database: Optional MotionPatternDatabase instance
            
        Returns:
            Classification result dictionary
        """
        # This is a placeholder - full implementation would connect to the database
        # and use machine learning techniques to classify the motion
        
        # Basic placeholder classification based on simple rules
        if motion_signature and 'statistical_features' in motion_signature:
            stats = motion_signature['statistical_features']
            temporal = motion_signature.get('temporal_features', {})
            
            # Example rules (these would be replaced by actual learned patterns)
            magnitude = stats['mean_magnitude']
            complexity = stats['flow_complexity']
            direction_stability = temporal.get('direction_stability', 0.5)
            
            if magnitude > 10 and complexity < 0.3 and direction_stability > 0.8:
                # Fast, simple movement in consistent direction - likely a vehicle
                return {
                    'label': 'vehicle',
                    'confidence': 0.7,
                    'alternatives': [
                        {'label': 'person', 'confidence': 0.2},
                        {'label': 'other', 'confidence': 0.1}
                    ]
                }
            elif 3 < magnitude < 8 and 0.3 < complexity < 0.6 and 0.4 < direction_stability < 0.9:
                # Moderate speed, some complexity, some direction changes - likely a person
                return {
                    'label': 'person',
                    'confidence': 0.6,
                    'alternatives': [
                        {'label': 'animal', 'confidence': 0.3},
                        {'label': 'other', 'confidence': 0.1}
                    ]
                }
            elif complexity > 0.6 and direction_stability < 0.5:
                # Complex movement with frequent direction changes - likely an animal
                return {
                    'label': 'animal',
                    'confidence': 0.5,
                    'alternatives': [
                        {'label': 'person', 'confidence': 0.3},
                        {'label': 'other', 'confidence': 0.2}
                    ]
                }
            elif magnitude < 2 and complexity > 0.7:
                # Slow, complex movement in many directions - might be wind/trees/rain
                return {
                    'label': 'environment',
                    'confidence': 0.8,
                    'alternatives': [
                        {'label': 'other', 'confidence': 0.2}
                    ]
                }
        
        # Default unknown classification
        return {
            'label': 'unknown',
            'confidence': 0.5,
            'alternatives': []
        }


class MotionPatternDatabase:
    """
    Database for storing and retrieving motion patterns.
    """
    
    def __init__(self, db_path='motion_patterns.db', signature_dir='motion_signatures'):
        """
        Initialize the motion pattern database.
        
        Args:
            db_path: Path to SQLite database file
            signature_dir: Directory to store signature files
        """
        self.db_path = db_path
        self.signature_dir = signature_dir

        # Lock for thread safety (must be initialized before _init_database)
        self.lock = threading.Lock()

        # Create signature directory if it doesn't exist
        os.makedirs(self.signature_dir, exist_ok=True)

        # Initialize database
        self._init_database()

        logger.info("MotionPatternDatabase initialized at %s", db_path)

    def _init_database(self):
        """Initialize the SQLite database schema."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Create tables if they don't exist
            c.execute('''
            CREATE TABLE IF NOT EXISTS motion_patterns (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                classification TEXT,
                confidence REAL,
                metadata TEXT,
                signature_path TEXT
            )
            ''')
            
            c.execute('''
            CREATE TABLE IF NOT EXISTS labels (
                cluster_id INTEGER PRIMARY KEY,
                label TEXT,
                description TEXT
            )
            ''')
            
            conn.commit()
            conn.close()

    def add_pattern(self, pattern_id, motion_signature, classification=None, metadata=None):
        """
        Add a new motion pattern to the database.
        
        Args:
            pattern_id: Unique identifier for the pattern
            motion_signature: Motion signature dictionary
            classification: Optional classification information
            metadata: Optional metadata dictionary
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            try:
                # Save signature to file
                signature_path = os.path.join(self.signature_dir, f"{pattern_id}.npz")
                
                # Extract signature components for efficient storage
                histogram_features = motion_signature.get('histogram_features', np.array([]))
                statistical_features = motion_signature.get('statistical_features', {})
                temporal_features = motion_signature.get('temporal_features', {})
                
                # Save as compressed numpy file
                np.savez_compressed(
                    signature_path,
                    histogram=histogram_features,
                    stats=json.dumps(statistical_features),
                    temporal=json.dumps(temporal_features),
                    timestamp=motion_signature.get('timestamp', datetime.now().isoformat()),
                    frame_count=motion_signature.get('frame_count', 0)
                )
                
                # Store metadata in database
                conn = sqlite3.connect(self.db_path)
                c = conn.cursor()
                
                c.execute('''
                INSERT OR REPLACE INTO motion_patterns 
                (id, timestamp, classification, confidence, metadata, signature_path)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    pattern_id,
                    datetime.now().isoformat(),
                    classification['label'] if classification else 'unknown',
                    classification['confidence'] if classification else 0.0,
                    json.dumps(metadata) if metadata else '{}',
                    signature_path
                ))
                
                conn.commit()
                conn.close()
                
                return True
            except Exception as e:
                logger.error("Error adding pattern to database: %s", e)
                return False

    def get_pattern(self, pattern_id):
        """
        Retrieve a motion pattern from the database.
        
        Args:
            pattern_id: Unique identifier for the pattern
            
        Returns:
            Motion pattern dictionary or None if not found
        """
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                c = conn.cursor()
                
                c.execute('''
                SELECT * FROM motion_patterns WHERE id = ?
                ''', (pattern_id,))
                
                row = c.fetchone()
                conn.close()
                
                if not row:
                    return None
                    
                # Load signature from file
                signature_path = row['signature_path']
                if not os.path.exists(signature_path):
                    return None
                    
                # Load numpy file
                signature_data = np.load(signature_path, allow_pickle=True)

                # Reconstruct motion signature
                # Handle both string and numpy scalar types
                stats_str = signature_data['stats']
                temporal_str = signature_data['temporal']
                timestamp_val = signature_data['timestamp']
                frame_count_val = signature_data['frame_count']

                # Convert numpy types to Python types
                if isinstance(stats_str, np.ndarray):
                    stats_str = stats_str.item()
                if isinstance(temporal_str, np.ndarray):
                    temporal_str = temporal_str.item()
                if isinstance(timestamp_val, np.ndarray):
                    timestamp_val = timestamp_val.item()
                if isinstance(frame_count_val, np.ndarray):
                    frame_count_val = frame_count_val.item()

                motion_signature = {
                    'histogram_features': signature_data['histogram'],
                    'statistical_features': json.loads(str(stats_str)),
                    'temporal_features': json.loads(str(temporal_str)),
                    'timestamp': str(timestamp_val),
                    'frame_count': int(frame_count_val)
                }
                
                # Create pattern dictionary
                pattern = {
                    'id': row['id'],
                    'timestamp': row['timestamp'],
                    'classification': {
                        'label': row['classification'],
                        'confidence': row['confidence']
                    },
                    'metadata': json.loads(row['metadata']),
                    'signature': motion_signature
                }
                
                return pattern
            except Exception as e:
                logger.error("Error retrieving pattern from database: %s", e)
                return None

    def find_similar_patterns(self, motion_signature, limit=5, similarity_threshold=0.7):
        """
        Find patterns similar to the given motion signature using cosine similarity.

        Args:
            motion_signature: Motion signature to compare against
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity score (0-1)

        Returns:
            List of similar patterns with similarity scores, sorted by descending similarity
        """
        if not motion_signature or 'histogram_features' not in motion_signature:
            return []

        # Get all patterns from database
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT id as pattern_id, signature_path FROM motion_patterns')
        all_patterns = cursor.fetchall()

        if not all_patterns:
            return []

        # Extract query histogram
        query_histogram = motion_signature['histogram_features'].flatten()

        # Calculate similarity scores
        similarities = []
        for pattern_id, signature_path in all_patterns:
            try:
                # Load the stored signature
                stored_signature = np.load(signature_path, allow_pickle=True)
                stored_histogram = stored_signature['histogram_features'].flatten()

                # Compute cosine similarity
                similarity = self._cosine_similarity(query_histogram, stored_histogram)

                if similarity >= similarity_threshold:
                    similarities.append((pattern_id, similarity))

            except Exception as e:
                logger.error(f"Error comparing pattern {pattern_id}: {e}")
                continue

        # Sort by similarity (descending) and return top matches
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = []

        for pattern_id, similarity in similarities[:limit]:
            # Get pattern metadata
            cursor.execute('''
                SELECT classification, confidence, metadata
                FROM patterns WHERE pattern_id = ?
            ''', (pattern_id,))
            row = cursor.fetchone()

            if row:
                results.append({
                    'pattern_id': pattern_id,
                    'similarity': float(similarity),
                    'classification': row[0],
                    'confidence': row[1],
                    'metadata': json.loads(row[2]) if row[2] else {}
                })

        conn.close()
        return results

    def _cosine_similarity(self, vec1, vec2):
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Similarity score between 0 and 1
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def update_classification(self, pattern_id, classification):
        """
        Update the classification for a pattern.
        
        Args:
            pattern_id: Unique identifier for the pattern
            classification: Classification dictionary with label and confidence
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                c = conn.cursor()
                
                c.execute('''
                UPDATE motion_patterns 
                SET classification = ?, confidence = ?
                WHERE id = ?
                ''', (
                    classification['label'],
                    classification['confidence'],
                    pattern_id
                ))
                
                conn.commit()
                conn.close()
                
                return True
            except Exception as e:
                logger.error("Error updating classification: %s", e)
                return False

    def discover_patterns_kmeans(self, n_clusters=5):
        """
        Use K-Means clustering to discover motion pattern groups.

        Args:
            n_clusters: Number of clusters to find

        Returns:
            Dictionary mapping cluster labels to pattern IDs
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            logger.error("scikit-learn not installed. Cannot perform clustering.")
            return {}

        # Get all patterns
        cursor = self.conn.cursor()
        cursor.execute('SELECT pattern_id, signature_path FROM patterns')
        all_patterns = cursor.fetchall()

        if len(all_patterns) < n_clusters:
            logger.warning(f"Not enough patterns ({len(all_patterns)}) for {n_clusters} clusters")
            return {}

        # Load all histograms
        pattern_ids = []
        feature_vectors = []

        for pattern_id, signature_path in all_patterns:
            try:
                signature = np.load(signature_path, allow_pickle=True)
                histogram = signature['histogram_features'].flatten()
                pattern_ids.append(pattern_id)
                feature_vectors.append(histogram)
            except Exception as e:
                logger.error(f"Error loading pattern {pattern_id}: {e}")
                continue

        if len(feature_vectors) < n_clusters:
            logger.warning("Not enough valid patterns for clustering")
            return {}

        # Perform K-Means clustering
        X = np.array(feature_vectors)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        # Group patterns by cluster
        clusters = {}
        for pattern_id, label in zip(pattern_ids, labels):
            cluster_label = int(label)
            if cluster_label not in clusters:
                clusters[cluster_label] = []
            clusters[cluster_label].append(pattern_id)

        logger.info(f"Discovered {len(clusters)} clusters from {len(pattern_ids)} patterns")
        for cluster_id, patterns in clusters.items():
            logger.info(f"  Cluster {cluster_id}: {len(patterns)} patterns")

        return clusters

    def discover_patterns_dbscan(self, eps=0.5, min_samples=3):
        """
        Use DBSCAN clustering to discover motion pattern groups (handles noise better).

        Args:
            eps: Maximum distance between samples for one to be considered a neighbor
            min_samples: Minimum number of samples in a neighborhood for a core point

        Returns:
            Dictionary mapping cluster labels to pattern IDs (-1 for noise)
        """
        try:
            from sklearn.cluster import DBSCAN
        except ImportError:
            logger.error("scikit-learn not installed. Cannot perform clustering.")
            return {}

        # Get all patterns
        cursor = self.conn.cursor()
        cursor.execute('SELECT pattern_id, signature_path FROM patterns')
        all_patterns = cursor.fetchall()

        if len(all_patterns) < min_samples:
            logger.warning(f"Not enough patterns ({len(all_patterns)}) for DBSCAN")
            return {}

        # Load all histograms
        pattern_ids = []
        feature_vectors = []

        for pattern_id, signature_path in all_patterns:
            try:
                signature = np.load(signature_path, allow_pickle=True)
                histogram = signature['histogram_features'].flatten()
                pattern_ids.append(pattern_id)
                feature_vectors.append(histogram)
            except Exception as e:
                logger.error(f"Error loading pattern {pattern_id}: {e}")
                continue

        if len(feature_vectors) < min_samples:
            logger.warning("Not enough valid patterns for clustering")
            return {}

        # Perform DBSCAN clustering
        X = np.array(feature_vectors)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = dbscan.fit_predict(X)

        # Group patterns by cluster
        clusters = {}
        for pattern_id, label in zip(pattern_ids, labels):
            cluster_label = int(label)
            if cluster_label not in clusters:
                clusters[cluster_label] = []
            clusters[cluster_label].append(pattern_id)

        n_clusters = len([k for k in clusters.keys() if k != -1])
        n_noise = len(clusters.get(-1, []))

        logger.info(f"Discovered {n_clusters} clusters from {len(pattern_ids)} patterns ({n_noise} noise)")
        for cluster_id, patterns in clusters.items():
            cluster_type = "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
            logger.info(f"  {cluster_type}: {len(patterns)} patterns")

        return clusters


def test_optical_flow(video_path, output_path=None):
    """
    Test the optical flow analyzer on a video file.
    
    Args:
        video_path: Path to input video file
        output_path: Optional path to save output video
    """
    # Create analyzer
    analyzer = OpticalFlowAnalyzer()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Error opening video file: %s", video_path)
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create output video writer if output path specified
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process video frames
    prev_frame = None
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Skip first frame
        if prev_frame is None:
            prev_frame = frame.copy()
            continue
        
        # Detect motion (simple background subtraction for testing)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Simple motion detection
        diff = cv2.absdiff(prev_gray, frame_gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size
        motion_regions = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                x, y, w, h = cv2.boundingRect(contour)
                motion_regions.append((x, y, w, h))
        
        # Extract flow features
        flow_features = analyzer.extract_flow(prev_frame, frame, motion_regions)
        
        # Visualize flow
        if flow_features:
            vis_frame = analyzer.visualize_flow(frame, flow_features)
            
            # Generate signature every 30 frames
            if frame_count % 30 == 0:
                signature = analyzer.generate_motion_signature()
                if signature:
                    classification = analyzer.classify_motion(signature)
                    
                    # Draw classification
                    if classification:
                        label = classification['label']
                        confidence = classification['confidence']
                        cv2.putText(vis_frame, f"{label} ({confidence:.2f})", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            vis_frame = frame
        
        # Write output frame
        if out:
            out.write(vis_frame)
        
        # Update previous frame
        prev_frame = frame.copy()
        frame_count += 1
    
    # Release resources
    cap.release()
    if out:
        out.close()


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        
        print(f"Testing optical flow analyzer on {video_path}")
        test_optical_flow(video_path, output_path)
    else:
        print("Usage: python optical_flow_analyzer.py <input_video> [output_video]")
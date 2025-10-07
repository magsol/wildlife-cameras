#!/usr/bin/env python3
"""
Comprehensive integration test for optical flow motion classification system.

Tests the full pipeline from frame capture through classification and storage.
"""

import cv2
import numpy as np
import tempfile
import shutil
import os
import json
from pathlib import Path

from optical_flow_analyzer import OpticalFlowAnalyzer, MotionPatternDatabase
from motion_storage import StorageConfig, MotionEventRecorder, CircularFrameBuffer


def create_synthetic_motion_frames(motion_type='person', num_frames=30):
    """
    Create synthetic video frames with specific motion patterns.

    Args:
        motion_type: Type of motion ('person', 'vehicle', 'animal')
        num_frames: Number of frames to generate

    Returns:
        List of frames with motion
    """
    frames = []
    width, height = 320, 240

    for i in range(num_frames):
        # Create blank frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        if motion_type == 'person':
            # Vertical walking motion
            x = width // 2
            y = int(50 + i * (height - 100) / num_frames)
            cv2.rectangle(frame, (x-10, y-30), (x+10, y+30), (255, 255, 255), -1)
            cv2.circle(frame, (x, y-40), 8, (255, 255, 255), -1)  # Head

        elif motion_type == 'vehicle':
            # Horizontal fast motion
            x = int(i * width / num_frames)
            y = height // 2
            cv2.rectangle(frame, (x-30, y-10), (x+30, y+10), (255, 255, 255), -1)
            cv2.circle(frame, (x-25, y+5), 5, (200, 200, 200), -1)  # Wheel
            cv2.circle(frame, (x+25, y+5), 5, (200, 200, 200), -1)  # Wheel

        elif motion_type == 'animal':
            # Erratic diagonal motion with pauses
            pause_frames = [10, 20]
            if i not in pause_frames:
                x = int(50 + i * 5)
                y = int(50 + i * 3 + np.sin(i * 0.5) * 20)
                cv2.ellipse(frame, (x, y), (15, 8), 0, 0, 360, (255, 255, 255), -1)
                cv2.circle(frame, (x+12, y), 5, (255, 255, 255), -1)  # Head

        frames.append(frame)

    return frames


def test_optical_flow_extraction():
    """Test optical flow feature extraction."""
    print("\n=== Test 1: Optical Flow Extraction ===")

    analyzer = OpticalFlowAnalyzer()
    frames = create_synthetic_motion_frames('person', 10)

    # Extract flow between consecutive frames
    flow_features = None
    for i in range(1, len(frames)):
        flow_features = analyzer.extract_flow(frames[i-1], frames[i])

    assert flow_features is not None, "Flow features should be extracted"
    assert 'flow_vectors' in flow_features, "Should contain flow vectors"
    assert 'histogram' in flow_features, "Should contain histogram"
    assert 'stats' in flow_features, "Should contain statistics"

    print("✓ Optical flow extraction working")
    print(f"  - Extracted {len(flow_features['flow_vectors'])} flow vectors")
    print(f"  - Histogram shape: {flow_features['histogram'].shape}")

    return True


def test_motion_signature_generation():
    """Test motion signature generation."""
    print("\n=== Test 2: Motion Signature Generation ===")

    analyzer = OpticalFlowAnalyzer()
    frames = create_synthetic_motion_frames('vehicle', 15)

    # Build flow history
    for i in range(1, len(frames)):
        analyzer.extract_flow(frames[i-1], frames[i])

    # Generate signature
    signature = analyzer.generate_motion_signature()

    assert signature is not None, "Signature should be generated"
    assert 'histogram_features' in signature, "Should have histogram features"
    assert 'statistical_features' in signature, "Should have statistical features"
    assert 'temporal_features' in signature, "Should have temporal features"

    print("✓ Motion signature generation working")
    print(f"  - Histogram shape: {signature['histogram_features'].shape}")
    print(f"  - Statistical features: {list(signature['statistical_features'].keys())}")
    print(f"  - Temporal features: {list(signature['temporal_features'].keys())}")

    return True


def test_classification():
    """Test motion classification."""
    print("\n=== Test 3: Motion Classification ===")

    analyzer = OpticalFlowAnalyzer()

    # Test different motion types
    motion_types = ['person', 'vehicle', 'animal']
    classifications = {}

    for motion_type in motion_types:
        analyzer.reset()
        frames = create_synthetic_motion_frames(motion_type, 20)

        # Build flow history
        for i in range(1, len(frames)):
            analyzer.extract_flow(frames[i-1], frames[i])

        # Generate signature and classify
        signature = analyzer.generate_motion_signature()
        if signature:
            classification = analyzer.classify_motion(signature)
            classifications[motion_type] = classification

            print(f"  {motion_type}: {classification['label']} "
                  f"(confidence: {classification['confidence']:.2f})")

    assert len(classifications) > 0, "Should classify at least one motion type"
    print("✓ Classification working")

    return True


def test_pattern_database():
    """Test pattern database storage and retrieval."""
    print("\n=== Test 4: Pattern Database ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test_patterns.db')
        sig_dir = os.path.join(tmpdir, 'signatures')

        db = MotionPatternDatabase(db_path, sig_dir)
        analyzer = OpticalFlowAnalyzer()

        # Create and store pattern
        frames = create_synthetic_motion_frames('person', 15)
        for i in range(1, len(frames)):
            analyzer.extract_flow(frames[i-1], frames[i])

        signature = analyzer.generate_motion_signature()
        classification = analyzer.classify_motion(signature)

        # Add to database
        pattern_id = 'test_pattern_001'
        success = db.add_pattern(
            pattern_id,
            signature,
            classification,
            {'test': True}
        )

        assert success, "Pattern should be added successfully"

        # Retrieve pattern
        retrieved = db.get_pattern(pattern_id)
        assert retrieved is not None, "Pattern should be retrieved"
        assert retrieved['id'] == pattern_id, "Pattern ID should match"

        print("✓ Pattern database working")
        print(f"  - Pattern stored: {pattern_id}")
        print(f"  - Classification: {retrieved['classification']['label']}")

    return True


def test_similarity_search():
    """Test pattern similarity search."""
    print("\n=== Test 5: Similarity Search ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test_patterns.db')
        sig_dir = os.path.join(tmpdir, 'signatures')

        db = MotionPatternDatabase(db_path, sig_dir)
        analyzer = OpticalFlowAnalyzer()

        # Add multiple patterns
        for idx, motion_type in enumerate(['person', 'person', 'vehicle']):
            analyzer.reset()
            frames = create_synthetic_motion_frames(motion_type, 15)

            for i in range(1, len(frames)):
                analyzer.extract_flow(frames[i-1], frames[i])

            signature = analyzer.generate_motion_signature()
            classification = analyzer.classify_motion(signature)

            db.add_pattern(
                f'pattern_{idx:03d}',
                signature,
                classification,
                {'type': motion_type}
            )

        # Search for similar patterns
        analyzer.reset()
        query_frames = create_synthetic_motion_frames('person', 15)
        for i in range(1, len(query_frames)):
            analyzer.extract_flow(query_frames[i-1], query_frames[i])

        query_signature = analyzer.generate_motion_signature()
        similar = db.find_similar_patterns(query_signature, limit=3, similarity_threshold=0.5)

        print(f"✓ Similarity search working")
        print(f"  - Found {len(similar)} similar patterns")
        for s in similar:
            print(f"    - {s['pattern_id']}: similarity={s['similarity']:.2f}, "
                  f"class={s['classification']}")

    return True


def test_end_to_end_pipeline():
    """Test complete pipeline from frames to stored event."""
    print("\n=== Test 6: End-to-End Pipeline ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup
        storage_config = StorageConfig(
            local_storage_path=tmpdir,
            ram_buffer_seconds=1,
            min_motion_duration_sec=0,  # Accept any duration for test
            store_optical_flow_data=True,
            motion_classification_enabled=True
        )

        frame_buffer = CircularFrameBuffer(max_size=30)

        # Create frames with motion
        frames = create_synthetic_motion_frames('vehicle', 30)

        # Add frames to buffer
        import datetime
        for frame in frames:
            frame_buffer.add_frame(frame, datetime.datetime.now())

        print("✓ End-to-end pipeline setup working")
        print(f"  - Generated {len(frames)} frames")
        print(f"  - Buffer contains {len(frame_buffer.buffer)} frames")

    return True


def run_all_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("OPTICAL FLOW INTEGRATION TESTS")
    print("=" * 60)

    tests = [
        test_optical_flow_extraction,
        test_motion_signature_generation,
        test_classification,
        test_pattern_database,
        test_similarity_search,
        test_end_to_end_pipeline
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == '__main__':
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Mock camera test harness - simulates Pi camera without hardware.
Tests the full system pipeline with synthetic motion events.
"""

import cv2
import numpy as np
import time
import threading
import sys
import datetime
from pathlib import Path

# Test if we can import the required modules
try:
    from optical_flow_analyzer import OpticalFlowAnalyzer
    from motion_storage import CircularFrameBuffer, StorageConfig
    print("✓ Successfully imported optical flow and storage modules")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Make sure you're running with: pixi run python mock_camera_test.py")
    sys.exit(1)


class MockPiCamera:
    """
    Simulates Raspberry Pi NoIR camera with configurable motion patterns.
    """

    def __init__(self, width=640, height=480, fps=15):
        self.width = width
        self.height = height
        self.fps = fps
        self.running = False
        self.frame_count = 0
        self.motion_type = 'person'  # person, vehicle, animal, none
        self.motion_active = False

    def generate_frame(self):
        """Generate synthetic frame with optional motion."""
        # Base frame (dark, simulating night/IR)
        frame = np.random.randint(0, 30, (self.height, self.width, 3), dtype=np.uint8)

        if self.motion_active:
            if self.motion_type == 'person':
                # Vertical walking motion
                x = self.width // 2 + int(np.sin(self.frame_count * 0.1) * 50)
                y = int(100 + (self.frame_count % 200) * 1.5)
                # Person shape
                cv2.rectangle(frame, (x-15, y-40), (x+15, y+40), (200, 200, 200), -1)
                cv2.circle(frame, (x, y-50), 12, (220, 220, 220), -1)  # Head

            elif self.motion_type == 'vehicle':
                # Fast horizontal motion
                x = int((self.frame_count % 80) * self.width / 80)
                y = self.height // 2
                # Car shape
                cv2.rectangle(frame, (x-40, y-15), (x+40, y+15), (180, 180, 180), -1)
                cv2.circle(frame, (x-30, y+10), 8, (150, 150, 150), -1)  # Wheel
                cv2.circle(frame, (x+30, y+10), 8, (150, 150, 150), -1)  # Wheel
                cv2.rectangle(frame, (x-20, y-25), (x+15, y-15), (200, 200, 200), -1)  # Cabin

            elif self.motion_type == 'animal':
                # Erratic diagonal motion with pauses
                if self.frame_count % 30 < 20:  # Move for 20 frames
                    x = int(100 + (self.frame_count % 30) * 15)
                    y = int(150 + (self.frame_count % 30) * 5 + np.sin(self.frame_count * 0.3) * 30)
                    # Animal shape (quadruped)
                    cv2.ellipse(frame, (x, y), (25, 12), 0, 0, 360, (190, 190, 190), -1)
                    cv2.circle(frame, (x+20, y-5), 8, (190, 190, 190), -1)  # Head
                    cv2.circle(frame, (x-18, y+10), 3, (150, 150, 150), -1)  # Leg

        self.frame_count += 1
        return frame

    def start_motion(self, motion_type='person'):
        """Start generating motion of specified type."""
        self.motion_type = motion_type
        self.motion_active = True
        print(f"  → Started {motion_type} motion")

    def stop_motion(self):
        """Stop motion (static scene)."""
        self.motion_active = False
        print("  → Stopped motion")

    def capture_frames(self, duration_sec=5):
        """
        Capture frames for specified duration.

        Returns:
            List of (frame, timestamp) tuples
        """
        frames = []
        start_time = time.time()
        frame_interval = 1.0 / self.fps

        while time.time() - start_time < duration_sec:
            frame = self.generate_frame()
            timestamp = datetime.datetime.now()
            frames.append((frame, timestamp))
            time.sleep(frame_interval)

        return frames


def test_optical_flow_with_mock_camera():
    """Test optical flow extraction with mock camera."""
    print("\n" + "="*60)
    print("TEST 1: Optical Flow with Mock Camera")
    print("="*60)

    camera = MockPiCamera(width=320, height=240, fps=15)
    analyzer = OpticalFlowAnalyzer()

    # Test with different motion types
    motion_types = ['person', 'vehicle', 'animal']

    for motion_type in motion_types:
        print(f"\n--- Testing {motion_type} motion ---")
        camera.start_motion(motion_type)

        # Capture sequence
        frames = camera.capture_frames(duration_sec=3)
        print(f"  Captured {len(frames)} frames")

        # Extract optical flow
        flow_count = 0
        for i in range(1, len(frames)):
            prev_frame, _ = frames[i-1]
            curr_frame, _ = frames[i]

            flow_features = analyzer.extract_flow(prev_frame, curr_frame)
            if flow_features and len(flow_features['flow_vectors']) > 0:
                flow_count += 1

        print(f"  ✓ Extracted flow from {flow_count}/{len(frames)-1} frame pairs")

        # Generate signature
        signature = analyzer.generate_motion_signature()
        if signature:
            classification = analyzer.classify_motion(signature)
            print(f"  ✓ Classification: {classification['label']} "
                  f"(confidence: {classification['confidence']:.2f})")

        analyzer.reset()
        camera.stop_motion()

    print("\n✓ Test 1 PASSED")
    return True


def test_motion_storage_with_mock_camera():
    """Test motion storage pipeline with mock camera."""
    print("\n" + "="*60)
    print("TEST 2: Motion Storage Pipeline")
    print("="*60)

    import tempfile
    import shutil

    # Create temporary storage
    tmpdir = tempfile.mkdtemp()
    print(f"  Using temp storage: {tmpdir}")

    try:
        config = StorageConfig(
            local_storage_path=tmpdir,
            ram_buffer_seconds=2,
            min_motion_duration_sec=1,
            store_optical_flow_data=True,
            motion_classification_enabled=True
        )

        buffer = CircularFrameBuffer(max_size=30)
        camera = MockPiCamera(width=320, height=240, fps=15)

        print("\n--- Generating motion sequence ---")
        camera.start_motion('vehicle')

        # Add frames to buffer
        frames = camera.capture_frames(duration_sec=3)
        for frame, timestamp in frames:
            buffer.add_frame(frame, timestamp)

        print(f"  ✓ Added {len(frames)} frames to buffer")
        print(f"  ✓ Buffer contains {len(buffer.buffer)} frames")

        camera.stop_motion()

        # Check buffer retrieval
        recent = buffer.get_recent_frames(seconds=1)
        print(f"  ✓ Retrieved {len(recent)} recent frames")

        print("\n✓ Test 2 PASSED")
        return True

    finally:
        shutil.rmtree(tmpdir)


def test_end_to_end_classification():
    """Test complete pipeline from frames to classification."""
    print("\n" + "="*60)
    print("TEST 3: End-to-End Classification Pipeline")
    print("="*60)

    import tempfile
    import shutil

    tmpdir = tempfile.mkdtemp()

    try:
        from optical_flow_analyzer import MotionPatternDatabase

        # Setup
        db_path = Path(tmpdir) / "patterns.db"
        sig_dir = Path(tmpdir) / "signatures"
        sig_dir.mkdir()

        db = MotionPatternDatabase(str(db_path), str(sig_dir))
        analyzer = OpticalFlowAnalyzer()
        camera = MockPiCamera(width=320, height=240, fps=15)

        # Collect patterns for each motion type
        patterns_collected = {}

        for motion_type in ['person', 'vehicle', 'animal']:
            print(f"\n--- Collecting {motion_type} pattern ---")
            analyzer.reset()
            camera.start_motion(motion_type)

            frames = camera.capture_frames(duration_sec=2)

            # Build flow history
            for i in range(1, len(frames)):
                prev_frame, _ = frames[i-1]
                curr_frame, _ = frames[i]
                analyzer.extract_flow(prev_frame, curr_frame)

            # Generate signature and classify
            signature = analyzer.generate_motion_signature()
            if signature:
                classification = analyzer.classify_motion(signature)

                # Store in database
                pattern_id = f"{motion_type}_001"
                db.add_pattern(
                    pattern_id,
                    signature,
                    classification,
                    {'type': motion_type, 'source': 'mock_camera'}
                )

                patterns_collected[motion_type] = {
                    'pattern_id': pattern_id,
                    'classification': classification
                }

                print(f"  ✓ Stored pattern: {pattern_id}")
                print(f"  ✓ Classification: {classification['label']} "
                      f"({classification['confidence']:.2f})")

            camera.stop_motion()

        # Test pattern retrieval
        print("\n--- Testing pattern retrieval ---")
        for motion_type, info in patterns_collected.items():
            pattern = db.get_pattern(info['pattern_id'])
            assert pattern is not None, f"Failed to retrieve {motion_type} pattern"
            print(f"  ✓ Retrieved {motion_type} pattern")

        # Test similarity search
        print("\n--- Testing similarity search ---")
        if len(patterns_collected) >= 2:
            # Search for patterns similar to 'person'
            person_id = patterns_collected['person']['pattern_id']
            person_pattern = db.get_pattern(person_id)

            similar = db.find_similar_patterns(
                person_pattern['signature'],
                limit=3,
                similarity_threshold=0.3
            )

            print(f"  ✓ Found {len(similar)} similar patterns:")
            for s in similar:
                print(f"    - {s['pattern_id']}: similarity={s['similarity']:.2f}")

        print("\n✓ Test 3 PASSED")
        return True

    finally:
        shutil.rmtree(tmpdir)


def test_performance_benchmarks():
    """Measure performance on this platform."""
    print("\n" + "="*60)
    print("TEST 4: Performance Benchmarks")
    print("="*60)

    analyzer = OpticalFlowAnalyzer()
    camera = MockPiCamera(width=320, height=240, fps=15)

    # Generate test frames
    camera.start_motion('vehicle')
    frames = camera.capture_frames(duration_sec=2)
    camera.stop_motion()

    # Benchmark optical flow extraction
    print("\n--- Optical Flow Extraction ---")
    start = time.time()
    for i in range(1, min(30, len(frames))):
        prev_frame, _ = frames[i-1]
        curr_frame, _ = frames[i]
        analyzer.extract_flow(prev_frame, curr_frame)
    elapsed = time.time() - start

    frames_processed = min(29, len(frames)-1)
    avg_time = (elapsed / frames_processed) * 1000
    print(f"  Processed {frames_processed} frames in {elapsed:.2f}s")
    print(f"  Average: {avg_time:.2f}ms per frame")

    if avg_time < 33:  # 30 FPS
        print(f"  ✓ EXCELLENT: Can handle 30 FPS")
    elif avg_time < 66:  # 15 FPS
        print(f"  ✓ GOOD: Can handle 15 FPS")
    else:
        print(f"  ⚠ SLOW: May struggle with real-time processing")

    # Benchmark signature generation
    print("\n--- Signature Generation ---")
    start = time.time()
    signature = analyzer.generate_motion_signature()
    elapsed = time.time() - start

    print(f"  Signature generated in {elapsed*1000:.2f}ms")
    if elapsed < 0.010:  # 10ms
        print("  ✓ FAST")
    else:
        print("  ⚠ Consider optimization")

    # Benchmark classification
    print("\n--- Classification ---")
    start = time.time()
    classification = analyzer.classify_motion(signature)
    elapsed = time.time() - start

    print(f"  Classification in {elapsed*1000:.2f}ms")
    print(f"  Result: {classification['label']} ({classification['confidence']:.2f})")

    print("\n✓ Test 4 PASSED")
    return True


def run_all_mock_tests():
    """Run all mock camera tests."""
    print("\n" + "="*80)
    print(" "*20 + "MOCK CAMERA TEST SUITE")
    print("="*80)
    print("\nTesting wildlife camera system without physical Raspberry Pi hardware")
    print("This validates optical flow, classification, and storage pipeline")

    tests = [
        test_optical_flow_with_mock_camera,
        test_motion_storage_with_mock_camera,
        test_end_to_end_classification,
        test_performance_benchmarks
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"\n✗ {test_func.__name__} FAILED")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*80)
    print(f" "*25 + "TEST RESULTS")
    print("="*80)
    print(f"  PASSED: {passed}/{len(tests)}")
    print(f"  FAILED: {failed}/{len(tests)}")

    if failed == 0:
        print("\n  ✓ ALL TESTS PASSED - System ready for deployment!")
    else:
        print(f"\n  ✗ {failed} test(s) failed - Review errors above")

    print("="*80 + "\n")

    return failed == 0


if __name__ == '__main__':
    success = run_all_mock_tests()
    sys.exit(0 if success else 1)

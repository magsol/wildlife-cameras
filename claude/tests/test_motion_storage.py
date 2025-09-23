import os
import sys
import unittest
import tempfile
import shutil
import time
import datetime
import json
import cv2
import numpy as np
from unittest.mock import MagicMock, patch, mock_open
from threading import Condition

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import test helpers to set up mocks before importing the modules
from tests.test_helpers import setup_picamera2_mocks
setup_picamera2_mocks()

# Import module to test
from motion_storage import (
    StorageConfig,
    CircularFrameBuffer,
    MotionEventRecorder,
    WiFiMonitor,
    TransferManager
)

class TestStorageConfig(unittest.TestCase):
    """Test cases for StorageConfig class"""

    def test_default_values(self):
        """Test that default values are set correctly"""
        config = StorageConfig()
        self.assertEqual(config.ram_buffer_seconds, 30)
        self.assertEqual(config.max_ram_segments, 300)
        self.assertEqual(config.local_storage_path, "/tmp/motion_events")
        self.assertEqual(config.max_disk_usage_mb, 1000)
        self.assertEqual(config.min_motion_duration_sec, 3)
        self.assertEqual(config.upload_throttle_kbps, 500)
        self.assertTrue(config.wifi_monitoring)
        self.assertTrue(config.generate_thumbnails)
        self.assertTrue(config.chunk_upload)

    def test_custom_values(self):
        """Test that custom values can be set"""
        config = StorageConfig()
        config.ram_buffer_seconds = 60
        config.max_disk_usage_mb = 2000
        config.wifi_monitoring = False
        
        self.assertEqual(config.ram_buffer_seconds, 60)
        self.assertEqual(config.max_disk_usage_mb, 2000)
        self.assertFalse(config.wifi_monitoring)


class TestCircularFrameBuffer(unittest.TestCase):
    """Test cases for CircularFrameBuffer class"""

    def setUp(self):
        """Set up test environment"""
        self.buffer = CircularFrameBuffer(max_size=5)
        self.timestamp = datetime.datetime.now()
        
        # Create test frames
        self.frames = []
        for i in range(10):
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.putText(frame, str(i), (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (255, 255, 255), 2)
            self.frames.append(frame)

    def test_initialization(self):
        """Test proper initialization"""
        self.assertEqual(self.buffer.buffer.maxlen, 5)
        self.assertIsInstance(self.buffer.lock, type(Condition()))
        self.assertEqual(len(self.buffer.buffer), 0)

    def test_add_frame(self):
        """Test adding frames to the buffer"""
        # Add a frame
        self.buffer.add_frame(self.frames[0], self.timestamp)
        self.assertEqual(len(self.buffer.buffer), 1)
        
        # Add more frames than the buffer can hold
        for i in range(1, 10):
            new_timestamp = self.timestamp + datetime.timedelta(seconds=i)
            self.buffer.add_frame(self.frames[i], new_timestamp)
        
        # Check that buffer maintained its size limit
        self.assertEqual(len(self.buffer.buffer), 5)
        
        # Check that the buffer contains the latest 5 frames (5-9)
        stored_indices = [np.sum(frame[50, 40] == 255) for frame, _ in self.buffer.buffer]
        expected_indices = list(range(5, 10))
        self.assertEqual(stored_indices, expected_indices)

    def test_get_recent_frames(self):
        """Test retrieving recent frames"""
        # Add frames with 1 second intervals
        for i in range(5):
            new_timestamp = self.timestamp + datetime.timedelta(seconds=i)
            self.buffer.add_frame(self.frames[i], new_timestamp)
        
        # Get frames from the last 2 seconds
        recent_frames = self.buffer.get_recent_frames(2)
        
        # Should return 3 frames (current, 1 sec ago, 2 secs ago)
        self.assertEqual(len(recent_frames), 3)
        
        # Get frames from the last 10 seconds (more than we have)
        all_frames = self.buffer.get_recent_frames(10)
        self.assertEqual(len(all_frames), 5)  # Should return all available frames
        
        # Test with empty buffer
        empty_buffer = CircularFrameBuffer(max_size=5)
        self.assertEqual(len(empty_buffer.get_recent_frames(5)), 0)


class TestMotionEventRecorder(unittest.TestCase):
    """Test cases for MotionEventRecorder class"""

    def setUp(self):
        """Set up test environment"""
        # Create temp directory for test
        self.test_dir = tempfile.mkdtemp()
        
        # Create test config with test directory
        self.config = StorageConfig()
        self.config.local_storage_path = self.test_dir
        self.config.generate_thumbnails = False  # Disable thumbnails for testing
        
        # Create circular buffer with test frames
        self.frame_buffer = CircularFrameBuffer(max_size=30)
        
        # Create test frames
        self.frames = []
        for i in range(10):
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.putText(frame, str(i), (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (255, 255, 255), 2)
            self.frames.append(frame)
            
        # Add frames to buffer with timestamps
        timestamp = datetime.datetime.now()
        for i, frame in enumerate(self.frames):
            new_timestamp = timestamp + datetime.timedelta(seconds=i)
            self.frame_buffer.add_frame(frame, new_timestamp)
        
        # Create recorder
        self.recorder = MotionEventRecorder(self.frame_buffer, self.config)
        
        # Test motion regions
        self.motion_regions = [(10, 10, 30, 30), (50, 50, 20, 20)]

    def tearDown(self):
        """Clean up after test"""
        # Remove temp directory
        shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """Test proper initialization"""
        self.assertEqual(self.recorder.frame_buffer, self.frame_buffer)
        self.assertEqual(self.recorder.config, self.config)
        self.assertFalse(self.recorder.recording)
        self.assertIsNone(self.recorder.current_event)
        self.assertTrue(os.path.exists(self.test_dir))

    def test_start_recording(self):
        """Test starting a recording"""
        self.recorder.start_recording(self.motion_regions)
        
        self.assertTrue(self.recorder.recording)
        self.assertIsNotNone(self.recorder.current_event)
        self.assertEqual(self.recorder.current_event["regions"], self.motion_regions)
        
        # Check that pre-motion frames were added
        self.assertTrue(len(self.recorder.current_event["frames"]) > 0)

    def test_add_frame(self):
        """Test adding frames to a recording"""
        # Start recording
        self.recorder.start_recording(self.motion_regions)
        initial_frame_count = len(self.recorder.current_event["frames"])
        
        # Add new frames
        for i in range(5):
            new_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            self.recorder.add_frame(new_frame, self.motion_regions)
            
        # Check that frames were added
        new_frame_count = len(self.recorder.current_event["frames"])
        self.assertEqual(new_frame_count, initial_frame_count + 5)

    @patch('motion_storage.MotionEventRecorder._encode_video')
    def test_stop_recording_sufficient_duration(self, mock_encode):
        """Test stopping a recording with sufficient duration"""
        # Mock the _encode_video method to avoid actual encoding
        mock_encode.return_value = None
        
        # Start recording
        self.recorder.start_recording(self.motion_regions)
        
        # Set start time to ensure sufficient duration
        self.recorder.current_event["start_time"] = datetime.datetime.now() - datetime.timedelta(seconds=5)
        
        # Stop recording
        self.recorder.stop_recording()
        
        # Check that recording stopped
        self.assertFalse(self.recorder.recording)
        self.assertIsNone(self.recorder.current_event)
        
        # Event should have been added to the queue for processing
        self.assertTrue(not self.recorder.events_queue.empty())

    def test_stop_recording_insufficient_duration(self):
        """Test stopping a recording with insufficient duration"""
        # Start recording
        self.recorder.start_recording(self.motion_regions)
        
        # Set start time to ensure insufficient duration
        self.recorder.current_event["start_time"] = datetime.datetime.now() - datetime.timedelta(
            seconds=self.config.min_motion_duration_sec - 1)
        
        # Remember event ID
        event_id = self.recorder.current_event["id"]
        
        # Stop recording
        self.recorder.stop_recording()
        
        # Check that recording stopped
        self.assertFalse(self.recorder.recording)
        self.assertIsNone(self.recorder.current_event)
        
        # Event should have been discarded (queue should be empty)
        self.assertTrue(self.recorder.events_queue.empty())

    @patch('motion_storage.cv2.VideoWriter')
    def test_encode_video(self, mock_video_writer):
        """Test video encoding"""
        # Mock VideoWriter
        mock_writer = MagicMock()
        mock_video_writer.return_value = mock_writer
        
        # Create test frames
        test_frames = [(self.frames[0], datetime.datetime.now())]
        
        # Call _encode_video method
        with tempfile.NamedTemporaryFile(suffix='.mp4') as temp_file:
            self.recorder._encode_video(test_frames, temp_file.name)
            
            # Check that VideoWriter was called correctly
            mock_video_writer.assert_called_once()
            
            # Check that write was called for each frame
            self.assertEqual(mock_writer.write.call_count, len(test_frames))
            
            # Check that release was called
            mock_writer.release.assert_called_once()


class TestWiFiMonitor(unittest.TestCase):
    """Test cases for WiFiMonitor class"""

    def setUp(self):
        """Set up test environment"""
        self.config = StorageConfig()
        # Use a mock for WiFi module availability
        self.config.wifi_monitoring = True
        self.config.wifi_adapter = "test_wlan0"
        self.config.wifi_throttle_poor = 100
        self.config.wifi_throttle_medium = 300
        self.config.wifi_throttle_good = 800

    @patch('motion_storage.WIFI_AVAILABLE', False)
    def test_initialization_wifi_unavailable(self):
        """Test initialization when WiFi is unavailable"""
        monitor = WiFiMonitor(self.config)
        self.assertFalse(monitor.enabled)
        self.assertEqual(monitor.current_throttle, self.config.upload_throttle_kbps)

    @patch('subprocess.check_output')
    @patch('motion_storage.WIFI_AVAILABLE', True)
    def test_initialization_wifi_available(self, mock_check_output):
        """Test initialization when WiFi module is available"""
        with patch('threading.Thread') as mock_thread:
            monitor = WiFiMonitor(self.config)
            self.assertTrue(monitor.enabled)
            mock_thread.assert_called_once()

    def test_get_current_throttle_disabled(self):
        """Test get_current_throttle when WiFi monitoring is disabled"""
        self.config.wifi_monitoring = False
        monitor = WiFiMonitor(self.config)
        self.assertEqual(monitor.get_current_throttle(), self.config.upload_throttle_kbps)

    @patch('motion_storage.WIFI_AVAILABLE', True)
    def test_get_current_throttle_enabled(self):
        """Test get_current_throttle when WiFi monitoring is enabled"""
        monitor = WiFiMonitor(self.config)
        # Set a throttle value manually
        monitor.current_throttle = 250
        self.assertEqual(monitor.get_current_throttle(), 250)

    @patch('motion_storage.subprocess.check_output')
    def test_get_wifi_signal_strength(self, mock_check_output):
        """Test getting WiFi signal strength"""
        # Mock subprocess output for iwconfig
        mock_check_output.return_value = """
        wlan0     IEEE 802.11  ESSID:"TestNetwork"  
                  Mode:Managed  Frequency:2.462 GHz  Access Point: 00:11:22:33:44:55   
                  Bit Rate=72.2 Mb/s   Tx-Power=20 dBm   
                  Retry min limit:7   RTS thr=2347 B   Fragment thr:off
                  Power Management:on
                  Link Quality=70/70  Signal level=-65 dBm  
                  Rx invalid nwid:0  Rx invalid crypt:0  Rx invalid frag:0
                  Tx excessive retries:0  Invalid misc:0   Missed beacon:0
        """
        
        monitor = WiFiMonitor(self.config)
        # Patch WIFI_MODULE_AVAILABLE
        with patch('motion_storage.WIFI_AVAILABLE', True):
            signal = monitor._get_wifi_signal_strength()
            self.assertEqual(signal, -65)


class TestTransferManager(unittest.TestCase):
    """Test cases for TransferManager class"""

    def setUp(self):
        """Set up test environment"""
        # Create temp directory for test
        self.test_dir = tempfile.mkdtemp()
        
        # Create test config with test directory
        self.config = StorageConfig()
        self.config.local_storage_path = self.test_dir
        self.config.remote_storage_url = "http://test.example.com/storage"
        
        # Create wifi monitor mock
        self.wifi_monitor = MagicMock()
        self.wifi_monitor.get_current_throttle.return_value = 300
        
        # Create transfer manager
        self.manager = TransferManager(self.config, self.wifi_monitor)

    def tearDown(self):
        """Clean up after test"""
        # Remove temp directory
        shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """Test proper initialization"""
        self.assertEqual(self.manager.config, self.config)
        self.assertEqual(self.manager.wifi_monitor, self.wifi_monitor)
        self.assertIsInstance(self.manager.transfer_queue, type(self.manager.transfer_queue))
        self.assertEqual(len(self.manager.pending_events), 0)
        self.assertEqual(len(self.manager.active_transfers), 0)

    def test_add_event(self):
        """Test adding an event to the transfer queue"""
        # Add an event
        self.manager.add_event("test-event-123")
        
        # Event should be in pending_events
        self.assertIn("test-event-123", self.manager.pending_events)
        
        # Event should be in transfer queue
        self.assertFalse(self.manager.transfer_queue.empty())

    def test_add_event_duplicate(self):
        """Test adding a duplicate event"""
        # Add an event twice
        self.manager.add_event("test-event-123")
        queue_size_before = self.manager.transfer_queue.qsize()
        
        # Add same event again
        self.manager.add_event("test-event-123")
        queue_size_after = self.manager.transfer_queue.qsize()
        
        # Queue size should not change
        self.assertEqual(queue_size_before, queue_size_after)

    @patch('os.path.exists')
    def test_transfer_worker_nonexistent_event(self, mock_exists):
        """Test transfer worker with nonexistent event"""
        # Mock os.path.exists to return False
        mock_exists.return_value = False
        
        # Add event to queue manually
        self.manager.pending_events.add("nonexistent-event")
        self.manager.transfer_queue.put((50, "nonexistent-event"))
        
        # Call transfer worker method directly
        with patch('time.sleep'):  # Patch sleep to avoid delays
            self.manager._transfer_worker()
        
        # Event should be removed from pending_events
        self.assertNotIn("nonexistent-event", self.manager.pending_events)

    @patch('os.path.exists')
    @patch('time.sleep')
    def test_transfer_schedule_outside_window(self, mock_sleep, mock_exists):
        """Test transfer scheduling outside of window"""
        # Mock os.path.exists to return True
        mock_exists.return_value = True
        
        # Set transfer schedule active
        self.config.transfer_schedule_active = True
        
        # Set current hour outside of window
        current_hour = 12
        self.config.transfer_schedule_start = 1
        self.config.transfer_schedule_end = 5
        
        # Mock datetime.now to return a specific hour
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.datetime(2023, 1, 1, current_hour)
            
            # Add event
            self.manager.transfer_queue.put((50, "test-event"))
            
            # Call transfer worker method
            self.manager._transfer_worker()
            
            # Event should be rescheduled, not processed
            self.assertFalse(self.manager.transfer_queue.empty())
            # Queue should contain the event with low priority
            _, event_id = self.manager.transfer_queue.get()
            self.assertEqual(event_id, "test-event")

    @patch('motion_storage.TransferManager._upload_event')
    @patch('os.path.exists')
    def test_transfer_success(self, mock_exists, mock_upload):
        """Test successful transfer"""
        # Mock os.path.exists to return True
        mock_exists.return_value = True
        
        # Mock _upload_event to return True (success)
        mock_upload.return_value = True
        
        # Set up for transfer
        event_id = "test-event-123"
        self.manager.pending_events.add(event_id)
        self.manager.transfer_queue.put((50, event_id))
        
        # Mock _cleanup_event
        with patch.object(self.manager, '_cleanup_event') as mock_cleanup:
            # Call transfer worker
            self.manager._transfer_worker()
            
            # Verify _upload_event was called
            mock_upload.assert_called_once_with(event_id)
            
            # Verify cleanup was called
            mock_cleanup.assert_called_once_with(event_id)
            
            # Event should be removed from pending_events
            self.assertNotIn(event_id, self.manager.pending_events)

    @patch('motion_storage.TransferManager._upload_event')
    @patch('os.path.exists')
    @patch('time.sleep')
    def test_transfer_failure(self, mock_sleep, mock_exists, mock_upload):
        """Test failed transfer"""
        # Mock os.path.exists to return True
        mock_exists.return_value = True
        
        # Mock _upload_event to return False (failure)
        mock_upload.return_value = False
        
        # Set up for transfer
        event_id = "test-event-123"
        self.manager.pending_events.add(event_id)
        self.manager.transfer_queue.put((50, event_id))
        
        # Call transfer worker
        self.manager._transfer_worker()
        
        # Verify _upload_event was called
        mock_upload.assert_called_once_with(event_id)
        
        # Event should still be in pending_events
        self.assertIn(event_id, self.manager.pending_events)
        
        # Event should be requeued with higher priority
        self.assertFalse(self.manager.transfer_queue.empty())

    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('os.path.isdir')
    @patch('os.path.getsize')
    @patch('os.path.getctime')
    def test_check_disk_usage_under_limit(self, mock_getctime, mock_getsize, 
                                         mock_isdir, mock_listdir, mock_exists):
        """Test disk usage check when under limit"""
        # Mock os.path.exists to return True
        mock_exists.return_value = True
        
        # Mock os.listdir to return event IDs
        mock_listdir.return_value = ["event1", "event2"]
        
        # Mock os.path.isdir to return True
        mock_isdir.return_value = True
        
        # Mock os.path.getsize to return a small size
        mock_getsize.return_value = 1024 * 1024  # 1MB
        
        # Mock os.path.getctime
        mock_getctime.return_value = time.time()
        
        # Set max disk usage high
        self.config.max_disk_usage_mb = 100  # 100MB
        
        # Add event to pending_events
        self.manager.pending_events.add("event1")
        
        # Call check_disk_usage
        with patch('os.walk') as mock_walk:
            # Mock os.walk to return files
            mock_walk.return_value = [
                (os.path.join(self.test_dir, "event1"), [], ["file1"]),
                (os.path.join(self.test_dir, "event2"), [], ["file2"])
            ]
            
            self.manager.check_disk_usage()
        
        # No events should be removed since we're under the limit
        mock_listdir.assert_called_once_with(self.test_dir)
        # os.walk should be called for each event
        self.assertEqual(mock_walk.call_count, 1)
        # No deletion should occur

    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('os.path.isdir')
    @patch('os.path.getsize')
    @patch('os.path.getctime')
    @patch('shutil.rmtree')
    def test_check_disk_usage_over_limit(self, mock_rmtree, mock_getctime, 
                                       mock_getsize, mock_isdir, mock_listdir, 
                                       mock_exists):
        """Test disk usage check when over limit"""
        # Mock os.path.exists to return True
        mock_exists.return_value = True
        
        # Mock os.listdir to return event IDs
        mock_listdir.return_value = ["event1", "event2"]
        
        # Mock os.path.isdir to return True
        mock_isdir.return_value = True
        
        # Mock os.path.getsize to return a large size
        mock_getsize.return_value = 10 * 1024 * 1024  # 10MB per file
        
        # Mock os.path.getctime to sort events by time
        # event1 is older than event2
        mock_getctime.side_effect = lambda path: time.time() - 100 if "event1" in path else time.time()
        
        # Set max disk usage low to trigger cleanup
        self.config.max_disk_usage_mb = 5  # 5MB
        
        # Call check_disk_usage
        with patch('os.walk') as mock_walk:
            # Mock os.walk to return files
            mock_walk.return_value = [
                (os.path.join(self.test_dir, "event1"), [], ["file1"]),
                (os.path.join(self.test_dir, "event2"), [], ["file2"])
            ]
            
            self.manager.check_disk_usage()
        
        # The oldest event (event1) should be removed
        mock_rmtree.assert_called_once_with(os.path.join(self.test_dir, "event1"))


if __name__ == '__main__':
    unittest.main()
import os
import sys
import unittest
import tempfile
import shutil
import time
import datetime
import json
import base64
import uuid
from unittest.mock import MagicMock, patch, mock_open
from io import BytesIO

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import module to test
import storage_server
from storage_server import app

class TestStorageServer(unittest.TestCase):
    """Test cases for Storage Server endpoints"""

    def setUp(self):
        """Set up test environment"""
        # Create test client
        self.client = TestClient(app)
        
        # Create temp directories for testing
        self.test_storage_base = tempfile.mkdtemp()
        self.test_temp_upload_dir = tempfile.mkdtemp()
        
        # Save original values
        self.original_storage_base = storage_server.STORAGE_BASE
        self.original_temp_upload_dir = storage_server.TEMP_UPLOAD_DIR
        self.original_api_key = storage_server.API_KEY
        
        # Set test values
        storage_server.STORAGE_BASE = self.test_storage_base
        storage_server.TEMP_UPLOAD_DIR = self.test_temp_upload_dir
        storage_server.API_KEY = "test_api_key"

    def tearDown(self):
        """Clean up after test"""
        # Remove temp directories
        shutil.rmtree(self.test_storage_base)
        shutil.rmtree(self.test_temp_upload_dir)
        
        # Restore original values
        storage_server.STORAGE_BASE = self.original_storage_base
        storage_server.TEMP_UPLOAD_DIR = self.original_temp_upload_dir
        storage_server.API_KEY = self.original_api_key

    def test_server_info(self):
        """Test server info endpoint"""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["name"], "Motion Storage Server")
        self.assertEqual(data["storage_path"], self.test_storage_base)

    def test_unauthorized_access(self):
        """Test unauthorized access to protected endpoints"""
        # Try to access storage status without API key
        response = self.client.get("/storage/stats")
        self.assertEqual(response.status_code, 403)
        
        # Try to upload without API key
        response = self.client.post("/storage", files={"video": "dummy"}, data={"metadata": "{}"})
        self.assertEqual(response.status_code, 403)
        
        # Try to initialize chunked upload without API key
        response = self.client.post("/storage/chunked/init", json={"metadata": "{}"})
        self.assertEqual(response.status_code, 403)

    def test_authorized_access(self):
        """Test authorized access to protected endpoints"""
        # Access storage stats with API key
        response = self.client.get("/storage/stats", headers={"X-API-Key": "test_api_key"})
        self.assertEqual(response.status_code, 200)

    def test_chunked_upload_init(self):
        """Test initializing a chunked upload"""
        metadata = {
            "id": "test-event-123",
            "start_time": datetime.datetime.now().isoformat(),
            "duration": 5.0,
            "frame_count": 150
        }
        
        request_data = {
            "metadata": json.dumps(metadata),
            "file_size": 1024 * 1024,  # 1MB
            "chunk_size": 1024 * 256,  # 256KB
            "total_chunks": 4
        }
        
        response = self.client.post(
            "/storage/chunked/init",
            json=request_data,
            headers={"X-API-Key": "test_api_key"}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("upload_id", data)
        
        # Check that upload directory was created
        upload_id = data["upload_id"]
        upload_dir = os.path.join(self.test_temp_upload_dir, upload_id)
        self.assertTrue(os.path.exists(upload_dir))
        
        # Check that metadata file was created
        metadata_file = os.path.join(upload_dir, "metadata.json")
        self.assertTrue(os.path.exists(metadata_file))
        
        # Verify metadata content
        with open(metadata_file, "r") as f:
            saved_metadata = json.load(f)
        
        self.assertEqual(saved_metadata["metadata"]["id"], metadata["id"])
        self.assertEqual(saved_metadata["file_size"], 1024 * 1024)
        self.assertEqual(saved_metadata["total_chunks"], 4)

    def test_chunked_upload_chunk(self):
        """Test uploading a chunk"""
        # First create an upload
        metadata = {
            "id": "test-event-123",
            "start_time": datetime.datetime.now().isoformat()
        }
        
        init_data = {
            "metadata": json.dumps(metadata),
            "file_size": 1024 * 10,  # 10KB
            "chunk_size": 1024,  # 1KB
            "total_chunks": 10
        }
        
        init_response = self.client.post(
            "/storage/chunked/init",
            json=init_data,
            headers={"X-API-Key": "test_api_key"}
        )
        
        upload_id = init_response.json()["upload_id"]
        
        # Create a test chunk
        test_data = b"test chunk data" * 100  # ~1.2KB
        encoded_chunk = base64.b64encode(test_data).decode("utf-8")
        
        # Upload chunk
        chunk_data = {
            "upload_id": upload_id,
            "chunk_index": 0,
            "chunk_data": encoded_chunk
        }
        
        response = self.client.post(
            "/storage/chunked/upload",
            json=chunk_data,
            headers={"X-API-Key": "test_api_key"}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["chunk_index"], 0)
        
        # Check that chunk was saved
        chunks_dir = os.path.join(self.test_temp_upload_dir, upload_id, "chunks")
        self.assertTrue(os.path.exists(chunks_dir))
        chunk_file = os.path.join(chunks_dir, "000000")
        self.assertTrue(os.path.exists(chunk_file))
        
        # Verify chunk content
        with open(chunk_file, "rb") as f:
            saved_chunk = f.read()
        self.assertEqual(saved_chunk, test_data)
        
        # Check that metadata was updated
        metadata_file = os.path.join(self.test_temp_upload_dir, upload_id, "metadata.json")
        with open(metadata_file, "r") as f:
            updated_metadata = json.load(f)
        self.assertEqual(updated_metadata["received_chunks"], 1)

    @patch('os.makedirs')
    def test_chunked_upload_finalize(self, mock_makedirs):
        """Test finalizing a chunked upload"""
        # Mock uuid to get predictable IDs
        with patch('uuid.uuid4') as mock_uuid:
            mock_uuid.return_value = uuid.UUID('12345678123456781234567812345678')
            
            # Create an upload
            upload_id = f"upload_{int(time.time())}_12345678"
            upload_dir = os.path.join(self.test_temp_upload_dir, upload_id)
            os.makedirs(upload_dir, exist_ok=True)
            
            # Create chunks directory and files
            chunks_dir = os.path.join(upload_dir, "chunks")
            os.makedirs(chunks_dir, exist_ok=True)
            
            # Create chunk files
            chunk_data = b"test chunk data"
            for i in range(3):
                chunk_file = os.path.join(chunks_dir, f"{i:06d}")
                with open(chunk_file, "wb") as f:
                    f.write(chunk_data)
            
            # Create metadata file
            metadata = {
                "metadata": {
                    "id": "test-event-123",
                    "start_time": "2023-01-01T12:00:00"
                },
                "file_size": len(chunk_data) * 3,
                "chunk_size": len(chunk_data),
                "total_chunks": 3,
                "received_chunks": 3,
                "start_time": "2023-01-01T12:00:00"
            }
            
            metadata_path = os.path.join(upload_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)
            
            # Call finalize endpoint
            response = self.client.post(
                "/storage/chunked/finalize",
                json={"upload_id": upload_id},
                headers={"X-API-Key": "test_api_key"}
            )
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertTrue(data["success"])
            self.assertEqual(data["event_id"], "test-event-123")
            
            # Check that final directory structure was created
            # 2023/01/01/test-event-123
            final_path = os.path.join(
                self.test_storage_base,
                "2023", "01", "01", "test-event-123"
            )
            mock_makedirs.assert_called_with(final_path, exist_ok=True)

    def test_upload_complete_video(self):
        """Test uploading a complete video file"""
        # Create test video content
        video_content = b"test video data" * 1000
        
        # Create test metadata
        metadata = {
            "id": "test-event-123",
            "start_time": "2023-01-01T12:00:00",
            "end_time": "2023-01-01T12:00:05",
            "duration": 5.0,
            "frame_count": 150
        }
        
        # Mock out file operations
        with patch('builtins.open', mock_open()), \
             patch('os.makedirs'), \
             patch('storage_server.datetime') as mock_datetime:
            
            # Mock datetime.datetime.fromisoformat
            dt = datetime.datetime(2023, 1, 1, 12, 0, 0)
            mock_datetime.datetime.fromisoformat.return_value = dt
            
            # Mock datetime.datetime.now
            now = datetime.datetime(2023, 1, 1, 12, 30, 0)
            mock_datetime.datetime.now.return_value = now
            
            # Upload video
            response = self.client.post(
                "/storage",
                files={
                    "video": ("video.mp4", BytesIO(video_content), "video/mp4")
                },
                data={
                    "metadata": json.dumps(metadata)
                },
                headers={"X-API-Key": "test_api_key"}
            )
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertTrue(data["success"])
            self.assertEqual(data["event_id"], "test-event-123")
            self.assertEqual(data["file_size"], len(video_content))

    def test_get_storage_stats_empty(self):
        """Test getting storage stats with empty storage"""
        response = self.client.get(
            "/storage/stats",
            headers={"X-API-Key": "test_api_key"}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_events"], 0)
        self.assertEqual(data["total_size_mb"], 0)
        self.assertEqual(data["events_by_day"], {})

    @patch('os.listdir')
    @patch('os.path.isdir')
    @patch('os.walk')
    def test_get_storage_stats_with_events(self, mock_walk, mock_isdir, mock_listdir):
        """Test getting storage stats with events"""
        # Mock directory structure
        mock_listdir.return_value = ["2023"]
        mock_isdir.return_value = True
        
        # Mock nested directory structure
        def mock_listdir_side_effect(path):
            if path == os.path.join(self.test_storage_base, "2023"):
                return ["01"]
            elif path == os.path.join(self.test_storage_base, "2023", "01"):
                return ["01"]
            elif path == os.path.join(self.test_storage_base, "2023", "01", "01"):
                return ["event1", "event2"]
            else:
                return []
                
        mock_listdir.side_effect = mock_listdir_side_effect
        
        # Mock os.walk to return files with sizes
        mock_walk.return_value = [
            (os.path.join(self.test_storage_base, "2023/01/01/event1"), 
             [], ["video.mp4", "metadata.json"]),
            (os.path.join(self.test_storage_base, "2023/01/01/event2"), 
             [], ["video.mp4", "metadata.json"])
        ]
        
        # Mock os.path.getsize to return file sizes
        with patch('os.path.getsize') as mock_getsize:
            mock_getsize.return_value = 1024 * 1024  # 1MB per file
            
            response = self.client.get(
                "/storage/stats",
                headers={"X-API-Key": "test_api_key"}
            )
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["total_events"], 2)
            self.assertEqual(data["total_size_mb"], 4.0)  # 4 files * 1MB
            self.assertIn("2023-01-01", data["events_by_day"])
            self.assertEqual(data["events_by_day"]["2023-01-01"]["events"], 2)

    @patch('os.path.exists')
    def test_event_not_found(self, mock_exists):
        """Test getting a nonexistent event"""
        mock_exists.return_value = False
        
        response = self.client.get(
            "/storage/events/nonexistent-event",
            headers={"X-API-Key": "test_api_key"}
        )
        
        self.assertEqual(response.status_code, 404)
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("not found", data["detail"])

    @patch('os.walk')
    def test_list_events(self, mock_walk):
        """Test listing events"""
        # Create a mock directory structure
        base_path = os.path.join(self.test_storage_base)
        mock_walk.return_value = [
            (base_path, ["2023"], []),
            (os.path.join(base_path, "2023"), ["01"], []),
            (os.path.join(base_path, "2023/01"), ["01"], []),
            (os.path.join(base_path, "2023/01/01"), ["event1", "event2"], [])
        ]
        
        # Mock Path.is_dir to return True
        with patch('pathlib.Path.is_dir') as mock_is_dir, \
             patch('os.listdir') as mock_listdir, \
             patch('os.path.exists') as mock_exists:
            
            mock_is_dir.return_value = True
            mock_exists.return_value = True
            
            # Mock listdir to return appropriate values for each path
            def mock_listdir_effect(path):
                if "2023" in str(path) and "01" not in str(path):
                    return ["01"]
                elif "2023/01" in str(path) and "01/01" not in str(path):
                    return ["01"]
                elif "2023/01/01" in str(path):
                    return ["event1", "event2"]
                elif "event1" in str(path) or "event2" in str(path):
                    return ["metadata.json", "video.mp4"]
                else:
                    return []
                    
            mock_listdir.side_effect = mock_listdir_effect
            
            # Mock open to return metadata
            event1_metadata = {
                "id": "event1",
                "start_time": "2023-01-01T10:00:00",
                "duration": 5.0
            }
            
            event2_metadata = {
                "id": "event2",
                "start_time": "2023-01-01T11:00:00",
                "duration": 10.0
            }
            
            def mock_open_effect(*args, **kwargs):
                if "event1" in args[0]:
                    return mock_open(read_data=json.dumps(event1_metadata))()
                elif "event2" in args[0]:
                    return mock_open(read_data=json.dumps(event2_metadata))()
                else:
                    return mock_open()()
                    
            with patch('builtins.open', mock_open_effect):
                # Test listing all events
                response = self.client.get(
                    "/storage/events",
                    headers={"X-API-Key": "test_api_key"}
                )
                
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertEqual(len(data["events"]), 2)
                
                # Test with limit and offset
                response = self.client.get(
                    "/storage/events?limit=1&offset=1",
                    headers={"X-API-Key": "test_api_key"}
                )
                
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertEqual(len(data["events"]), 1)
                self.assertEqual(data["limit"], 1)
                self.assertEqual(data["offset"], 1)

    @patch('os.path.exists')
    @patch('shutil.rmtree')
    def test_delete_event(self, mock_rmtree, mock_exists):
        """Test deleting an event"""
        # Mock os.path.exists to return True for our event
        mock_exists.return_value = True
        
        # Mock os.walk to find our event
        with patch('os.walk') as mock_walk:
            mock_walk.return_value = [
                (self.test_storage_base, [], []),
                (os.path.join(self.test_storage_base, "test-event"), [], ["video.mp4"])
            ]
            
            response = self.client.delete(
                "/storage/events/test-event",
                headers={"X-API-Key": "test_api_key"}
            )
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertTrue(data["success"])
            
            # Check that rmtree was called to delete the event
            mock_rmtree.assert_called_once()


if __name__ == '__main__':
    unittest.main()
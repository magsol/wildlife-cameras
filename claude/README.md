# Raspberry Pi Camera Motion Storage System

This system provides a complete solution for motion-triggered video recording, local storage management, and efficient network transfer for Raspberry Pi cameras.

## Overview

The system consists of three main components:

1. **MJPEG Streaming Server** (`fastapi_mjpeg_server_with_storage.py`): The main server that handles camera interaction, motion detection, and video streaming with a web interface.

2. **Motion Storage Module** (`motion_storage.py`): A module that manages efficient video recording, storage, and network transfers.

3. **Remote Storage Server** (`storage_server.py`): A server that receives and stores videos from the camera for permanent storage.

## Features

- **Real-time video streaming** with configurable resolution and frame rate
- **Motion detection** with visual highlighting of detected regions
- **Timestamp overlay** with configurable position and appearance
- **RAM buffering** to capture motion events with pre-motion footage
- **Efficient disk usage** with automatic cleanup of oldest events
- **WiFi signal monitoring** for adaptive upload throttling (uses iwconfig)
- **Chunked uploads** for reliable transfer of large files
- **Thumbnail generation** for quick preview of events
- **Web interface** for monitoring and configuration
- **REST API** for programmatic control and integration

## Installation

### Prerequisites

- Raspberry Pi with camera module
- Raspberry Pi OS with picamera2 library
- Python 3.6 or newer
- Network connection (WiFi or Ethernet)
- Optional: Separate storage server (can be any computer on the network)

### Required Packages

Install the necessary Python packages on both the Raspberry Pi and storage server:

```bash
pip install fastapi uvicorn opencv-python numpy pydantic
```

## Setup

### 1. Camera Server (Raspberry Pi)

Copy these files to your Raspberry Pi:
- `fastapi_mjpeg_server_with_storage.py` (Main server)
- `motion_storage.py` (Storage module)

Create a directory for local storage:

```bash
mkdir -p /home/pi/motion_storage
```

### 2. Remote Storage Server

Copy this file to your storage server:
- `storage_server.py`

Create a directory for permanent storage:

```bash
mkdir -p /path/to/permanent/storage
```

Set up environment variables for the storage server:

```bash
export STORAGE_BASE="/path/to/permanent/storage"
export TEMP_UPLOAD_DIR="/tmp/motion_uploads"
export API_KEY="your_secret_api_key_here"
```

## Usage

### Starting the Camera Server

Run the server on your Raspberry Pi:

```bash
python fastapi_mjpeg_server_with_storage.py --storage-path /home/pi/motion_storage --remote-url http://storage-server-ip:8080/storage --api-key your_secret_api_key_here
```

### Starting the Remote Storage Server

Run the server on your storage computer:

```bash
python storage_server.py
```

The storage server will start on port 8080 by default.

### Accessing the Camera Stream

Access the camera web interface by navigating to `http://raspberry-pi-ip:8000/` in your browser.

## Configuration Options

### Camera Server

```
usage: fastapi_mjpeg_server_with_storage.py [-h] [--width WIDTH] [--height HEIGHT] [--fps FPS]
                                          [--rotation {0,90,180,270}] [--host HOST] [--port PORT]
                                          [--max-clients MAX_CLIENTS] [--client-timeout CLIENT_TIMEOUT]
                                          [--no-timestamp] 
                                          [--timestamp-position {top-left,top-right,bottom-left,bottom-right}]
                                          [--no-motion] [--motion-threshold MOTION_THRESHOLD]
                                          [--motion-min-area MOTION_MIN_AREA] [--no-highlight]
                                          [--storage-path STORAGE_PATH] [--max-storage MAX_STORAGE]
                                          [--remote-url REMOTE_URL] [--api-key API_KEY] [--no-upload]
                                          [--upload-throttle UPLOAD_THROTTLE] [--no-wifi-monitoring]
                                          [--no-thumbnails]
```

#### Camera Settings
- `--width WIDTH`: Camera width in pixels (default: 640)
- `--height HEIGHT`: Camera height in pixels (default: 480)
- `--fps FPS`: Camera frame rate (default: 30)
- `--rotation {0,90,180,270}`: Camera rotation in degrees (default: 0)

#### Server Settings
- `--host HOST`: Server host address (default: 0.0.0.0)
- `--port PORT`: Server port (default: 8000)
- `--max-clients MAX_CLIENTS`: Maximum number of clients (default: 10)
- `--client-timeout CLIENT_TIMEOUT`: Client timeout in seconds (default: 30)

#### Timestamp Settings
- `--no-timestamp`: Disable timestamp display
- `--timestamp-position {top-left,top-right,bottom-left,bottom-right}`: Timestamp position (default: bottom-right)

#### Motion Detection Settings
- `--no-motion`: Disable motion detection
- `--motion-threshold MOTION_THRESHOLD`: Motion detection threshold (5-100, lower is more sensitive, default: 25)
- `--motion-min-area MOTION_MIN_AREA`: Minimum pixel area to consider as motion (default: 500)
- `--no-highlight`: Disable motion highlighting in video

#### Storage Settings
- `--storage-path STORAGE_PATH`: Path to store motion events (default: /tmp/motion_events)
- `--max-storage MAX_STORAGE`: Maximum storage usage in MB (default: 1000)
- `--remote-url REMOTE_URL`: URL of remote storage server
- `--api-key API_KEY`: API key for remote storage server
- `--no-upload`: Disable uploading to remote server
- `--upload-throttle UPLOAD_THROTTLE`: Upload throttle in KB/s (default: 500)
- `--no-wifi-monitoring`: Disable WiFi signal monitoring
- `--no-thumbnails`: Disable thumbnail generation

### Example Configurations

#### High-Quality Surveillance Camera

```bash
python fastapi_mjpeg_server_with_storage.py \
  --width 1280 --height 720 --fps 30 \
  --motion-threshold 20 --motion-min-area 400 \
  --storage-path /home/pi/surveillance_footage \
  --max-storage 5000 \
  --upload-throttle 1000
```

#### Low-Power Outdoor Camera

```bash
python fastapi_mjpeg_server_with_storage.py \
  --width 640 --height 480 --fps 15 \
  --motion-threshold 30 --motion-min-area 600 \
  --storage-path /home/pi/outdoor_footage \
  --max-storage 2000 \
  --upload-throttle 200
```

#### Night Vision Camera

```bash
python fastapi_mjpeg_server_with_storage.py \
  --width 800 --height 600 --fps 10 \
  --motion-threshold 15 --motion-min-area 300 \
  --timestamp-position top-left \
  --storage-path /home/pi/night_footage
```

## Web Interface

The web interface provides:

1. **Live Video Stream**: Real-time view from the camera
2. **Motion Detection**: Visual highlighting of detected motion
3. **Configuration Panel**: Change settings on-the-fly
4. **Motion History**: List of recent motion events
5. **Storage Statistics**: Monitor storage usage and transfer status

### Configuration Panel

Click "Show Configuration" to access settings:
- Toggle timestamp display
- Toggle motion detection
- Adjust motion sensitivity
- Change timestamp position

### Motion History

Click "Show Motion History" to view recent motion events with timestamps.

### Storage Stats

Click "Show Storage Stats" to view:
- Local storage usage
- Pending transfers
- WiFi signal strength (if enabled)
- Current upload throttle

## API Endpoints

### Camera Server Endpoints

- `GET /`: Web interface
- `GET /stream`: MJPEG video stream
- `GET /status`: Server and camera status
- `GET /motion_status`: Motion detection status and history
- `GET /storage/status`: Storage statistics and pending transfers
- `POST /config`: Update camera configuration
- `POST /storage/config`: Update storage configuration
- `POST /storage/transfer/{event_id}`: Force transfer of a specific event
- `GET /storage/events`: List all motion events

### Storage Server Endpoints

- `POST /storage`: Upload a complete video
- `POST /storage/chunked/init`: Initialize a chunked upload
- `POST /storage/chunked/upload`: Upload a chunk of data
- `POST /storage/chunked/finalize`: Finalize a chunked upload
- `GET /storage/stats`: Get storage statistics
- `GET /storage/events`: List all stored events
- `GET /storage/events/{event_id}`: Get details of a specific event
- `GET /storage/events/{event_id}/video`: Download video for a specific event
- `DELETE /storage/events/{event_id}`: Delete a specific event

## Running as Services

### Camera Server Service

Create a systemd service file:

```bash
sudo nano /etc/systemd/system/camera-server.service
```

Add the following content:

```
[Unit]
Description=Raspberry Pi Camera Server with Motion Storage
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi
ExecStart=/usr/bin/python3 /home/pi/fastapi_mjpeg_server_with_storage.py --storage-path /home/pi/motion_storage --remote-url http://storage-server-ip:8080/storage --api-key your_secret_api_key_here
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl enable camera-server
sudo systemctl start camera-server
```

### Storage Server Service

Create a systemd service file:

```bash
sudo nano /etc/systemd/system/storage-server.service
```

Add the following content:

```
[Unit]
Description=Motion Storage Server
After=network.target

[Service]
User=your_user
WorkingDirectory=/path/to/storage_server
Environment="STORAGE_BASE=/path/to/permanent/storage"
Environment="TEMP_UPLOAD_DIR=/tmp/motion_uploads"
Environment="API_KEY=your_secret_api_key_here"
ExecStart=/usr/bin/python3 /path/to/storage_server.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl enable storage-server
sudo systemctl start storage-server
```

## Troubleshooting

### Camera Server Issues

1. **Camera Not Initializing**
   - Check camera connection
   - Verify camera is enabled: `sudo raspi-config`
   - Check for camera errors in logs: `sudo journalctl -u camera-server`

2. **High CPU Usage**
   - Lower resolution and frame rate
   - Increase motion threshold to reduce false positives
   - Disable thumbnail generation with `--no-thumbnails`

3. **Storage Full**
   - Increase max storage with `--max-storage`
   - Verify remote server is accepting uploads
   - Check upload throttle settings

### Storage Server Issues

1. **Upload Errors**
   - Verify network connectivity between camera and server
   - Check API key configuration
   - Check disk space on storage server
   - Verify file permissions on storage directories

2. **Chunked Upload Failures**
   - Check for timeout settings in web server configuration
   - Increase chunk size for better performance
   - Decrease chunk size if experiencing timeouts

## Advanced Usage

### WiFi Signal Monitoring

When enabled, the system will monitor WiFi signal strength and adjust upload throttling accordingly:

- **Good Signal** (> -65 dBm): Uses maximum configured throttle
- **Medium Signal** (-75 to -65 dBm): Uses medium throttle
- **Poor Signal** (< -75 dBm): Uses minimum throttle

### Scheduled Transfers

Configure transfer schedule to only upload videos during specific hours:

```bash
python fastapi_mjpeg_server_with_storage.py --transfer-schedule-active --transfer-schedule-start 1 --transfer-schedule-end 5
```

This will only upload videos between 1 AM and 5 AM to avoid network congestion during the day.

## License

This project is open source and available under the MIT License.
# Wildlife Camera Framework - Quick Start Guide

**Get your wildlife camera running in under 30 minutes!**

This guide covers minimal setup to get started quickly, plus comprehensive deployment instructions for production use.

---

## 🚀 Ultra-Quick Start (15 Minutes)

**Prerequisites**: Raspberry Pi 4B with camera connected, Raspberry Pi OS installed, SSH enabled.

### 1. Connect & Update
```bash
ssh pi@<your-pi-ip>
sudo apt update && sudo apt upgrade -y
```

### 2. Enable Camera
```bash
sudo raspi-config
# Navigate to: Interface Options → Camera → Enable
sudo reboot
```

### 3. Install Dependencies
```bash
sudo apt install -y python3-picamera2 python3-opencv python3-numpy python3-pip git
```

### 4. Get the Code
```bash
cd ~
git clone <your-repository-url> wildlife-camera
cd wildlife-camera
```

Or transfer files manually:
```bash
# On your computer:
scp -r /path/to/wildlife-cameras/claude pi@<pi-ip>:~/wildlife-camera
```

### 5. Install Python Dependencies
```bash
cd ~/wildlife-camera
pip3 install fastapi uvicorn pydantic pyyaml requests
```

### 6. Create Configuration
```bash
cp config.yaml config.yaml.example  # Save template
nano config.yaml
```

Minimal configuration (or use defaults):
```yaml
storage:
  local_storage_path: ./motion_events
  upload_throttle_kbps: 0  # Disable remote upload for now

server:
  host: 0.0.0.0
  port: 8000
```

### 7. Run It!
```bash
python3 fastapi_mjpeg_server_with_storage.py
```

### 8. Access Web Interface
Open browser to: `http://<pi-ip>:8000`

**That's it!** Wave your hand in front of the camera to trigger motion detection.

---

## 📋 Table of Contents

- [Ultra-Quick Start](#-ultra-quick-start-15-minutes) (above)
- [Hardware Requirements](#-hardware-requirements)
- [Complete Installation](#-complete-installation)
- [Configuration Guide](#-configuration-guide)
- [Running as a Service](#-running-as-a-service-auto-start)
- [Remote Storage Setup](#-remote-storage-setup)
- [Performance Tuning](#-performance-tuning)
- [Troubleshooting](#-troubleshooting)
- [Next Steps](#-next-steps)

---

## 🛠️ Hardware Requirements

### Required
- **Raspberry Pi 4B** (2GB RAM minimum, 4GB recommended)
- **Camera Module**: Pi Camera Module 3 NoIR or Camera Module 2 NoIR
- **MicroSD Card**: 32GB+ (64GB recommended for local storage)
- **Power Supply**: Official 5V 3A USB-C adapter
- **Network**: WiFi or Ethernet connection

### Optional
- **External Storage**: USB drive for motion events
- **Case**: With camera mount (improves stability)
- **Cooling**: Heatsinks or fan for 24/7 operation
- **IR Illuminator**: For night vision (NoIR cameras only)

### Network Storage Server (Optional)
- Any computer on your network to receive uploaded videos
- Could be: another Pi, NAS, Linux server, or Windows PC

---

## 📦 Complete Installation

### Step 1: Prepare Raspberry Pi

#### 1.1 Flash Raspberry Pi OS
- Download **Raspberry Pi OS (64-bit)** from [raspberrypi.com/software](https://www.raspberrypi.com/software/)
- Flash to microSD using Raspberry Pi Imager
- **During setup**:
  - Enable SSH
  - Set hostname (e.g., "wildlife-cam")
  - Configure WiFi credentials
  - Set username/password

#### 1.2 First Boot
```bash
ssh pi@wildlife-cam.local
# Or: ssh pi@<ip-address>

# Update system
sudo apt update && sudo apt upgrade -y

# Enable camera interface
sudo raspi-config
# Interface Options → Camera → Enable
# Finish → Reboot
```

### Step 2: Install System Packages

```bash
# Core packages
sudo apt install -y \
    python3-picamera2 \
    python3-opencv \
    python3-numpy \
    python3-pip \
    git \
    libopencv-dev \
    libatlas-base-dev

# Optional: Pixi package manager (recommended)
curl -fsSL https://pixi.sh/install.sh | bash
source ~/.bashrc
pixi --version
```

### Step 3: Deploy Application Code

```bash
# Clone repository
cd ~
git clone <your-repository-url> wildlife-camera
cd wildlife-camera

# Or transfer via SCP:
# scp -r /local/path/wildlife-cameras/claude pi@<pi-ip>:~/wildlife-camera
```

### Step 4: Install Python Dependencies

**Option A: Using Pixi (Recommended)**
```bash
cd ~/wildlife-camera
pixi install  # Reads pixi.toml and installs all dependencies
```

**Note**: Pixi is configured to automatically access system site-packages for `picamera2`. The `pixi-activation.sh` script adds `/usr/lib/python3/dist-packages` to `PYTHONPATH` so the pixi environment can use the system-installed picamera2 library.

**Option B: Using pip**
```bash
cd ~/wildlife-camera
pip3 install fastapi uvicorn pydantic pyyaml requests opencv-python numpy
```

### Step 5: Test Camera Connection

```bash
# Test with libcamera
libcamera-hello --timeout 2000

# Test with Python
python3 -c "from picamera2 import Picamera2; cam = Picamera2(); cam.start(); print('Camera working!'); cam.stop()"
```

You should see a preview window or "Camera working!" message.

---

## ⚙️ Configuration Guide

The system uses a centralized configuration file: `config.yaml`

### Generate Default Configuration

```bash
cd ~/wildlife-camera

# Generate with current defaults
python3 -c "from config import generate_default_config; generate_default_config('config.yaml')"
```

### Edit Configuration

```bash
nano config.yaml
```

### Configuration Structure

```yaml
# Camera hardware settings
camera:
  width: 640                    # Resolution width
  height: 480                   # Resolution height
  frame_rate: 30                # FPS (reduce to 15 for lower CPU usage)
  rotation: 0                   # 0, 90, 180, or 270
  max_clients: 10               # Max simultaneous web viewers
  show_timestamp: true          # Overlay timestamp on video
  timestamp_position: bottom-right

# Motion detection algorithm
motion_detection:
  enabled: true
  threshold: 25                 # Lower = more sensitive (5-100)
  min_area: 500                 # Minimum pixels to trigger
  blur_kernel_size: 21
  highlight_motion: true        # Draw boxes around motion
  history_size: 50              # Number of events to keep in memory

# Optical flow motion classification
optical_flow:
  enabled: true                 # Set to false to disable classification
  frame_skip: 2                 # Process every Nth frame (higher = faster)
  max_resolution: [320, 240]    # Downscale for processing (faster)
  feature_max: 100              # Number of tracking points
  min_distance: 7
  quality_level: 0.3
  grid_size: [8, 8]
  direction_bins: 8
  visualization: false          # CPU-expensive, leave disabled

# Local and remote storage
storage:
  # RAM buffering
  ram_buffer_seconds: 30        # Pre-motion buffer duration
  max_ram_segments: 300

  # Local disk storage
  local_storage_path: ./motion_events
  max_disk_usage_mb: 1000       # 1GB (adjust based on SD card size)
  min_motion_duration_sec: 3    # Minimum event duration to save

  # Remote storage (optional)
  remote_storage_url: http://192.168.1.100:8080/storage
  remote_api_key: your_api_key_here
  upload_throttle_kbps: 500     # 0 = disabled, otherwise KB/s limit

  # Transfer scheduling (upload only during specific hours)
  transfer_schedule_active: false
  transfer_schedule_start: 1    # 1 AM
  transfer_schedule_end: 5      # 5 AM

  # Thumbnails
  generate_thumbnails: true
  thumbnail_width: 320
  thumbnail_height: 240
  thumbnails_per_event: 3

  # WiFi monitoring (adaptive throttling)
  wifi_monitoring: true
  wifi_adapter: wlan0

# Optical flow pattern database
optical_flow_storage:
  store_data: true
  signature_dir: flow_signatures
  database_path: motion_patterns.db
  classification_enabled: true
  min_classification_confidence: 0.5
  save_visualizations: false

# Web server settings
server:
  host: 0.0.0.0                 # Listen on all interfaces
  port: 8000
  log_level: info               # info, debug, warning, error
```

### Configuration Priority

Settings are loaded in this order (later overrides earlier):
1. **Defaults** (in code)
2. **Config file** (`config.yaml`)
3. **Environment variables** (prefix: `WC_`)
4. **Command-line arguments** (highest priority)

### Environment Variable Examples

```bash
export WC_CAMERA_WIDTH=1920
export WC_CAMERA_HEIGHT=1080
export WC_MOTION_THRESHOLD=20
export WC_STORAGE_PATH=/mnt/usb/motion_events
export WC_UPLOAD_THROTTLE=1000
```

### Command-Line Examples

```bash
# Override specific settings
python3 fastapi_mjpeg_server_with_storage.py \
  --width 1280 --height 720 --fps 15 \
  --motion-threshold 20 \
  --storage-path /mnt/usb/motion_events \
  --upload-throttle 1000

# Disable features
python3 fastapi_mjpeg_server_with_storage.py \
  --no-upload --no-wifi-monitoring --no-thumbnails

# Use config file (recommended)
python3 fastapi_mjpeg_server_with_storage.py
# Automatically reads config.yaml
```

### Recommended Settings by Use Case

#### Wildlife Camera (Battery-Powered)
```yaml
camera:
  frame_rate: 10
optical_flow:
  frame_skip: 4
  feature_max: 50
storage:
  upload_throttle_kbps: 200
  transfer_schedule_active: true  # Upload at night only
```

#### High-Quality Surveillance
```yaml
camera:
  width: 1920
  height: 1080
  frame_rate: 30
storage:
  max_disk_usage_mb: 5000
  upload_throttle_kbps: 1000
```

#### Low-Resource Pi Zero 2W
```yaml
camera:
  width: 640
  height: 480
  frame_rate: 10
optical_flow:
  enabled: false  # Disable for CPU savings
storage:
  generate_thumbnails: false
```

---

## 🔄 Running as a Service (Auto-Start)

Set up the camera to start automatically on boot.

### Create Systemd Service

```bash
sudo nano /etc/systemd/system/wildlife-camera.service
```

Paste this configuration:

```ini
[Unit]
Description=Wildlife Camera Motion Detection System
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/wildlife-camera
Environment="PATH=/home/pi/.local/bin:/usr/local/bin:/usr/bin:/bin"

# If using Pixi:
ExecStart=/home/pi/.pixi/bin/pixi run start

# Or if using system Python:
# ExecStart=/usr/bin/python3 /home/pi/wildlife-camera/fastapi_mjpeg_server_with_storage.py

Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### Enable and Start Service

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service (start on boot)
sudo systemctl enable wildlife-camera

# Start service now
sudo systemctl start wildlife-camera

# Check status
sudo systemctl status wildlife-camera
```

You should see: `Active: active (running)` in green.

### Service Management Commands

```bash
# View live logs
sudo journalctl -u wildlife-camera -f

# View recent logs
sudo journalctl -u wildlife-camera -n 100

# Restart service
sudo systemctl restart wildlife-camera

# Stop service
sudo systemctl stop wildlife-camera

# Disable auto-start
sudo systemctl disable wildlife-camera
```

---

## 🌐 Remote Storage Setup

Store motion events on a separate server for redundancy.

### On Storage Server (Any Computer)

#### 1. Install Dependencies
```bash
# On Ubuntu/Debian:
sudo apt install python3-pip
pip3 install fastapi uvicorn pyyaml

# On Windows:
# Install Python from python.org, then:
pip install fastapi uvicorn pyyaml
```

#### 2. Get Storage Server Code
```bash
# Transfer storage_server.py from repository
scp pi@wildlife-cam:~/wildlife-camera/storage_server.py .
scp pi@wildlife-cam:~/wildlife-camera/config.py .
scp pi@wildlife-cam:~/wildlife-camera/cli.py .
```

#### 3. Create Configuration
```bash
nano storage-config.yaml
```

```yaml
storage_server:
  host: 0.0.0.0
  port: 8080
  storage_path: /path/to/permanent/storage
  max_storage_gb: 50
  require_api_key: true
  api_keys:
    - "your_secret_api_key_here"
  enable_chunked_uploads: true
  chunk_size_mb: 5
```

#### 4. Run Storage Server
```bash
# Foreground (for testing)
python3 storage_server.py --config storage-config.yaml

# Background (production)
nohup python3 storage_server.py --config storage-config.yaml > storage.log 2>&1 &

# Or create systemd service (recommended)
```

### On Camera (Raspberry Pi)

Edit `config.yaml`:
```yaml
storage:
  remote_storage_url: http://192.168.1.100:8080/storage
  remote_api_key: your_secret_api_key_here
  upload_throttle_kbps: 500  # Set to 0 to disable uploads
```

Restart camera service:
```bash
sudo systemctl restart wildlife-camera
```

### Verify Connection

Check camera logs for upload attempts:
```bash
sudo journalctl -u wildlife-camera | grep -i upload
```

Check storage server logs:
```bash
tail -f storage.log  # Or journalctl if using systemd
```

---

## ⚡ Performance Tuning

### Monitor System Resources

```bash
# Install monitoring tools
sudo apt install htop -y

# CPU, RAM, processes
htop

# CPU temperature
vcgencmd measure_temp

# Disk space
df -h

# Network usage
sudo apt install nethogs -y
sudo nethogs
```

### Optimize for Performance

If experiencing high CPU (>80%), high temperature (>70°C), or lag:

#### Reduce Frame Rate
```yaml
camera:
  frame_rate: 15  # Or 10 for low-power mode
```

#### Increase Frame Skip
```yaml
optical_flow:
  frame_skip: 4  # Process every 4th frame instead of every 2nd
```

#### Lower Resolution for Optical Flow
```yaml
optical_flow:
  max_resolution: [240, 180]  # Smaller = faster
  feature_max: 50             # Fewer features = faster
```

#### Disable Expensive Features
```yaml
optical_flow:
  visualization: false  # Should already be disabled
storage:
  generate_thumbnails: false  # Save CPU
```

#### External Storage
Move motion events to USB drive:
```bash
# Plug in USB drive
lsblk  # Find device (e.g., /dev/sda1)

# Mount
sudo mkdir -p /mnt/usb-storage
sudo mount /dev/sda1 /mnt/usb-storage

# Auto-mount on boot
sudo nano /etc/fstab
# Add line (replace UUID):
# UUID=XXXX-XXXX /mnt/usb-storage auto defaults,nofail 0 2

# Update config
nano config.yaml
# storage:
#   local_storage_path: /mnt/usb-storage/motion_events
```

### Cooling Solutions

If temperature >75°C consistently:
- Add heatsinks to CPU and power management chip
- Add 5V fan (connect to GPIO pins 4 & 6)
- Ensure good ventilation around Pi
- Consider official Pi case with built-in fan

---

## 🔧 Troubleshooting

### Camera Not Detected

```bash
# Check camera status
vcgencmd get_camera
# Should show: supported=1 detected=1

# Test camera
libcamera-hello --timeout 2000
```

**Solutions**:
- Verify camera interface enabled: `sudo raspi-config`
- Check ribbon cable connections (both ends)
- Ensure blue side of cable faces correct direction
- Try a different ribbon cable
- Check for camera LED (if present) when system starts

### Import Errors

```bash
# picamera2 not found
sudo apt install --reinstall python3-picamera2

# Other packages
pip3 install fastapi uvicorn pydantic pyyaml requests
```

### Service Won't Start

```bash
# Check logs for errors
sudo journalctl -u wildlife-camera -n 50

# Common issues:
# 1. Wrong paths in service file
# 2. Port 8000 already in use
# 3. Permission denied on storage directory
# 4. Missing Python packages
```

**Solutions**:
```bash
# Check if port in use
sudo netstat -tlnp | grep 8000

# Fix permissions
sudo chown -R pi:pi /home/pi/wildlife-camera
chmod 755 /home/pi/wildlife-camera

# Test manually
cd ~/wildlife-camera
python3 fastapi_mjpeg_server_with_storage.py
# Check error messages
```

### High CPU Usage

Monitor with `htop`, then:
- Reduce `frame_rate` to 10 or 15
- Increase `frame_skip` to 3 or 4
- Reduce `feature_max` to 50
- Lower `max_resolution` to [240, 180]
- Disable `optical_flow` entirely if not needed

### Motion Events Not Saving

```bash
# Check directory permissions
ls -ld motion_events/
# Should be owned by pi user

# Check disk space
df -h
# Ensure space available

# Check logs
sudo journalctl -u wildlife-camera | grep -i error

# Manually test write
touch motion_events/test.txt
# Should succeed without errors
```

### Web Interface Not Accessible

```bash
# Verify service running
sudo systemctl status wildlife-camera

# Check if listening on port 8000
sudo netstat -tlnp | grep 8000

# Check firewall (usually not enabled)
sudo ufw status

# Try local access first
curl http://localhost:8000
# Should return HTML

# Check network connectivity
ping <pi-ip-address>
```

### Pattern Database Errors

```bash
# Check database file
ls -lh motion_patterns.db

# If corrupted, backup and recreate
mv motion_patterns.db motion_patterns.db.backup
sudo systemctl restart wildlife-camera
# New database will be created

# To recover patterns (advanced):
sqlite3 motion_patterns.db.backup .dump > backup.sql
sqlite3 motion_patterns.db < backup.sql
```

### WiFi Monitoring Errors

```bash
# Check iwconfig available
which iwconfig
# If not found:
sudo apt install wireless-tools

# Test WiFi monitoring
iwconfig wlan0
# Should show signal strength

# Disable if problematic
nano config.yaml
# storage:
#   wifi_monitoring: false
```

---

## 📊 Next Steps

### After Initial Setup

1. **Let it run for 24-48 hours** to collect motion events
2. **Access web interface**: `http://<pi-ip>:8000`
3. **Check "Show Motion Patterns"** to see optical flow classifications
4. **Review and correct** misclassifications using Relabel button
5. **Monitor performance**:
   ```bash
   sudo journalctl -u wildlife-camera -f
   htop
   vcgencmd measure_temp
   ```

### Configure Remote Storage (Optional)

- Set up storage server on network
- Update `config.yaml` with server URL and API key
- Test uploads and verify files appear on storage server

### Set Up Automatic Cleanup

```bash
# Create cleanup script
nano ~/cleanup_old_events.sh
```

```bash
#!/bin/bash
# Delete events older than 30 days
find ~/wildlife-camera/motion_events -type d -mtime +30 -exec rm -rf {} +
echo "Cleanup completed: $(date)"
```

```bash
chmod +x ~/cleanup_old_events.sh

# Add to crontab (3 AM daily)
crontab -e
# Add line:
# 0 3 * * * ~/cleanup_old_events.sh >> ~/cleanup.log 2>&1
```

### Advanced Configuration

- Set up **S3-compatible storage** (coming soon - see ROADMAP.md)
- Configure **motion zones** for specific areas (future feature)
- Set up **multi-camera coordination** (future feature)
- Integrate with **Home Assistant** or other smart home systems

### Learning & Improvement

- The system collects motion patterns over time
- Optical flow classifier improves with more labeled data
- Use web UI to correct misclassifications
- Patterns are stored in SQLite database (`motion_patterns.db`)

---

## 📚 Additional Resources

- **Full Documentation**: See `documentation/` folder
- **API Reference**: Check README.md for all endpoints
- **Roadmap**: See `documentation/ROADMAP.md` for future features
- **Configuration Details**: See `documentation/CONFIGURATION_MIGRATION_COMPLETE.md`
- **Deployment Guide**: See `documentation/RASPBERRY_PI_DEPLOYMENT.md`

---

## 🎉 Quick Reference Commands

Save to `~/wildlife-camera-commands.txt`:

```bash
cat > ~/wildlife-camera-commands.txt << 'EOF'
# Wildlife Camera Quick Reference

# Service Management
sudo systemctl status wildlife-camera    # Check status
sudo systemctl restart wildlife-camera   # Restart
sudo journalctl -u wildlife-camera -f   # Live logs

# System Monitoring
htop                                     # CPU/RAM
vcgencmd measure_temp                   # Temperature
df -h                                    # Disk space
sudo netstat -tlnp | grep 8000          # Port check

# Camera Testing
libcamera-hello --timeout 2000          # Test camera
vcgencmd get_camera                     # Camera status

# Motion Events
ls -lt motion_events/ | head            # Recent events
du -sh motion_events/                   # Storage used

# Configuration
nano ~/wildlife-camera/config.yaml      # Edit config
sudo systemctl restart wildlife-camera  # Apply changes

# Web Interface
http://<pi-ip>:8000                     # Access UI

# Cleanup
find ~/wildlife-camera/motion_events -mtime +30 -exec rm -rf {} +
EOF
```

---

## 💡 Tips for Success

1. **Start Simple**: Use default settings first, optimize later
2. **Monitor Initially**: Watch logs for first few hours
3. **Test Motion Detection**: Wave hand to trigger, verify events saved
4. **Check Temperature**: Ensure Pi stays cool (<70°C)
5. **Set Static IP**: Makes access more reliable
6. **Regular Backups**: Copy motion_patterns.db periodically
7. **Update Software**: Keep system and packages updated
8. **Document Changes**: Note any config modifications

---

**Congratulations! Your wildlife camera is now operational!**

For issues or questions, check logs first:
```bash
sudo journalctl -u wildlife-camera -n 100
```

Then review the Troubleshooting section above.

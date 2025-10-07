# Raspberry Pi 4B Deployment Guide

Complete step-by-step instructions for deploying the wildlife camera system on a Raspberry Pi 4B with NoIR camera module.

## Prerequisites

### Hardware Required
- Raspberry Pi 4B (2GB+ RAM recommended, 4GB ideal)
- Raspberry Pi Camera Module 3 NoIR (or Camera Module 2 NoIR)
- MicroSD card (32GB+ recommended)
- Power supply (official 5V 3A USB-C recommended)
- Camera ribbon cable (included with camera)
- Optional: Case with camera mount
- Optional: External storage (USB drive) for motion events

### Software Required
- Raspberry Pi OS Bullseye (64-bit recommended)
- Fresh base installation with SSH enabled

---

## Step 1: Initial Raspberry Pi Setup

### 1.1 Connect to Your Pi

SSH into your Raspberry Pi:
```bash
ssh pi@<your-pi-ip-address>
# Default password is usually "raspberry"
```

### 1.2 Update System

```bash
sudo apt update && sudo apt upgrade -y
```

### 1.3 Enable Camera Interface

```bash
sudo raspi-config
```

Navigate to:
- **Interface Options** → **Camera** → **Enable**
- Reboot when prompted: `sudo reboot`

---

## Step 2: Install System Dependencies

### 2.1 Install Required Packages

```bash
sudo apt install -y \
    python3-picamera2 \
    python3-opencv \
    python3-numpy \
    python3-pip \
    git \
    libopencv-dev \
    libatlas-base-dev \
    libjasper-dev \
    libqt5gui5 \
    libqt5test5 \
    libhdf5-dev \
    python3-dev
```

**Note:** This may take 10-15 minutes.

### 2.2 Install Pixi (Package Manager)

```bash
curl -fsSL https://pixi.sh/install.sh | bash
source ~/.bashrc
```

Verify installation:
```bash
pixi --version
```

---

## Step 3: Deploy the Wildlife Camera Code

### 3.1 Clone or Transfer Repository

**Option A: If using Git:**
```bash
cd ~
git clone <your-repository-url> wildlife-camera
cd wildlife-camera
```

**Option B: If transferring files manually:**
```bash
# On your local machine:
scp -r /path/to/wildlife-cameras/claude pi@<pi-ip>:~/wildlife-camera

# Then on Pi:
cd ~/wildlife-camera
```

### 3.2 Install Python Dependencies with Pixi

```bash
cd ~/wildlife-camera
pixi install
```

This will install all dependencies defined in `pixi.toml`.

---

## Step 4: Hardware Setup

### 4.1 Connect the NoIR Camera

1. **Power off the Pi:**
   ```bash
   sudo shutdown -h now
   ```

2. **Locate the camera connector:**
   - On Pi 4B: between HDMI and audio jack
   - The connector has a black plastic clip

3. **Insert the ribbon cable:**
   - Gently pull up the black clip (it slides up, not out)
   - Insert cable with **blue side facing the audio jack** (contacts facing HDMI)
   - Push the clip back down to secure

4. **Connect camera module:**
   - Attach the other end to the camera module
   - Blue side faces away from the lens

5. **Power on:**
   ```bash
   # Reconnect power to the Pi
   ```

### 4.2 Test Camera

```bash
# Test with libcamera
libcamera-hello --timeout 2000

# Or test with Python
python3 -c "from picamera2 import Picamera2; cam = Picamera2(); cam.start(); print('Camera working!'); cam.stop()"
```

If you see a preview window or "Camera working!", the camera is connected correctly.

---

## Step 5: Configure the Application

### 5.1 Create Storage Directories

```bash
cd ~/wildlife-camera
mkdir -p motion_events flow_signatures
```

### 5.2 Review Configuration

The main configuration is in `fastapi_mjpeg_server_with_storage.py`. Default settings:

```python
# Camera settings
width: 640
height: 480
frame_rate: 15

# Motion detection
motion_detection: True
motion_threshold: 25
motion_min_area: 500

# Optical flow
optical_flow_enabled: True
optical_flow_frame_skip: 2  # Process every 2nd frame

# Storage
store_motion_events: True
motion_storage_path: "./motion_events"
```

**To customize settings**, edit the `CameraConfig` class in `fastapi_mjpeg_server_with_storage.py` or use the web UI after starting.

### 5.3 Optional: Configure Remote Storage

If you want to upload events to a remote server, edit `StorageConfig` in `motion_storage.py`:

```python
# Remote storage settings
remote_storage_enabled: True
remote_storage_url: "https://your-server.com/upload"
remote_storage_api_key: "your-api-key"
```

---

## Step 6: Test the System

### 6.1 Run Manual Test

```bash
cd ~/wildlife-camera
pixi run start
```

You should see:
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 6.2 Access Web Interface

From another computer on the same network:
```
http://<pi-ip-address>:8000
```

You should see:
- Live camera stream
- Motion detection status
- Configuration panel
- Storage statistics

### 6.3 Test Motion Detection

Wave your hand in front of the camera. You should see:
- "Motion Detected!" alert appears
- Motion events logged in history
- Files created in `motion_events/` directory

### 6.4 Check Motion Events

```bash
ls -lh motion_events/
```

You should see directories with timestamps containing:
- `frames/` - Individual frame images
- `metadata.json` - Event details with optical flow classification
- `video.mp4` - Compiled video (if enabled)

### 6.5 Stop the Server

Press `Ctrl+C` in the terminal.

---

## Step 7: Set Up as System Service (Auto-Start)

### 7.1 Create Systemd Service File

```bash
sudo nano /etc/systemd/system/wildlife-camera.service
```

Paste the following (adjust paths if needed):

```ini
[Unit]
Description=Wildlife Camera Motion Detection System
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/wildlife-camera
Environment="PATH=/home/pi/.pixi/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/home/pi/.pixi/bin/pixi run start
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Save and exit (`Ctrl+X`, `Y`, `Enter`).

### 7.2 Enable and Start Service

```bash
# Reload systemd to recognize new service
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable wildlife-camera

# Start service now
sudo systemctl start wildlife-camera

# Check status
sudo systemctl status wildlife-camera
```

You should see "active (running)" in green.

### 7.3 View Logs

```bash
# View live logs
sudo journalctl -u wildlife-camera -f

# View recent logs
sudo journalctl -u wildlife-camera -n 100
```

### 7.4 Service Management Commands

```bash
# Stop service
sudo systemctl stop wildlife-camera

# Restart service
sudo systemctl restart wildlife-camera

# Disable auto-start
sudo systemctl disable wildlife-camera
```

---

## Step 8: Performance Tuning

### 8.1 Monitor System Resources

```bash
# Install htop
sudo apt install htop -y

# Monitor CPU, RAM, temperature
htop

# Check CPU temperature
vcgencmd measure_temp
```

### 8.2 Optimize for Performance

If system is slow (CPU > 80%, temp > 70°C):

**Option A: Reduce Frame Rate**
Edit `CameraConfig` in `fastapi_mjpeg_server_with_storage.py`:
```python
frame_rate: 10  # Reduce from 15
```

**Option B: Increase Frame Skip**
```python
optical_flow_frame_skip: 3  # Process every 3rd frame instead of 2nd
```

**Option C: Reduce Resolution**
```python
optical_flow_max_resolution: (240, 180)  # Reduce from (320, 240)
```

**Option D: Reduce Feature Points**
```python
optical_flow_feature_max: 50  # Reduce from 100
```

After changes:
```bash
sudo systemctl restart wildlife-camera
```

### 8.3 Add Cooling (If Overheating)

If temperature consistently > 75°C:
- Add heatsinks to CPU and power chip
- Add a case fan (recommended for 24/7 operation)
- Ensure good airflow around Pi

---

## Step 9: Storage Management

### 9.1 Check Disk Usage

```bash
df -h
```

### 9.2 Set Up Automatic Cleanup

Create cleanup script:
```bash
nano ~/wildlife-camera/cleanup_old_events.sh
```

Paste:
```bash
#!/bin/bash
# Delete motion events older than 30 days
find /home/pi/wildlife-camera/motion_events -type d -mtime +30 -exec rm -rf {} +
echo "Cleanup completed: $(date)"
```

Make executable:
```bash
chmod +x ~/wildlife-camera/cleanup_old_events.sh
```

Add to crontab (run daily at 3 AM):
```bash
crontab -e
```

Add line:
```
0 3 * * * /home/pi/wildlife-camera/cleanup_old_events.sh >> /home/pi/cleanup.log 2>&1
```

### 9.3 Optional: Use External USB Storage

```bash
# Plug in USB drive, find its device name
lsblk

# Create mount point
sudo mkdir -p /mnt/wildlife-storage

# Get UUID
sudo blkid

# Add to /etc/fstab for auto-mount
sudo nano /etc/fstab
# Add line (replace UUID with yours):
# UUID=xxxx-xxxx /mnt/wildlife-storage auto defaults,nofail 0 2

# Mount
sudo mount -a

# Update storage path in config to use /mnt/wildlife-storage
```

---

## Step 10: Remote Access Setup

### 10.1 Set Static IP Address

```bash
sudo nano /etc/dhcpcd.conf
```

Add at the end (adjust for your network):
```
interface wlan0
static ip_address=192.168.1.100/24
static routers=192.168.1.1
static domain_name_servers=192.168.1.1 8.8.8.8
```

Reboot:
```bash
sudo reboot
```

### 10.2 Enable Port Forwarding (Optional)

To access from outside your network:

1. Log into your router's admin panel
2. Find "Port Forwarding" settings
3. Forward external port 8000 → Pi IP:8000
4. Access via: `http://your-public-ip:8000`

**Security Note:** Consider using a reverse proxy with authentication (nginx + basic auth) for internet exposure.

---

## Verification Checklist

After deployment, verify:

- [ ] Camera stream visible at `http://<pi-ip>:8000`
- [ ] Motion detection triggers and creates events
- [ ] Motion events saved to disk with frames and metadata
- [ ] Optical flow classification appears in metadata.json
- [ ] Pattern database (`motion_patterns.db`) being populated
- [ ] System service auto-starts after reboot
- [ ] CPU usage reasonable (< 70% average)
- [ ] Temperature stable (< 70°C)
- [ ] Disk space monitored and cleanup configured
- [ ] Web UI "Show Motion Patterns" panel works

---

## Troubleshooting

### Camera Not Detected

```bash
# Check if camera is detected
vcgencmd get_camera

# Should show: supported=1 detected=1
```

If not detected:
- Check ribbon cable connection (both ends)
- Ensure camera interface enabled in raspi-config
- Try different ribbon cable

### ImportError: picamera2

```bash
# Reinstall picamera2
sudo apt install --reinstall python3-picamera2
```

### Service Fails to Start

```bash
# Check logs
sudo journalctl -u wildlife-camera -n 50

# Common issues:
# - Wrong paths in service file
# - Permissions on directories
# - Port 8000 already in use
```

### High CPU Usage

- Reduce frame_rate (10 FPS instead of 15)
- Increase optical_flow_frame_skip (3 or 4)
- Reduce optical_flow_feature_max (50 or 30)
- Disable optical_flow_visualization

### Motion Events Not Saving

```bash
# Check permissions
ls -ld motion_events/
# Should be writable by pi user

# Check disk space
df -h

# Check logs for errors
sudo journalctl -u wildlife-camera | grep ERROR
```

### Web Interface Not Accessible

```bash
# Check if service is running
sudo systemctl status wildlife-camera

# Check if port 8000 is listening
sudo netstat -tlnp | grep 8000

# Check firewall (usually disabled on Pi)
sudo ufw status
```

### Pattern Database Errors

```bash
# Check database file
ls -lh motion_patterns.db

# If corrupted, remove and restart
rm motion_patterns.db
sudo systemctl restart wildlife-camera
```

---

## Next Steps After Deployment

1. **Let it run for 24-48 hours** to collect diverse motion events
2. **Review classifications** using "Show Motion Patterns" in web UI
3. **Correct misclassifications** using the Relabel button
4. **Monitor performance** and tune parameters as needed
5. **Set up remote storage** if desired
6. **Configure alerts** (future enhancement)

---

## Support Files

Quick reference commands saved to `~/wildlife-camera-commands.txt`:

```bash
cat > ~/wildlife-camera-commands.txt << 'EOF'
# Wildlife Camera Quick Commands

# View live logs
sudo journalctl -u wildlife-camera -f

# Restart service
sudo systemctl restart wildlife-camera

# Check status
sudo systemctl status wildlife-camera

# View recent events
ls -lt motion_events/ | head

# Check disk space
df -h

# Check temperature
vcgencmd measure_temp

# Test camera
libcamera-hello --timeout 2000

# Access web UI (from browser)
http://<pi-ip>:8000
EOF
```

---

## Contact & Support

For issues or questions:
- Check logs: `sudo journalctl -u wildlife-camera -n 100`
- Review this guide's Troubleshooting section
- Check GitHub issues (if repository is public)

**Congratulations! Your wildlife camera system is now deployed and running!**

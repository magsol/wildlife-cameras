#!/bin/bash
# Wildlife Camera Raspberry Pi Setup Script
# Run this script on a fresh Raspberry Pi OS Bullseye installation

set -e  # Exit on error

echo "=========================================="
echo "Wildlife Camera System - Setup Script"
echo "=========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Raspberry Pi
if [ ! -f /proc/device-tree/model ]; then
    echo -e "${RED}Error: This script must be run on a Raspberry Pi${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Detected Raspberry Pi:${NC}"
cat /proc/device-tree/model
echo ""

# Check if running as pi user (not root)
if [ "$EUID" -eq 0 ]; then
    echo -e "${RED}Error: Please run as pi user, not as root${NC}"
    echo "Usage: bash setup_pi.sh"
    exit 1
fi

# Step 1: Update system
echo -e "${YELLOW}Step 1: Updating system...${NC}"
sudo apt update
sudo apt upgrade -y
echo -e "${GREEN}✓ System updated${NC}"
echo ""

# Step 2: Enable camera interface
echo -e "${YELLOW}Step 2: Checking camera interface...${NC}"
if ! sudo raspi-config nonint get_camera | grep -q "0"; then
    echo "Enabling camera interface..."
    sudo raspi-config nonint do_camera 0
    echo -e "${GREEN}✓ Camera interface enabled${NC}"
    REBOOT_REQUIRED=1
else
    echo -e "${GREEN}✓ Camera interface already enabled${NC}"
fi
echo ""

# Step 3: Install system dependencies
echo -e "${YELLOW}Step 3: Installing system dependencies...${NC}"
echo "This may take 10-15 minutes..."
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
    python3-dev \
    htop \
    || {
        echo -e "${RED}Error: Failed to install dependencies${NC}"
        exit 1
    }
echo -e "${GREEN}✓ System dependencies installed${NC}"
echo ""

# Step 4: Install Pixi
echo -e "${YELLOW}Step 4: Installing Pixi package manager...${NC}"
if command -v pixi &> /dev/null; then
    echo -e "${GREEN}✓ Pixi already installed${NC}"
else
    curl -fsSL https://pixi.sh/install.sh | bash
    source ~/.bashrc
    export PATH="$HOME/.pixi/bin:$PATH"
    echo -e "${GREEN}✓ Pixi installed${NC}"
fi
echo ""

# Step 5: Create project directories
echo -e "${YELLOW}Step 5: Creating project directories...${NC}"
mkdir -p ~/wildlife-camera/motion_events
mkdir -p ~/wildlife-camera/flow_signatures
echo -e "${GREEN}✓ Project directories created${NC}"
echo ""

# Step 6: Test camera
echo -e "${YELLOW}Step 6: Testing camera connection...${NC}"
if vcgencmd get_camera | grep -q "detected=1"; then
    echo -e "${GREEN}✓ Camera detected${NC}"

    # Try to capture a test image
    if timeout 5 libcamera-hello --timeout 1000 &> /dev/null; then
        echo -e "${GREEN}✓ Camera is working${NC}"
    else
        echo -e "${YELLOW}⚠ Camera detected but test capture failed${NC}"
        echo "This may be normal if camera is not yet connected."
    fi
else
    echo -e "${YELLOW}⚠ Camera not detected${NC}"
    echo "Please ensure:"
    echo "  1. Camera ribbon cable is properly connected"
    echo "  2. Camera interface is enabled in raspi-config"
    echo "  3. You have rebooted after enabling camera"
fi
echo ""

# Step 7: Create quick reference commands file
echo -e "${YELLOW}Step 7: Creating quick reference commands...${NC}"
cat > ~/wildlife-camera-commands.txt << 'EOF'
# Wildlife Camera Quick Commands

# View live logs
sudo journalctl -u wildlife-camera -f

# Restart service
sudo systemctl restart wildlife-camera

# Check status
sudo systemctl status wildlife-camera

# View recent events
ls -lt ~/wildlife-camera/motion_events/ | head

# Check disk space
df -h

# Check temperature
vcgencmd measure_temp

# Test camera
libcamera-hello --timeout 2000

# Monitor system resources
htop

# Access web UI (from browser on same network)
# http://<pi-ip>:8000

# Manual start (for testing)
cd ~/wildlife-camera && pixi run start
EOF
echo -e "${GREEN}✓ Quick reference saved to ~/wildlife-camera-commands.txt${NC}"
echo ""

# Step 8: Create systemd service template
echo -e "${YELLOW}Step 8: Creating systemd service template...${NC}"
cat > ~/wildlife-camera.service << 'EOF'
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
EOF
echo -e "${GREEN}✓ Service template created at ~/wildlife-camera.service${NC}"
echo ""

# Summary
echo "=========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Transfer your wildlife camera code to ~/wildlife-camera/"
echo "   Example: scp -r local/path/* pi@<pi-ip>:~/wildlife-camera/"
echo ""
echo "2. Install Python dependencies:"
echo "   cd ~/wildlife-camera && pixi install"
echo ""
echo "3. Test the system:"
echo "   cd ~/wildlife-camera && pixi run start"
echo "   Then visit http://<pi-ip>:8000 in your browser"
echo ""
echo "4. Set up as system service:"
echo "   sudo cp ~/wildlife-camera.service /etc/systemd/system/"
echo "   sudo systemctl daemon-reload"
echo "   sudo systemctl enable wildlife-camera"
echo "   sudo systemctl start wildlife-camera"
echo ""
echo "5. Check logs:"
echo "   sudo journalctl -u wildlife-camera -f"
echo ""
echo "Quick reference: cat ~/wildlife-camera-commands.txt"
echo ""

# Check if reboot is required
if [ "$REBOOT_REQUIRED" = "1" ]; then
    echo -e "${YELLOW}⚠ REBOOT REQUIRED${NC}"
    echo "Camera interface was just enabled. Please reboot:"
    echo "  sudo reboot"
    echo ""
fi

echo "For full deployment instructions, see RASPBERRY_PI_DEPLOYMENT.md"
echo ""

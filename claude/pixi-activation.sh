#!/bin/bash
# Pixi activation script to enable access to system site-packages
# This is required for picamera2 which is only available via apt on Raspberry Pi

# Add system Python site-packages to PYTHONPATH
# Common locations for Raspberry Pi OS
if [ -d "/usr/lib/python3/dist-packages" ]; then
    export PYTHONPATH="/usr/lib/python3/dist-packages:${PYTHONPATH}"
fi

# Also check for python3.11 specific path (Raspberry Pi OS Bookworm)
if [ -d "/usr/lib/python3.11/dist-packages" ]; then
    export PYTHONPATH="/usr/lib/python3.11/dist-packages:${PYTHONPATH}"
fi

# Check for python3.9 (older Raspberry Pi OS)
if [ -d "/usr/lib/python3.9/dist-packages" ]; then
    export PYTHONPATH="/usr/lib/python3.9/dist-packages:${PYTHONPATH}"
fi

echo "System site-packages added to PYTHONPATH for picamera2 access"

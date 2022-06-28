# Wildlife Cameras 🦌️🐦️📷️

Scripts related to the outdoor wildlife cameras.

## Overview

There are three cameras:
  - [Pi Camera Module 2 NoIR](https://www.raspberrypi.com/products/pi-noir-camera-v2/)
  - Luxonis [OAK-1](https://docs.luxonis.com/projects/hardware/en/latest/pages/BW1093.html)
  - Luxonis [OAK-D-Lite](https://docs.luxonis.com/projects/hardware/en/latest/pages/DM9095.html)

The Luxonis cameras are meant for daytime use, while the NoIR camera will operate at night.

This repo will also host any processing scripts, though the Luxonis cameras have many built-in processing models that we'll leverage.

## Dependencies

These are running off a [Raspberry Pi 3B+](https://www.raspberrypi.com/products/raspberry-pi-3-model-b-plus/), so we're already a few years old, but that's ok.

 - running [micromamba](https://mamba.readthedocs.io/en/latest/installation.html#micromamba)
   - mamba 0.24.0
   - conda 4.13.0
   - python 3.9.13
 - DepthAI [install instructions](https://docs.luxonis.com/projects/api/en/latest/install/#raspberry-pi-os) for Raspberry Pi OS (in practice, this ends up using system tools, and so all it needs is basic Python + pip)
 - [imagezmq](https://github.com/jeffbass/imagezmq#dependencies-and-installation)
   - This hasn't been updated in a bit -- am hoping to submit some PRs as this progresses -- but it still works, even with the current versions of all the dependencies listed
   - `pip install imutils pyzmq imagezmq`
   - `apt-get install python3-opencv`
 - [PiCamera2](https://github.com/raspberrypi/picamera2#picamera2-on-pi-3-and-ealier-devices)
   - `apt-get install python3-libcamera python3-kms++`
   - `NOGUI=1 pip install picamera2`

This initializes a default environment that works well with both PiCamera2 and DepthAI devices.

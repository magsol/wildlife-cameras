# Wildlife Cameras 🦌️🐦️📷️

Scripts related to the outdoor wildlife cameras.

## Overview

There are three cameras:
  - [Pi Camera Module 3 NoIR Wide](https://www.raspberrypi.com/products/camera-module-3/)
  - Luxonis [OAK-D Pro W](https://shop.luxonis.com/products/oak-d-pro-w)

Both cameras are designed for daytime and nighttime use.

This repo will also host any processing scripts, though the Luxonis cameras have many built-in processing models that we'll leverage.

## Dependencies

These are running off a [Raspberry Pi 4 Model B](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/), so we're already a few years old, but that's ok.

 - DepthAI [install instructions](https://docs.luxonis.com/projects/api/en/latest/install/#raspberry-pi-os) for Raspberry Pi OS (in practice, this ends up using system tools, and so all it needs is basic Python + pip)
 - [PiCamera2](https://github.com/raspberrypi/picamera2)
   - `apt-get install build-essential python3-libcamera python3-pyqt5 python3-opengl python3-numpy python3-opencv python3-ipython`

The drawback here is that there are essentially two distinct Python environments: the one installed via `apt` which operates the Pi Camera Module, and the one installed over `pip` that operates the Luxonis cameras. Fortunately, we don't really need these two cameras to interact with each other, but it's something to keep in mind moving forward.
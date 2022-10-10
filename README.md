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

 - running [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge)
   - mamba 0.27.0
   - conda 22.9.0
   - python 3.9.13
 - DepthAI [install instructions](https://docs.luxonis.com/projects/api/en/latest/install/#raspberry-pi-os) for Raspberry Pi OS (in practice, this ends up using system tools, and so all it needs is basic Python + pip)
 - [imagezmq](https://github.com/jeffbass/imagezmq#dependencies-and-installation)
   - This hasn't been updated in a bit -- am hoping to submit some PRs as this progresses -- but it still works, even with the current versions of all the dependencies listed
   - `pip install imagezmq`
   - used conda-forge versions of `opencv`, `pip`, and `pyzmq`
 - [PiCamera2](https://github.com/raspberrypi/picamera2#picamera2-on-pi-3-and-ealier-devices)
   - `apt-get install build-essential python3-libcamera python3-kms++`
   - `mamba install piexif pillow`
   - `pip install pidng simplejpeg v4l2-python3 python-prctl`
   - `pip install --no-deps picamera2`
   - the final step here involved setting symlinks for `pykms` and `libcamera` inside the `${MAMBA_ROOT}`

This initializes a default environment that works well with both PiCamera2 and DepthAI devices.

## Troubleshooting

### New versions of Picamera2

A common problem I've been running into is that, when a new version of Picamera2 is released, the previous versions break--even to the point of not being able to ask the version that's installed to compare against the newest version. 

However, [a recent change is that the group is now releasing the latest Picamera2 as apt packages](https://github.com/raspberrypi/picamera2/issues/303). They're still available on PyPI--and I'm continuing to use them there, too--but evidently this seems to be the long-term way forward.

For the time being, though, continuing to use `pip install -U --no-deps picamera2` seems to work.
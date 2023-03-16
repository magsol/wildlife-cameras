FROM balenalib/raspberrypi3-debian:bullseye-run

LABEL org.opencontainers.image.source=https://github.com/magsol/wildlife-cameras
LABEL org.opencontainers.image.description="Dev environment for wildlife cameras project."
LABEL org.opencontainers.image.licenses=MIT

# These follow from the Picamera2 installation instructions,
# specifically omitting GUI dependencies.
# https://github.com/raspberrypi/picamera2#installation-using-apt
RUN apt-get update && apt-get -y upgrade && apt-get install -y \
    python3-picamera2 \
    python3-numpy \
    python3-astral \
    python3-twilio \
    python3-opencv \
    git \
&& rm -rf /var/lib/apt/lists/*

# Create a `python` alias in the PATH.
RUN ln -s /usr/bin/python3 /usr/bin/python

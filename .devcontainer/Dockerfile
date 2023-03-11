FROM balenalib/raspberrypi3-debian:bullseye-run

LABEL org.quinnwitz.house.image.authors="Shannon Quinn"
LABEL org.quinnwitz.house.image.email="magsol@gmail.com"
LABEL version="1.0"
LABEL description="This is meant as a development enironment for a Raspberry Pi."

# These follow from the Picamera2 installation instructions,
# specifically omitting GUI dependencies.
# https://github.com/raspberrypi/picamera2#installation-using-apt
RUN apt-get update && apt-get -y upgrade && apt-get install -y \
    python3-picamera2 \
    python3-numpy \
    python3-astral \
    python3-twilio \
    python3-opencv \
&& rm -rf /var/lib/apt/lists/*

# Create a `python` alias in the PATH.
RUN ln -s /usr/bin/python3 /usr/bin/python

ENTRYPOINT "/bin/bash"

FROM debian:11

LABEL org.quinnwitz.house.image.authors="Shannon Quinn"
LABEL org.quinnwitz.house.image.email="magsol@gmail.com"
LABEL version="1.0"
LABEL description="This is meant as a development enironment for a Raspberry Pi."

RUN apt-get update && apt-get -y upgrade

# These follow from the Picamera2 installation instructions,
# specifically omitting GUI dependencies.
# https://github.com/raspberrypi/picamera2#installation-using-apt
RUN apt-get -y install python3-picamera2 --no-install-recommends

# Now installing other packages that are useful to this project.
RUN apt-get -y install \
    python3-numpy \
    python3-astral \
    python3-twilio \
    python3-opencv

import time
from datetime import datetime, timedelta
import signal
import os.path as osp

from astral import Location
import numpy as np
import cv2

from picamera2 import Picamera2, MappedArray
from picamera2.encoders import H264Encoder
from picamera2.outputs import CircularOutput

# Based on two examples:
# - https://github.com/raspberrypi/picamera2/blob/main/examples/timestamped_video.py
# - https://github.com/raspberrypi/picamera2/blob/main/examples/capture_circular.py

def apply_timestamp(request):
    ts = time.strftime("%Y-%m-%d %X")
    origin = (100, 100)
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 255, 0)
    thickness = 2

    with MappedArray(request, "main") as m:
        cv2.putText(
            img = m.array, 
            text = ts,
            org = origin,
            fontFace = font_face,
            fontScale = font_scale,
            color = color,
            thickness = thickness
        )

def sig_handler(signum, frame):
    pass

# Some preliminaries.
data_dir = '/opt/data'
signal.signal(signal.SIGINT, sig_handler)
signal.signal(signal.SIGTERM, sig_handler)
days = 3
l = Location(("Athens", "Georgia", 33.9519, -83.3576, "US/Eastern", 0))
set_offset = timedelta(hours = 1)
rise_offset = -timedelta(hours = 0.5)
lsize = (410, 308)
hsize = (1640, 1232)
picam2 = Picamera2()
picam2.pre_callback = apply_timestamp
video_config = picam2.create_video_configuration(
    main={"size": hsize, "format": "RGB888"},
    lores={"size": lsize, "format": "YUV420"})
picam2.configure(video_config)

# Set up the encoder.
encoder = H264Encoder(1000000, repeat=True)
encoder.output = CircularOutput()
picam2.encoder = encoder
picam2.start()
picam2.start_encoder()

# Start the loop.
for day in range(days):
    # Loop related variables.
    sun = l.sun()
    sunset = (sun["sunset"] + set_offset).timestamp()
    sunrise = (sun["sunrise"] + timedelta(days = 1) + rise_offset).timestamp()
    prev_gray = None
    encoding = False

    curr_time = time.time()
    mses = []

    if curr_time <= sunset and curr_time >= sunrise:
        # Sleep until nighttime.
        print(f"Sleeping for {sunset - curr_time} seconds...")
        time.sleep(sunset - curr_time)
        curr_time = time.time()
    
    print(f"I'm awake! Let's do day {day + 1}!")
    while curr_time < sunrise:
        curr_frame = picam2.capture_array("lores")
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_YUV420P2GRAY)
        if prev_gray is not None:
            # Measure pixels differences between current and
            # previous frame
            mse = np.square(curr_gray - prev_gray).mean()
            mses.append(mse)
            if mse > 2:
                if not encoding:
                    epoch = int(time.time())
                    print(f"New motion detected at ts {epoch}!")
                    encoder.output.fileoutput = osp.join(data_dir, f"{epoch}.h264")
                    encoder.output.start()
                    encoding = True
                ltime = time.time()
            else:
                if encoding and time.time() - ltime > 8.0:
                    encoder.output.stop()
                    encoding = False
                    print(f"Motion ended for {epoch}.")
        prev_gray = curr_gray
        curr_time = time.time()

    np.save(osp.join(data_dir, f"mse_day{day}.npy"), np.array(mses))
    print(f"Day {day + 1} is finished.")
    time.sleep(5) # Just to help things out.

print("THAT'S ALL FOLKS!")
picam2.stop_encoder()

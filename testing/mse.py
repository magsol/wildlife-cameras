import time
import os.path

import numpy as np
import cv2

from picamera2 import Picamera2, MappedArray
from picamera2.encoders import H264Encoder
from picamera2.outputs import CircularOutput

# - https://github.com/raspberrypi/picamera2/blob/main/examples/capture_circular.py

# Some preliminaries.
lsize = (1280, 720)
hsize = (1920, 1080)
picam2 = Picamera2()
video_config = picam2.create_video_configuration(
    controls={"FrameDurationLimits": (33333, 33333)},
    main={"size": hsize, "format": "RGB888"},
    lores={"size": lsize, "format": "YUV420"})
picam2.configure(video_config)
picam2.start()

out_dir = "/opt/data"
days = 3

for day in range(days):

    # Loop related variables.
    prev_gray = None
    curr_time = time.time()
    end_time = curr_time + (60 * 60 * 24) # Record for 24 hours.
    mses = []
    tss = []
    lum = []

    print(f"Starting day {day + 1}...")
    while curr_time < end_time:
        curr_frame = picam2.capture_array("lores")
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_YUV420P2GRAY)
        lum.append(curr_gray.sum())
        curr_time = time.time()
        if prev_gray is not None:
            # Measure pixels differences between current and
            # previous frame
            mse = np.square(curr_gray - prev_gray).mean()
            mses.append(mse)
            tss.append(int(curr_time))
        prev_gray = curr_gray

    print(f"Ending day {day + 1}...")
    np.save(os.path.join(out_dir, f"lum_{day + 2}.npy"), np.array(lum))
    np.save(os.path.join(out_dir, f"mse_{day + 2}.npy"), np.array(mses))
    np.save(os.path.join(out_dir, f"ts_{day + 2}.npy"), np.array(tss))

picam2.stop()
#picam2.stop_encoder()

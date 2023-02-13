import time

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

# Some preliminaries.
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

# Loop related variables.
prev_gray = None
encoding = False
curr_time = time.time()
#end_time = curr_time + (60 * 60 * 8) # Record for 8 hours.
end_time = curr_time + (60 * 10) # Record for 10 minutes.
ltime = 0
mses = []

# while curr_time < end_time:
#     #frame = picam2.capture_array("lores")
#     #frame = cv2.cvtColor(frame, cv2.COLOR_YUV420P2GRAY)
#     #frames.append(frame)
#     (main, lores), m = picam2.capture_arrays(["main", "lores"])
#     m_frames.append(main)
#     l_frames.append(lores)
#     metadata.append(m)
#     curr = time.time()

while curr_time < end_time:
    curr_frame = picam2.capture_array("lores")
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_YUV420P2GRAY)
    if prev_gray is not None:
        # Measure pixels differences between current and
        # previous frame
        mse = np.square(curr_gray - prev_gray).mean()
        mses.append(mse)
        if mse > 7:
            if not encoding:
                epoch = int(time.time())
                encoder.output.fileoutput = f"{epoch}.h264"
                encoder.output.start()
                encoding = True
                print("New Motion", mse)
            ltime = time.time()
        else:
            if encoding and time.time() - ltime > 5.0:
                encoder.output.stop()
                encoding = False
    prev_gray = curr_gray
    curr_time = time.time()

picam2.stop_encoder()
np.save("mse.npy", np.array(mses))
# np.save("m_video.npy", np.array(m_frames))
# np.save("l_video.npy", np.array(l_frames))
# print(metadata)

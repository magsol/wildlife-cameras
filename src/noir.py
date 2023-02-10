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
    font_scale = 5
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

#lsize = (640, 480)
#hsize = (1920, 1080)
lsize = (320, 240)
hsize = (1280, 720)

picam2 = Picamera2()
picam2.pre_callback = apply_timestamp
video_config = picam2.create_video_configuration(
    main={"size": hsize, "format": "RGB888"},
    lores={"size": lsize, "format": "YUV420"})
picam2.configure(video_config)
encoder = H264Encoder(1000000, repeat=True)
encoder.output = CircularOutput()
picam2.encoder = encoder
#picam2.start_encoder()
picam2.start()

w, h = lsize
prev = None
encoding = False
ltime = 0
curr = time.time()
# end = curr + (60 * 60 * 8) # Record for 8 hours.
end = curr + 60 # record for one minute

frames = []
while curr < end:
    frame = picam2.capture_array("lores")
    frames.append(frame)
    curr = time.time()

# while True:
#     curr = picam2.capture_array("lores")
#     curr = curr[h, :]
#     if prev is not None:
#         # Measure pixels differences between current and
#         # previous frame
#         mse = np.square(np.subtract(curr, prev)).mean()
#         if mse > 7:
#             if not encoding:
#                 epoch = int(time.time())
#                 encoder.output.fileoutput = "{}.h264".format(epoch)
#                 encoder.output.start()
#                 encoding = True
#                 print("New Motion", mse)
#             ltime = time.time()
#         else:
#             if encoding and time.time() - ltime > 5.0:
#                 encoder.output.stop()
#                 encoding = False
#     prev = curr

#picam2.stop_encoder()
np.save("video.npy", np.array(frames))

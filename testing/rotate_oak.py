import cv2
import depthai as dai
import numpy as np
import time

# Create pipeline.
pipeline = dai.Pipeline()

# Define pipeline nodes (in this case: source and output).
cam = pipeline.create(dai.node.ColorCamera)
xout = pipeline.create(dai.node.XLinkOut)

# Create node properties.
xout.setStreamName("video")
cam.setPreviewSize(300, 300)
cam.setInterleaved(False)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
cam.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)

# Link.
cam.preview.link(xout.input)

oak_d = dai.Device.getAllAvailableDevices()[0]
# oak_1 = dai.Device.getAllAvailableDevices()[1]

record_time = 10

vid = []
with dai.Device(pipeline, oak_d) as device:
    qrgb = device.getOutputQueue(name = "video", maxSize = 4, blocking = False)
    start = time.time()
    while True:
        inrgb = qrgb.get()
        f = np.array(inrgb.getCvFrame())
        vid.append(f)
        if time.time() - start > record_time:
            break
np.save("video.npy", np.array(vid))
print("Success!")

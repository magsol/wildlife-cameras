import socket
import time
import imagezmq
import cv2
import depthai as dai
import multiprocessing as mp

# DepthAI setup
pipeline = dai.Pipeline()
cameras = dai.Device.getAllAvailableDevices()
camRgb = pipeline.create(dai.node.ColorCamera)
xoutVideo = pipeline.create(dai.node.XLinkOut)
xoutVideo.setStreamName("video")

#camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
#camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
#camRgb.setVideoSize(1920, 1080)

camRgb.setPreviewSize(320, 240)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
#xoutVideo.input.setBlocking(False)
#xoutVideo.input.setQueueSize(1)

camRgb.preview.link(xoutVideo.input)
#camRgb.video.link(xoutVideo.input)

# ImageZMQ setup
sender = imagezmq.ImageSender(connect_to = "tcp://Dinraal.local:5555")
pi_name = socket.gethostname()
cam1 = f'{pi_name}_{cameras[0]}'
cam2 = f'{pi_name}_{cameras[1]}'

n_seconds = 10
with dai.Device(pipeline, cameras[1]) as device:
    video = device.getOutputQueue(name = "video", maxSize = 4, blocking = False)

    start_time = time.time()
    while time.time() - start_time < n_seconds:
        videoIn = video.get()
        frame = videoIn.getCvFrame()
        sender.send_image(cam1, frame)

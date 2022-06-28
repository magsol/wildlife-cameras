import cv2
import depthai as dai
import numpy as np

pipeline = dai.Pipeline()

cam = pipeline.create(dai.node.ColorCamera)

script = pipeline.create(dai.node.Script)
script.setScript("""
        import time
        ctrl = CameraControl()
        ctrl.setCaptureStill(True)
        node.io['out'].send(ctrl)
        """)

xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("still")

script.outputs['out'].link(cam.inputControl)
cam.still.link(xout.input)

# We have multiple devices, so let's do each one.
for dv in dai.Device.getAllAvailableDevices():
    dvId = dv.getMxId()
    _, dvInfo = dai.Device.getDeviceByMxId(dvId)

    with dai.Device(pipeline, dvInfo) as device:
        img = device.getOutputQueue("still").get()
        npy = img.getFrame()
        np.save(f"img_{dvId}.npy", npy)
print("Success!")

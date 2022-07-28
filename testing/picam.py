from picamera2 import Picamera2, MappedArray
from picamera2.outputs import FileOutput
import cv2
import time

picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration())

def apply_timestamp(request):
    ts = time.strftime("%Y-%m-%d %X")
    with MappedArray(request, "main") as m:
        cv2.putText(m.array, ts, (100, 100), 
            cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0))

picam2.pre_callback = apply_timestamp

picam2.start()
time.sleep(2)
metadata = picam2.capture_file("test.jpg")
print(metadata)
picam2.close()
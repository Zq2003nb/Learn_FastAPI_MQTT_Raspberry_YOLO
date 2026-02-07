from picamera2 import Picamera2
from ultralytics import YOLO
import torch
import cv2
import os



picam = Picamera2()
picam_config = picam.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"},
    lores={"size": (320, 240), "format": "YUV420"},
    display="lores"
)
picam.configure(picam_config)
picam.start()


model = YOLO("yolov8n.pt")


cv2.namedWindow("YOLO", cv2.WINDOW_NORMAL)

try:
    while True:

        pic_frame = picam.capture_array()


        results = model.predict(
            source=pic_frame,
            conf=0.5,
            device="cpu",
            verbose=False
        )


        annotated_frame = results[0].plot()

        cv2.imshow("YOLO", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:

    picam.stop()
    picam.close()
    cv2.destroyAllWindows()
    print("exit")

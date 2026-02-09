import time

from picamera2 import Picamera2
from ultralytics import YOLO
import torch
import cv2
import os

from ultralytics.engine.results import Boxes


def count_classes_num(boxes: list[Boxes], names: dict[int, str]) -> dict[str, int]:
    """
    :param boxes:
    :param names:
    :return:
    """
    dict_result: dict[str, int] = {}
    for box in boxes:
        cls_id: int = box.cls.cpu().detach().numpy()[0].astype(int)
        for key in names.keys():
            if cls_id == key:
                if names[key] in dict_result:
                    dict_result[names[key]] += 1
                else:
                    dict_result[names[key]] = 1
    return dict_result


picam = Picamera2()
picam_config = picam.create_still_configuration(
    main={"size": (640, 480), "format": "RGB888"}

)
picam.configure(picam_config)

model = YOLO("yolov8n.pt")
picam.start()
try:
    while True:
        picam.capture_file("p1.jpg")
        results = model.predict(source="p1.jpg",
                                conf=0.5,
                                device="cpu",
                                verbose=False)
        time.sleep(1)

        print(count_classes_num(results[0].boxes, results[0].names))
finally:
    picam.stop()
    picam.close()
    print("exit")

"""
version=1.0
目标实现实时检测和数量检测
实现数量缺失的图片保存
"""
import cv2
import time
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2

"""
全局数据类型
"""
model_path = "yolov8n.pt"
class_names = ['baishikele', 'chapai', 'fenda', 'binghongcha', 'xuebi', 'hongniu', 'person']
frame_interval = 2
picam = Picamera2()
model = YOLO(model_path)

#初始化摄像头
picam.configure(picam.create_preview_configuration(main={"format": 'RGB888', "size": (480, 360)}))
picam.start()


def detect_and_count(frame):
    dict_result: dict[str, int] = {}
    results = model.predict(source=frame,
                            conf=0.5,
                            device="cpu",
                            verbose=False)
    boxes = results[0].boxes
    names = results[0].names
    for box in boxes:
        cls_id: int = box.cls.cpu().detach().numpy()[0].astype(int)
        cls_name: str = names[cls_id]
        dict_result[cls_name] = dict_result.get(cls_name, 0) + 1
    return results[0].plot(), dict_result


if __name__ == "__CamShelf_raspi__":
    print("按“q”退出\n")
    last_detect_time = 0
    try:
        #将摄像头得到的frame检测
        while True:
            frame = picam.capture_array()
            current_time = time.time()
            if current_time - last_detect_time >= frame_interval:
                annotated_frame,counts = detect_and_count(frame)
                last_detect_time = current_time
                print(f"实时商品数：{counts}")
                cv2.imshow("实时画面", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break;
    except KeyboardInterrupt:
        print("按q结束了")
    finally:
        picam.stop()
        picam.close()
        cv2.destroyAllWindows()





from picamera2 import Picamera2
import time

'''
下面的功能是用树莓派的摄像头拍一张照片并保存
'''


def take_picture() -> None:
    picam = Picamera2()

    camera_config = picam.create_still_configuration(
        main={
            "size": (640, 480),
            "format":"RGB888"
        }
    )

    picam.configure(camera_config)
    picam.start()
    time.sleep(2)
    picam.capture_file("shelf_photo.jpg")
    picam.stop()
    picam.close()


if __name__ == "__day1__":
    take_picture()

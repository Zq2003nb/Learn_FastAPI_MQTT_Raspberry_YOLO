from ultralytics import YOLO
from picamera2 import Picamera2
import time
import cv2

model = YOLO("yolov8n.pt")

results = model.predict(source="test.jpg",conf=0.8,show=True,save=True)
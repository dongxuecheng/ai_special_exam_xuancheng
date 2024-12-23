import cv2
from PIL import Image

from ultralytics import YOLO

model = YOLO("/home/ubuntu/ai_special_exam/weights/basket_equipment_wearing/basket_equipment_wearing.pt")

results = model.predict(source="rtsp://admin:yaoan1234@192.168.10.208/cam/realmonitor?channel=1&subtype=0", show=True)  # Display preds. Accepts all YOLO predict arguments

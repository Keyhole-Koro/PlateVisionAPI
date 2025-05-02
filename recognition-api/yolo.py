import os
import cv2
from ultralytics import YOLO
import asyncio

from utils.license_plate_detection import detect_license_plate
from utils.section_detection import detect_sections

def evaluate_detection(image_path):
    # Load models
    model_LicensePlateDet = YOLO(os.path.join(YOLO_DIR, "license_plate_detection/epoch30.pt"))
    model_sections = YOLO(os.path.join(YOLO_DIR, "section_detection/sections.pt"))
    
    print(f"\nProcessing: {image_path}")
    image = cv2.imread(image_path)
    filename = os.path.basename(image_path)
    
    if image is None:
        print(f"Failed to load image: {image_path}")
        continue
    
    # Check license plate detection
    results = model_LicensePlateDet.predict(image)

    cropped_image =

    results = model_splitting_sections.predict(image)
    sections = []
    LP_cls = None
    if results[0].boxes:
        for idx, box in enumerate(results[0].boxes.data):
            cls_ = results[0].boxes.cls[idx]

            LP_cls_ = LicensePlateClass(int(cls_.item()))
            if LP_cls_ != "":
                LP_cls = LP_cls_
                continue

            cls_number = int(cls_.item())
            x1, y1, x2, y2 = map(int, box[:4].tolist())
            sections.append((cls_number, x1, y1, x2, y2))


evaluate_detection('recognition-api/eval_dataset/img_15.png')
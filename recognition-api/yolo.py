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

    sections, _ = detect_sections(model_sections, cropped)


evaluate_detection('recognition-api/eval_dataset/img_15.png')
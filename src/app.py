import asyncio
import cv2
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
import pytesseract

from utils.license_plate_detection import detect_license_plate
from utils.points_detection import detect_4points, apply_affine_transform
from utils.section_detection import detect_sections
from utils.ocr_processing import perform_ocr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")
YOLO_DIR = os.path.join(MODEL_DIR, "yolo")
TESSERACT_DIR = os.path.join(MODEL_DIR, "tesseract")

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# Create output directory
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

model_splitting_sections = YOLO(os.path.join(YOLO_DIR, "section_detection/sections.pt"))
#model_4PointsDet = YOLO(os.path.join(YOLO_DIR, "model/yolo/nano/4p_epoch55.pt"))
model_LicensePlateDet = YOLO(os.path.join(YOLO_DIR, "license_plate_detection/best.pt"))


def main(image_path):
    """ Full pipeline: detect, crop, transform, segment, and recognize text."""
    detections, image = detect_license_plate(model_LicensePlateDet, image_path) 
    
    for x1, y1, x2, y2 in detections:
        cropped = image[y1:y2, x1:x2]
        sections = detect_sections(model_splitting_sections, cropped)
        
        results = asyncio.run(perform_ocr(TESSERACT_DIR, cropped, sections))
        
        print("Recognized Text:", results)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    image_path = "../image/image7.png"  # Replace with your image path
    main(image_path)
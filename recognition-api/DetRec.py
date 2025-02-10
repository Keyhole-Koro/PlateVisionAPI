import asyncio
import cv2
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
import pytesseract
import psutil
import time

from utils.license_plate_detection import detect_license_plate
from utils.section_detection import detect_sections
from utils.ocr_processing import perform_ocr
from utils.compress import compress_image
from utils.memoryUsage import measure_time, monitor_memory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
YOLO_DIR = os.path.join(MODEL_DIR, "yolo")
TESSERACT_DIR = os.path.join(MODEL_DIR, "tesseract")

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

def load_yolo_model(model_path):
    """ Load the YOLO model in a blocking way. """
    return YOLO(model_path)

async def load_models():
    """ Load both YOLO models asynchronously. """
    loop = asyncio.get_event_loop()
    model_splitting_sections = await loop.run_in_executor(None, load_yolo_model, os.path.join(YOLO_DIR, "section_detection/sections.pt"))
    model_LicensePlateDet = await loop.run_in_executor(None, load_yolo_model, os.path.join(YOLO_DIR, "license_plate_detection/best.pt"))
    return model_splitting_sections, model_LicensePlateDet

async def detect_and_recognize(model_LicensePlateDet, model_splitting_sections, image, measure=False):
    """ Detect license plates, segment sections, and perform OCR. """
    async def process():
        results = []
        detections, image_lp = detect_license_plate(model_LicensePlateDet, image)
        for x1, y1, x2, y2 in detections:
            cropped = image_lp[y1:y2, x1:x2]
            sections = detect_sections(model_splitting_sections, cropped)
            result = await perform_ocr(TESSERACT_DIR, cropped, sections)
            results.append(result)

            print("Recognized Text:", result)
            plt.figure(figsize=(10, 6))
            plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
        return results

    return await measure_time(process)() if measure else await process()

async def full_pipeline(image, measure=False):
    """ Full pipeline: detect, crop, transform, segment, and recognize text. """
    
    if measure:
        memory_before = monitor_memory()
        print(f"Memory before execution: {memory_before} MB")
    else:
        memory_before = 0  # No memory measurement if measure is False
    
    model_splitting_sections, model_LicensePlateDet = await load_models()
    
    compressed_image = compress_image(image, compression_quality=80)
    
    result = await detect_and_recognize(model_LicensePlateDet, model_splitting_sections, compressed_image, measure)
    
    if measure:
        memory_after = monitor_memory()
        memory_used = memory_after - memory_before
        print(f"Memory after execution: {memory_after} MB")
        print(f"Memory used: {memory_used} MB")
    else:
        memory_used = 0  # No memory measurement if measure is False
    
    return result

if __name__ == "__main__":
    image_path = "../image/image7.png"
    image = cv2.imread(image_path)

    asyncio.run(full_pipeline(image, measure=True))

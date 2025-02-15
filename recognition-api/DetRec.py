import asyncio
import cv2
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
import pytesseract
import psutil
import time

from utils.license_plate_detection import detect_license_plate
from utils.section_detection import detect_sections, range_xy_sections
from utils.ocr_processing import perform_ocr
from utils.compress import compress_image
from utils.memoryUsage import measure_time, monitor_memory

from classification.inference import inference_class

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
    model_LicensePlateDet = await loop.run_in_executor(None, load_yolo_model, os.path.join(YOLO_DIR, "license_plate_detection/epoch30.pt"))
    return model_splitting_sections, model_LicensePlateDet

async def detect_and_recognize(model_LicensePlateDet, model_splitting_sections, image, measure=False):
    """ Detect license plates, segment sections, and perform OCR. """
    async def process():
        results = []
        detections, image_lp = detect_license_plate(model_LicensePlateDet, image)
        for x1, y1, x2, y2 in detections:
            cv2.rectangle(image_lp, (x1, y1), (x2, y2), (255, 0, 0), 2)

            cropped = image_lp[y1:y2, x1:x2]
            sections, LP_cls = detect_sections(model_splitting_sections, cropped)

            # for license plate class prediction
            min_x, min_y, max_x, max_y = range_xy_sections(sections)
            section_part_cropped = cropped[min_y+10:max_y-10, min_x+10:max_x-10]
            cv2.imwrite("section_part_cropped.jpg", section_part_cropped)
            lp_cls = inference_class(section_part_cropped,
                                     model_path="model/LicensePlateClassification/knn_model.pkl",
                                     scaler_path="model/LicensePlateClassification/scaler.pkl"
                                     )
            results.append({"class": lp_cls})

            result = await perform_ocr(TESSERACT_DIR,
                                       cropped,
                                       sections,
                                       de_pattern=lp_cls == "designed" or lp_cls == "private"
                                       )
            results.append(result)

            for (cls_number, s_x1, s_y1, s_x2, s_y2) in sections:
                # Draw bounding box and recognized text on the image
                cv2.rectangle(image_lp, (x1+s_x1, y1+s_y1), (x1+s_x2, y1+s_y2), (0, 255, 0), 2)

            print("Recognized Text:", result)
            plt.figure(figsize=(10, 6))
            plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
        
        # Show the final image with all bounding boxes and text
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(image_lp, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

        return results, image_lp

    return await measure_time(process)() if measure else await process()

async def full_pipeline(image, measure=False):
    """ Full pipeline: detect, crop, transform, segment, and recognize text. """
    
    if measure:
        memory_before = monitor_memory()
        print(f"Memory before execution: {memory_before} MB")
    else:
        memory_before = 0  # No memory measurement if measure is False
    
    model_splitting_sections, model_LicensePlateDet = await load_models()
    
    compressed_image = compress_image(image, compression_quality=100)
    
    result, processed_image = await detect_and_recognize(model_LicensePlateDet, model_splitting_sections, compressed_image, measure)
    
    if measure:
        memory_after = monitor_memory()
        memory_used = memory_after - memory_before
        print(f"Memory after execution: {memory_after} MB")
        print(f"Memory used: {memory_used} MB")
    else:
        memory_used = 0  # No memory measurement if measure is False
    
    return result, processed_image

if __name__ == "__main__":
    image_path = "../image/image7.png"
    image = cv2.imread(image_path)

    asyncio.run(full_pipeline(image, measure=True))

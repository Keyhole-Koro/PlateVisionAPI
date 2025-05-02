import asyncio
import cv2
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
import joblib

from utils.license_plate_detection import detect_license_plate
from utils.section_detection import detect_sections, range_xy_sections
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
    """ Load all models concurrently """
    loop = asyncio.get_event_loop()
    
    # Concurrent YOLO model loading
    yolo_tasks = [
        loop.run_in_executor(None, load_yolo_model, os.path.join(YOLO_DIR, "section_detection/new_sections.pt")),
        loop.run_in_executor(None, load_yolo_model, os.path.join(YOLO_DIR, "license_plate_detection/epoch30.pt"))
    ]
    
    # Concurrent joblib loading
    joblib_tasks = [
        loop.run_in_executor(None, joblib.load, os.path.join(MODEL_DIR, "LicensePlateClassification/knn_model.pkl")),
        loop.run_in_executor(None, joblib.load, os.path.join(MODEL_DIR, "LicensePlateClassification/scaler.pkl"))
    ]
    
    # Gather all tasks
    model_splitting_sections, model_LicensePlateDet = await asyncio.gather(*yolo_tasks)
    classification_model, classification_scaler = await asyncio.gather(*joblib_tasks)
    
    return model_splitting_sections, model_LicensePlateDet, classification_model, classification_scaler

async def detect_and_recognize(model_LicensePlateDet, model_splitting_sections, classification_model, classification_scaler, image, flags, measure=False, ocr_processor_config=None):
    """ Detect license plates, segment sections, and perform OCR. """
    async def process():
        results = []

        if not flags["recognition_only"]:
            detections, image_lp = detect_license_plate(model_LicensePlateDet, image)
        else:
            height = image.shape[0]
            width = image.shape[1]
            detections = [(0, 0, width, height)]
            image_lp = image

        for x1, y1, x2, y2 in detections:
            if flags["annotated_image"]:
                cv2.rectangle(image_lp, (x1, y1), (x2, y2), (255, 0, 0), 2)

            cropped = image_lp[y1:y2, x1:x2]
            sections, LP_cls = detect_sections(model_splitting_sections, cropped)

            if not sections:
                print("No sections detected, skipping...")
                continue
            
            lp_cls = inference_class(image_lp,
                                     model=classification_model,
                                     scaler=classification_scaler
                                     )
            
            result = {}

            for (cls_number, x1, y1, x2, y2) in sections:
                section_name = cls_[cls_number]
                if flags[section_name]:
                    count[section_name] += 1
                    section_part_cropped = cropped[y1:y2, x1:x2]
                    cv2.imwrite(f"output/{section_name}_{count[section_name]}.jpg", section_part_cropped)

                    engine_instance = ocr_processor_config[section_name]["engine_instance"]

                    if engine_instance is None:
                        print(f"No OCR engine for {section_name}, skipping...")
                        continue

                    result[section_name] = await engine_instance.recognize_text(section_part_cropped)
            
            result["class"] = lp_cls

            results.append(result)

            if flags["annotated_image"]:
                for (cls_number, s_x1, s_y1, s_x2, s_y2) in sections:
                    # Draw bounding box and recognized text on the image
                    cv2.rectangle(image_lp, (x1+s_x1, y1+s_y1), (x1+s_x2, y1+s_y2), (0, 255, 0), 2)

            print("Recognized Text:", result)

        return results, image_lp

    return await measure_time(process)() if measure else await process()

cls_ = {
    0: "region",
    1: "hiragana",
    2: "classification",
    3: "number"
}

count = {
    "region": 0,
    "hiragana": 0,
    "classification": 0,
    "number": 0
}
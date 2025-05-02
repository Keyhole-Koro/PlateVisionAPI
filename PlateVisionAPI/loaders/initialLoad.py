import asyncio
import os
from ultralytics import YOLO
import joblib
from root import BASE_DIR

MODEL_DIR = os.path.join(BASE_DIR, "models")
YOLO_DIR = os.path.join(MODEL_DIR, "yolo")
TESSERACT_DIR = os.path.join(MODEL_DIR, "tesseract")


def load_yolo_model(model_path):
    """ Load the YOLO model in a blocking way. """
    return YOLO(model_path)

def safe_load_joblib(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return joblib.load(file_path)

async def initial_load_models():
    """ Load all models concurrently """
    loop = asyncio.get_event_loop()
    
    # Concurrent YOLO model loading
    yolo_tasks = [
        loop.run_in_executor(None, load_yolo_model, os.path.join(YOLO_DIR, "section_detection", "new_sections.pt")),
        loop.run_in_executor(None, load_yolo_model, os.path.join(YOLO_DIR, "license_plate_detection", "epoch30.pt"))
    ]
    
    # Concurrent joblib loading with error handling
    joblib_tasks = [
        loop.run_in_executor(None, safe_load_joblib, os.path.join(MODEL_DIR, "LicensePlateClassification", "knn_model.pkl")),
        loop.run_in_executor(None, safe_load_joblib, os.path.join(MODEL_DIR, "LicensePlateClassification", "scaler.pkl"))
    ]
    
    # Gather all tasks
    model_splitting_sections, model_LicensePlateDet = await asyncio.gather(*yolo_tasks)
    classification_model, classification_scaler = await asyncio.gather(*joblib_tasks)
    
    return model_splitting_sections, model_LicensePlateDet, classification_model, classification_scaler

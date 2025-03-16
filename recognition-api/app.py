from fastapi import FastAPI, File, UploadFile, Query
import asyncio
from io import BytesIO
from PIL import Image, ImageOps
import base64
import os
import cv2
import numpy as np
import time

from DetRec import detect_and_recognize, load_models
from utils.ocr_processing import OCRProcessor

app = FastAPI()

UPLOAD_FOLDER = "upload"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

tesseract_engine = "tesseract"
paddle_engine = "paddle"

@app.on_event("startup")
async def startup_event():
    global model_LicensePlateDet, model_splitting_sections, classification_model, classification_scaler
    model_splitting_sections, model_LicensePlateDet, classification_model, classification_scaler = await load_models()

@app.post("/process_image/")
async def process_image(
    file: UploadFile = File(...),
    measure: bool = False,
    hiragana_model: str = Query("", description="OCR engine for hiragana"),
    classification_model: str = Query("", description="OCR engine for classification"),
    number_model: str = Query("", description="OCR engine for number"),
    recognition_only: bool = Query(False, description="Return only recognition result"),
    annotated_image: bool = Query(True, description="Return annotated image"),
    hiragana: bool = Query(True, description="Include hiragana in result"),
    classification: bool = Query(True, description="Include classification in result"),
    number: bool = Query(True, description="Include number in result")
):
    start_time = time.time()

    # Save the uploaded image to the upload folder
    file_location = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())

    # Read the image file into a numpy array
    image = Image.open(file_location)
    image = ImageOps.exif_transpose(image)
    image = np.array(image)

    flags = {
        "recognition_only": recognition_only,
        "annotated_image": annotated_image,
        "hiragana": hiragana,
        "classification": classification,
        "number": number,
        "region": False
    }

    # Initialize OCR processor with section-specific engines
    ocr_config = {
        "hiragana": {"engine": tesseract_engine, "model": hiragana_model, "lang": "japan"},
        "classification": {"engine": paddle_engine, "model": classification_model, "lang": "en"},
        "number": {"engine": paddle_engine, "model": number_model, "lang": "en"}
    }
    ocr_processor = OCRProcessor(config=ocr_config)

    # Run the pipeline with flags
    result, processed_image = await detect_and_recognize(
        model_LicensePlateDet, 
        model_splitting_sections,
        classification_model,
        classification_scaler, 
        image,
        flags,
        measure,
        ocr_processor
    )

    response = {"result": result}

    if annotated_image:
        # Convert and encode image only if requested
        processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        processed_image_location = os.path.join(UPLOAD_FOLDER, f"processed_{file.filename}")
        cv2.imwrite(processed_image_location, processed_image_rgb)
        _, buffer = cv2.imencode('.jpg', processed_image_rgb)
        response["image"] = base64.b64encode(buffer).decode('utf-8')

    # Add execution time
    response["execution_time"] = time.time() - start_time

    return response


async def main():
    start_time = time.time()

    # Load models
    model_splitting_sections, model_LicensePlateDet, classification_model, classification_scaler = await load_models()
    
    # Test parameters
    recognition_only = False
    annotated_image = True
    hiragana = True
    classification = True
    number = True
    
    # Use Linux-style path
    file_location = "img.jpg"
    image = cv2.imread(file_location)

    if image is None:
        print(f"Failed to load image: {file_location}")
        return

    flags = {
        "recognition_only": recognition_only,
        "annotated_image": annotated_image,
        "hiragana": hiragana,
        "classification": classification,
        "number": number,
        "region": False
    }

    # OCR configuration
    ocr_config = {
        "hiragana": {
            "engine": "paddle",
            "model": "hiragana",
            "lang": "japan"
        },
        "classification": {
            "engine": "paddle",
            "model": "classification",
            "lang": "en"
        },
        "number": {
            "engine": "paddle",
            "model": "number",
            "lang": "en"
        }
    }
    
    ocr_processor = OCRProcessor(config=ocr_config)

    # Run pipeline
    result, processed_image = await detect_and_recognize(
        model_LicensePlateDet,
        model_splitting_sections,
        classification_model,
        classification_scaler,
        image,
        flags,
        measure=True,
        ocr_processor=ocr_processor
    )

    print(f"Execution time: {time.time() - start_time}")
    print("Result:", result)

if __name__ == "__main__":
    asyncio.run(main())
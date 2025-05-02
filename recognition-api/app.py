from fastapi import FastAPI, File, UploadFile, Query
import asyncio
from PIL import Image, ImageOps
import base64
import os
import cv2
import numpy as np
import time

from DetRec import detect_and_recognize, load_models
from utils.ocr_processing import OCRProcessorConfig

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
    only_recognition: bool = Query(False, description="Return only recognition result"),
    return_image_annotation: bool = Query(True, description="Return annotated image"),
    hiragana: bool = Query(True, description="Include hiragana in result"),
    classification: bool = Query(True, description="Include classification in result"),
    number: bool = Query(True, description="Include number in result")
):

    # Save the uploaded image to the upload folder
    file_location = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())

    # Read the image file into a numpy array
    image = Image.open(file_location)
    image = ImageOps.exif_transpose(image)
    image = np.array(image)

    flags = {
        "only_recognition": only_recognition,
        "return_image_annotation": return_image_annotation,
        "hiragana": False,
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
    ocr_processor = OCRProcessorConfig(config=ocr_config)

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

    if return_image_annotation:
        # Convert and encode image only if requested
        processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        processed_image_location = os.path.join(UPLOAD_FOLDER, f"processed_{file.filename}")
        cv2.imwrite(processed_image_location, processed_image_rgb)
        _, buffer = cv2.imencode('.jpg', processed_image_rgb)
        response["anotated_image"] = base64.b64encode(buffer).decode('utf-8')

    return response


async def main():
    start_time = time.time()

    # Load models
    model_splitting_sections, model_LicensePlateDet, classification_model, classification_scaler = await load_models()
    
    # Test parameters
    only_recognition = False
    return_image_annotation = True
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
        "only_recognition": only_recognition,
        "return_image_annotation": return_image_annotation,
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
    
    ocr_processor_config = OCRProcessorConfig(config=ocr_config)

    # Run pipeline
    result, processed_image = await detect_and_recognize(
        model_LicensePlateDet,
        model_splitting_sections,
        classification_model,
        classification_scaler,
        image,
        flags,
        measure=True,
        ocr_processor_config=ocr_processor_config
    )

    print(f"Execution time: {time.time() - start_time}")
    print("Result:", result)

if __name__ == "__main__":
    asyncio.run(main())
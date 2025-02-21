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

app = FastAPI()

UPLOAD_FOLDER = "upload"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.on_event("startup")
async def startup_event():
    global model_LicensePlateDet, model_splitting_sections, classification_model, classification_scaler
    model_splitting_sections, model_LicensePlateDet, classification_model, classification_scaler = await load_models()

@app.post("/process_image/")
async def process_image(
    file: UploadFile = File(...),
    measure: bool = False,
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

    processed_section = {
        "hiragana": hiragana,
        "classification": classification,
        "number:": number,
        "region": False
    }

    # Run the pipeline with flags
    result, processed_image = await detect_and_recognize(
        model_LicensePlateDet, 
        model_splitting_sections,
        classification_model,
        classification_scaler, 
        image,
        processed_section,
        measure,
    )

    response = {"result": result}

    if annotated_image and not recognition_only:
        # Convert and encode image only if requested
        processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        processed_image_location = os.path.join(UPLOAD_FOLDER, f"processed_{file.filename}")
        cv2.imwrite(processed_image_location, processed_image_rgb)
        _, buffer = cv2.imencode('.jpg', processed_image_rgb)
        response["image"] = base64.b64encode(buffer).decode('utf-8')

    # Add execution time
    response["execution_time"] = time.time() - start_time

    return response
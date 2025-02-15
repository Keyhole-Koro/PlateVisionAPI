from fastapi import FastAPI, File, UploadFile
import asyncio
from io import BytesIO
from PIL import Image, ImageOps
import base64
import os
import cv2
import numpy as np

from DetRec import detect_and_recognize, load_models

app = FastAPI()

UPLOAD_FOLDER = "upload"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.on_event("startup")
async def startup_event():
    global model_LicensePlateDet, model_splitting_sections
    model_splitting_sections, model_LicensePlateDet = await load_models()

@app.post("/process_image/")
async def process_image(file: UploadFile = File(...), measure: bool = False):
    # Save the uploaded image to the upload folder
    file_location = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())

    # Read the image file into a numpy array
    image = Image.open(file_location)
    image = ImageOps.exif_transpose(image)  # Correct the orientation based on EXIF metadata
    image = np.array(image)

    # Run the pipeline
    result, processed_image = await detect_and_recognize(model_LicensePlateDet, model_splitting_sections, image, measure)
    
    # Convert the processed image from BGR to RGB
    processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    
    # Save the processed image to the upload folder
    processed_image_location = os.path.join(UPLOAD_FOLDER, f"processed_{file.filename}")
    cv2.imwrite(processed_image_location, processed_image_rgb)
    
    # Convert the processed image to a base64-encoded string
    _, buffer = cv2.imencode('.jpg', processed_image_rgb)
    processed_image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Return the result and the base64-encoded image
    return {"result": result, "image": processed_image_base64}
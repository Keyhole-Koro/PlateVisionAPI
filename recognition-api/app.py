from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import asyncio
from io import BytesIO
from PIL import Image

from DetRec import full_pipeline

app = FastAPI()

@app.post("/process_image/")
async def process_image(file: UploadFile = File(...), measure: bool = False):
    # Read the image file into a numpy array
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes))
    image = np.array(image)

    # Run the pipeline
    result = await full_pipeline(image, measure)
    
    # You can return the result in any desired format
    return {"result": result}
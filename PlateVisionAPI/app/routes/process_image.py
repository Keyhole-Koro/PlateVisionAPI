from fastapi import APIRouter, File, UploadFile, Query, Request
from PIL import Image, ImageOps
import base64
import cv2
import numpy as np
from io import BytesIO

from services.PlateVision import PlateVision
from services.recognition.ocr_processor import OCRProcessorConfig

router = APIRouter()

@router.post("/process_image/")
async def process_image(
    request: Request,
    file: UploadFile = File(...),
    measure: bool = False,
    classification_model: str = Query("", description="OCR engine for classification"),
    only_recognition: bool = Query(False, description="Return only recognition result"),
    return_image_annotation: bool = Query(True, description="Return annotated image"),
    return_classification: bool = Query(True, description="Include classification in result"),
    return_number: bool = Query(True, description="Include number in result")
):
    models = request.app.state.models
    model_splitting_sections = models["model_splitting_sections"]
    model_LicensePlateDet = models["model_LicensePlateDet"]
    classification_model = models["classification_model"]
    classification_scaler = models["classification_scaler"]

    file_content = await file.read()
    image = Image.open(BytesIO(file_content))
    image = ImageOps.exif_transpose(image)
    image = np.array(image)

    flags = {
        "only_recognition": only_recognition,
        "return_image_annotation": return_image_annotation,
        "hiragana": False,
        "classification": return_classification,
        "number": return_number,
        "region": False
    }

    # Initialize OCR processor with section-specific engines
    ocr_config = {
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
    ocr_processor = await OCRProcessorConfig(config=ocr_config)

    # Run the pipeline with flags
    result, processed_image = await PlateVision(
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
        _, buffer = cv2.imencode('.jpg', processed_image_rgb)
        response["annotated_image"] = base64.b64encode(buffer).decode('utf-8')

    return response
from fastapi import APIRouter, File, UploadFile, Query, Request, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image, ImageOps, UnidentifiedImageError
import base64
import cv2
import numpy as np
from io import BytesIO
import os
from dotenv import load_dotenv

from root import BASE_DIR
from services.PlateVision import PlateVision
from services.ocr.paddle_onnx_engine import PaddleonnxEngine
from services.detection.yolo_engine import YOLOEngine

# Load environment variables
load_dotenv()

router = APIRouter()

# Get the API keys from the environment and split them into a list
VALID_API_KEYS = os.getenv("API_KEYS", "").split(",")

print("Valid API keys:", VALID_API_KEYS)

class Base64ImageRequest(BaseModel):
    isBase64Encoded: bool
    body: str

@router.post("/process_image")
async def process_image(
    request: Request,
    file: UploadFile = File(None),
    measure: bool = False,
    only_recognition: bool = Query(False, description="Return only recognition result"),
    return_image_annotation: bool = Query(True, description="Return annotated image"),
    return_classification: bool = Query(True, description="Include classification in result"),
    return_number: bool = Query(True, description="Include number in result"),
    api_key: str = Header(None, description="API key for authentication")  # Get API key from headers
):
    try:
        # Validate API key
        #if api_key not in VALID_API_KEYS:
        #    return JSONResponse(content={"error": "Invalid API key"}, status_code=401)

        print("Received request to process image...")

        # Load image from file or base64
        if file:
            file_content = await file.read()
        else:
            data = await request.json()
            print("Request body:", data)
            if data.get("base64_data") and data["base64_data"].get("isBase64Encoded"):
                try:
                    file_content = base64.b64decode(data["base64_data"]["body"])
                except Exception as e:
                    return JSONResponse(content={"error": f"Invalid base64 data: {e}"}, status_code=400)
            else:
                return JSONResponse(content={
                    "error": f"No image data provided{type(file)}, {data.get('base64_data')}"
                }, status_code=400)

        # (rest of processing logic continues as before)

        try:
            image = Image.open(BytesIO(file_content))
            image = ImageOps.exif_transpose(image)
            image = np.array(image)
        except UnidentifiedImageError:
            return JSONResponse(content={"error": "Uploaded data is not a valid image"}, status_code=400)

        flags = {
            "only_recognition": only_recognition,
            "return_image_annotation": return_image_annotation,
            "hiragana": False,
            "classification": return_classification,
            "number": return_number,
            "region": False
        }

        classifying_model_config = {
            "model": {
                "path": os.path.join(BASE_DIR, "models", "classifying", "knn_model.pkl"),
            },
            "scaler": {
                "path": os.path.join(BASE_DIR, "models", "classifying", "scaler.pkl")
            }
        }

        detection_config = {
            "license_plate": {
                "engine": YOLOEngine,
                "engine_instance": None,
                "path": os.path.join(BASE_DIR, "models", "yolo", "license_plate.pt")
            },
            "splitting_sections": {
                "engine": YOLOEngine,
                "engine_instance": None,
                "path": os.path.join(BASE_DIR, "models", "yolo", "splitting_sections.pt")
            }
        }

        ocr_config = {
            "classification": {
                "engine": PaddleonnxEngine,
                "engine_instance": None,
                "path": os.path.join(BASE_DIR, "models", "paddle_onnx", "classification.onnx"),
                "dict_path": os.path.join(BASE_DIR, "models", "paddle_onnx", "en_dict.txt")
            },
            "number": {
                "engine": PaddleonnxEngine,
                "engine_instance": None,
                "path": os.path.join(BASE_DIR, "models", "paddle_onnx", "number.onnx"),
                "dict_path": os.path.join(BASE_DIR, "models", "paddle_onnx", "en_dict.txt")
            },
            "hiragana": None,
            "region": None
        }

        platevision_results = await PlateVision(
            image,
            classifying_model_config,
            detection_config,
            ocr_config,
            flags,
        )

        if return_image_annotation:
            for plate_result in platevision_results:
                annotated_image = create_annotated_image(image, plate_result)
                annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                _, buffer = cv2.imencode('.jpg', annotated_image_rgb)
                plate_result["annotated_image"] = base64.b64encode(buffer).decode('utf-8')

        return platevision_results

    except Exception as e:
        print(f"Error during processing: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


def create_annotated_image(image, plate_result):
    annotated_image = image.copy()
    
    # Get the splitting sections dictionary
    splitting_sections = plate_result.get("splitting_sections", {})

    # Iterate over each key-value pair in the splitting_sections dictionary
    for key, detection in splitting_sections.items():
        bbox = detection["bbox"]
        confidence = detection["confidence"]
        class_id = detection["class_id"]
        map_class_target = {0: "region", 1: "hiragana", 2: "classification", 3: "number"}
        label_key = map_class_target.get(class_id, key)  # Use the class ID to map the label
        
        # Draw the bounding box and label
        cv2.rectangle(annotated_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        label = f"{label_key}: {confidence:.2f}"
        cv2.putText(annotated_image, label, (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return annotated_image
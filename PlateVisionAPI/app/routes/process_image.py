from fastapi import APIRouter, File, UploadFile, Query, Request
from PIL import Image, ImageOps, UnidentifiedImageError
import base64
import cv2
import numpy as np
from io import BytesIO
import os

from root import BASE_DIR

from services.PlateVision import PlateVision
from services.ocr.paddle_onnx_engine import PaddleonnxEngine
from services.detection.yolo_engine import YOLOEngine

router = APIRouter()

@router.post("/process_image/")
async def process_image(
    request: Request,
    file: UploadFile = File(...),
    measure: bool = False,
    only_recognition: bool = Query(False, description="Return only recognition result"),
    return_image_annotation: bool = Query(True, description="Return annotated image"),
    return_classification: bool = Query(True, description="Include classification in result"),
    return_number: bool = Query(True, description="Include number in result")
):
    try:
        print("Processing image...")
        # Read the uploaded file as binary
        file_content = await file.read()

        # Convert the binary data to an image
        try:
            image = Image.open(BytesIO(file_content))
            image = ImageOps.exif_transpose(image)
            image = np.array(image)
        except UnidentifiedImageError:
            return {"error": "Uploaded file is not a valid image"}

        flags = {
            "only_recognition": only_recognition,
            "return_image_annotation": return_image_annotation,
            "hiragana": False,
            "classification": return_classification,
            "number": return_number,
            "region": False
        }

        classifying_model_config_no_loding_yet = {
            "model": {
                "path": os.path.join(BASE_DIR, "models", "classifying", "knn_model.pkl"),
            },
            "scaler": {
                "path": os.path.join(BASE_DIR, "models", "classifying", "scaler.pkl")
            }
        }

        detection_config_no_loding_yet = {
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

        ocr_config_no_loding_yet = {
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

        print("Loading models...")
        # Run the pipeline with flags
        platevisioin_results = await PlateVision(
            image,
            classifying_model_config_no_loding_yet,
            detection_config_no_loding_yet,
            ocr_config_no_loding_yet,
            flags,
        )
        print("Models loaded successfully.")

        response = {"result": platevisioin_results}

        print("Processing results...")
        if return_image_annotation:
            print("Creating annotated image...")
            annotated_image = create_annotated_image(image, platevisioin_results)
            # Convert and encode image only if requested
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            _, buffer = cv2.imencode('.jpg', annotated_image_rgb)
            response["annotated_image"] = base64.b64encode(buffer).decode('utf-8')

        return response

    except Exception as e:
        return {"error": str(e)}
    
def create_annotated_image(image, visionplate_result):
    """
    Create an annotated image with bounding boxes and labels.
    Args:
        image (numpy.ndarray): Original image.
        visionplate_result (dict): Dictionary containing OCR and detection results.
    Returns:
        numpy.ndarray: Annotated image.
    """
    annotated_image = image.copy()
    print("Annotated image shape:", annotated_image.shape)
    detections = visionplate_result["splitting_sections"]
    print("Detections:", detections)
    for detection in detections:
        print("Detection:", detection)
        bbox = detection["bbox"]
        confidence = detection["confidence"]
        class_id = detection["class_id"]
        print("Bounding box:", bbox)
        map_class_target = {
            0: "region",
            1: "hiragana",
            2: "classification",
            3: "number"
        }

        # Get OCR text for the current class
        key = map_class_target.get(class_id)
        ocr_text = visionplate_result.get(key, "")

        # Draw bounding box
        cv2.rectangle(annotated_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        # Label with class ID and confidence
        label = f"{key}: {confidence:.2f}"
        cv2.putText(annotated_image, label, (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Draw OCR text near the bottom of the bounding box
        if ocr_text:
            cv2.putText(annotated_image, str(ocr_text), (bbox[0], bbox[3] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return annotated_image
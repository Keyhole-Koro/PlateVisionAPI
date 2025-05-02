import cv2
import os

from services.detection.license_plate import det_license_plate
from services.detection.section import det_sections

from utils.memoryUsage import measure_time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
YOLO_DIR = os.path.join(MODEL_DIR, "yolo")
TESSERACT_DIR = os.path.join(MODEL_DIR, "tesseract")

async def PlateVision(model_LicensePlateDet, model_splitting_sections, classification_model, classification_scaler, image, flags, measure=False, ocr_processor_config=None):
    """ Detect license plates, segment sections, and perform OCR. """
    async def process():
        results = []

        if not flags["only_recognition"]:
            detections, image_lp = det_license_plate(model_LicensePlateDet, image)
        else:
            height = image.shape[0]
            width = image.shape[1]
            detections = [(0, 0, width, height)]
            image_lp = image

        for x1, y1, x2, y2 in detections:
            if flags["return_image_annotation"]:
                cv2.rectangle(image_lp, (x1, y1), (x2, y2), (255, 0, 0), 2)

            cropped = image_lp[y1:y2, x1:x2]
            sections, LP_cls = det_sections(model_splitting_sections, cropped)

            if not sections:
                print("No sections detected, skipping...")
                continue
            '''
            lp_cls = plate_class(image_lp,
                                     model=classification_model,
                                     scaler=classification_scaler
                                     )
            '''
            lp_cls = 0
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

            if flags["return_image_annotation"]:
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
import traceback
import logging
from loaders.loader import load_all_models

logger = logging.getLogger(__name__)

def format_error(message, exception):
    logger.error(f"{message}: {exception}\n{traceback.format_exc()}")
    return {"error": message}

async def PlateVision(image, classifying_model, detection_config, ocr_config, flags):
    try:
        # Load models
        configs = await load_models(classifying_model, detection_config, ocr_config)
        if "error" in configs:
            return configs

        results = []

        # Step 1: Detection or Fallback
        license_plates = [{"bbox": [0, 0, image.shape[1], image.shape[0]]}] if flags.get("only_recognition") \
            else await detect_license_plates(image, configs)
        if isinstance(license_plates, dict) and "error" in license_plates:
            return license_plates

        # Step 2: For each plate
        for plate in license_plates:
            result_plate = {}
            try:
                plate_image = crop_image(image, plate.get("bbox", []))
                if isinstance(plate_image, dict):  # error occurred
                    result_plate["error"] = plate_image["error"]
                    results.append(result_plate)
                    continue

                # Step 3: Section Detection
                sections = await detect_splitting_sections(plate_image, configs, flags)
                if isinstance(sections, dict) and "error" in sections:
                    result_plate["error"] = sections["error"]
                    results.append(result_plate)
                    continue
                result_plate["splitting_sections"] = sections

                # Step 4: OCR per section
                for section, section_data in sections.items():
                    cropped = crop_image(plate_image, section_data.get("bbox", []))
                    if isinstance(cropped, dict):  # error occurred
                        result_plate[section] = cropped
                        continue

                    ocr_result = await ocr_sections(cropped, section, configs)
                    result_plate[section] = ocr_result

            except Exception as e:
                result_plate["error"] = format_error("Unexpected error processing plate", e)
            results.append(result_plate)

        return results

    except Exception as e:
        return format_error("Fatal error in PlateVision", e)
async def detect_license_plates(image, configs):
    try:
        engine = configs["detection"]["license_plate"].get("engine_instance")
        if engine:
            return await engine.detect(image)
        return []
    except Exception as e:
        return format_error("Failed to detect license plates", e)

async def detect_splitting_sections(plate_image, configs, flags):
    try:
        engine = configs["detection"].get("splitting_sections", {}).get("engine_instance")
        if not engine:
            raise ValueError("Splitting section engine not loaded")

        raw_sections = await engine.detect(plate_image)
        if not raw_sections:
            return {}

        map_class_target = {
            0: "region",
            1: "hiragana",
            2: "classification",
            3: "number"
        }

        return {
            map_class_target[s.get("class_id")]: s
            for s in raw_sections
            if map_class_target.get(s.get("class_id")) in flags and flags[map_class_target[s["class_id"]]]
        }

    except Exception as e:
        return format_error("Failed to detect splitting sections", e)

async def ocr_sections(image, section, configs):
    try:
        engine = configs["ocr"].get(section, {}).get("engine_instance")
        if engine:
            return await engine.recognize_text(image)
        raise ValueError(f"OCR engine for section '{section}' is not available")
    except Exception as e:
        return format_error(f"OCR failed for section '{section}'", e)

async def load_models(class_model, detect_config, ocr_config):
    try:
        return await load_all_models(class_model, detect_config, ocr_config)
    except Exception as e:
        return format_error("Model loading failed", e)

def crop_image(image, bbox):
    try:
        if len(bbox) != 4:
            raise ValueError("Invalid bbox format")
        return image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    except Exception as e:
        return format_error("Cropping image failed", e)

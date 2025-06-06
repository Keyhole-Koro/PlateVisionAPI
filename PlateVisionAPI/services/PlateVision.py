import traceback
from loaders.loader import load_all_models

async def PlateVision(
    image,  # numpy.ndarray: The input image to process
    classifying_model,  # dict: Configuration for the classification model
    detection_config,  # dict: Configuration for the detection model
    ocr_config,  # dict: Configuration for the OCR model
    flags,  # dict: Flags to control the behavior of the function
):
    try:
        # Load models and validate configurations
        configs_with_engine_loaded_models = await load_models(
            classifying_model,
            detection_config,
            ocr_config,
        )

        # Initialize results
        results = []

        if flags.get("only_recognition", False):
            # Single bounding box covering the entire image
            license_plates = [{"bbox": [0, 0, image.shape[1], image.shape[0]]}]
        else:
            # Detect license plates
            license_plates = await detect_license_plates(image, configs_with_engine_loaded_models)

        for plate in license_plates:
            try:
                result_plate = {}
                print(f"plate: {plate}")
                # Crop the license plate image
                plate_image = crop_image(image, plate.get("bbox", []))
                # Detect sections within the license plate
                sections = await detect_splitting_sections(plate_image, configs_with_engine_loaded_models, flags)
                result_plate["splitting_sections"] = sections

                for section, section_data in sections.items():
                    try:
                        cropped_section_image = crop_image(plate_image, section_data.get("bbox", []))
                        # Perform OCR on each section
                        result_plate[section] = await ocr_sections(cropped_section_image, section, configs_with_engine_loaded_models)
                    except Exception as e:
                        result_plate[section] = {
                            "error": f"Error during OCR: {str(e)}",
                            "stacktrace": traceback.format_exc(),
                        }
                results.append(result_plate)

            except Exception as e:
                results.append({
                    "error": f"Error processing plate: {str(e)}",
                    'content': license_plates,
                    "stacktrace": traceback.format_exc(),
                })

        return results
    except Exception as e:
        return {
            "error": f"Error in PlateVision: {str(e)}",
            "stacktrace": traceback.format_exc(),
        }


async def detect_license_plates(image, configs_with_engine_loaded_models):
    try:
        detection_engine = configs_with_engine_loaded_models["detection"]["license_plate"].get("engine_instance")
        if detection_engine:
            return await detection_engine.detect(image)
        return []
    except Exception as e:
        return {
            "error": f"Error detecting license plates: {str(e)}",
            "stacktrace": traceback.format_exc(),
        }


async def detect_splitting_sections(plate_image, configs_with_engine_loaded_models, flags):
    try:
        splitting_sections_config = configs_with_engine_loaded_models["detection"].get("splitting_sections", {})
        detection_engine = splitting_sections_config.get("engine_instance")
        if not detection_engine:
            raise ValueError("Splitting sections detection engine is not loaded.")

        splitting_sections = await detection_engine.detect(plate_image)

        map_class_target = {
            0: "region",
            1: "hiragana",
            2: "classification",
            3: "number"
        }

        # Filter sections based on flags
        sections = {}
        for section in splitting_sections:
            target = map_class_target.get(section.get("class_id"))
            if not target:
                continue
            if not flags.get(target, False):
                continue
            sections[target] = section

        return sections
    except Exception as e:
        return {
            "error": f"Error detecting splitting sections: {str(e)}",
            "stacktrace": traceback.format_exc(),
        }


async def ocr_sections(plate_image, section, configs_with_engine_loaded_models):
    try:
        ocr_engine = configs_with_engine_loaded_models["ocr"].get(section, {}).get("engine_instance")
        if ocr_engine:
            return await ocr_engine.recognize_text(plate_image)
        return ""
    except Exception as e:
        return {
            "error": f"Error during OCR for section '{section}': {str(e)}",
            "stacktrace": traceback.format_exc(),
        }


async def load_models(classifying_model, detection_config, ocr_config):
    try:
        return await load_all_models(
            classifying_model,
            detection_config,
            ocr_config,
        )
    except Exception as e:
        return {
            "error": f"Error loading models: {str(e)}",
            "stacktrace": traceback.format_exc(),
        }


def crop_image(image, coordinates):
    try:
        if len(coordinates) == 4:
            return image[coordinates[1]:coordinates[3], coordinates[0]:coordinates[2]]
        raise ValueError(f"Invalid coordinates for cropping: {coordinates}")
    except Exception as e:
        return {
            "error": f"Error cropping image: {str(e)}",
            "stacktrace": traceback.format_exc(),
        }
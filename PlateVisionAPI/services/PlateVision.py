import cv2

from services.detection.license_plate import det_license_plate
from services.detection.section import det_sections
from services.classification.inference import plate_class

from utils.memoryUsage import measure_time_and_memory, monitor_memory

cls_ = {
    0: "region",
    1: "hiragana",
    2: "classification",
    3: "number"
}

@measure_time_and_memory(enabled=True)  # Toggle monitoring with the `enabled` flag
async def PlateVision(
    model_LicensePlateDet,
    model_splitting_sections,
    classification_model,
    classification_scaler,
    image,
    flags,
    measure=False,
    ocr_processor_config=None
):
    """ Detect license plates, segment sections, and perform OCR. """
    print(f"Memory usage before processing: {monitor_memory():.2f} MB")

    async def process():
        # Detect license plates or use the entire image
        detections, annotated_image = detect_plate_or_default(model_LicensePlateDet, image, flags)

        # Process each detection
        results = [
            await process_detection(
                detection,
                annotated_image,
                model_splitting_sections,
                classification_model,
                classification_scaler,
                flags,
                ocr_processor_config
            )
            for detection in detections
        ]

        # Flatten results and return
        flattened_results = [result for result, _ in results]
        final_image = combine_annotated_images([img for _, img in results])
        print(f"Memory usage after processing: {monitor_memory():.2f} MB")
        return flattened_results, final_image

    def detect_plate_or_default(model, image, flags):
        """Detect license plates or use the entire image if only recognition is requested."""
        if not flags["only_recognition"]:
            detections, annotated_image = det_license_plate(model, image)
        else:
            height, width = image.shape[:2]
            detections = [(0, 0, width, height)]
            annotated_image = image
        return detections, annotated_image

    async def process_detection(
        detection,
        image,
        model_splitting_sections,
        classification_model,
        classification_scaler,
        flags,
        ocr_processor_config
    ):
        """Process a single detection."""
        x1, y1, x2, y2 = detection
        cropped = crop_image(image, x1, y1, x2, y2)

        # Detect sections
        sections, LP_cls = det_sections(model_splitting_sections, cropped)
        if not sections:
            print("No sections detected, skipping...")
            return {}, image

        # Classify the license plate
        lp_cls = plate_class(cropped, model=classification_model, scaler=classification_scaler)

        # Process sections
        result = await process_sections(sections, cropped, flags, ocr_processor_config)
        result["class"] = lp_cls

        # Annotate sections on the image
        annotated_image = annotate_sections(image, x1, y1, sections, flags)

        return result, annotated_image

    async def process_sections(sections, cropped, flags, ocr_processor_config):
        """Process each section and perform OCR."""
        results = {}
        for cls_number, s_x1, s_y1, s_x2, s_y2 in sections:
            section_name = cls_[cls_number]
            if not flags.get(section_name, False):
                continue

            section_part_cropped = crop_image(cropped, s_x1, s_y1, s_x2, s_y2)
            engine_instance = ocr_processor_config[section_name]["engine_instance"]

            if engine_instance is None:
                print(f"No OCR engine for {section_name}, skipping...")
                continue

            results[section_name] = await engine_instance.recognize_text(section_part_cropped)
        return results

    def crop_image(image, x1, y1, x2, y2):
        """Crop a region from the image."""
        return image[y1:y2, x1:x2]

    def annotate_sections(image, x1, y1, sections, flags):
        """Annotate sections on the image."""
        annotated_image = image.copy()
        for cls_number, s_x1, s_y1, s_x2, s_y2 in sections:
            if flags["return_image_annotation"]:
                cv2.rectangle(
                    annotated_image,
                    (x1 + s_x1, y1 + s_y1),
                    (x1 + s_x2, y1 + s_y2),
                    (0, 255, 0),
                    2
                )
        return annotated_image

    def combine_annotated_images(images):
        """Combine all annotated images into one."""
        # For simplicity, return the first image (or implement a more complex combination logic)
        return images[0] if images else None
    
    return await process()

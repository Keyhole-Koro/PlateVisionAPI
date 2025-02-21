import cv2
import pytesseract
import asyncio
from functools import partial
from utils.dePatterns import dePatterns, visualize_clusters
from utils.result_whitelist import is_valid_hiragana, is_valid_number, is_valid_classification

def convert_to_binary(image, threshold=128):
    """Converts an image to binary (black and white) based on a threshold."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary

SECTION_MAP = {
    "region": (0, "region"),
    "hiragana": (1, "hiragana"),
    "classification": (2, "digits1"),
    "number": (3, "tlp")
}

async def perform_ocr(tessdata, image, sections, processed_section, de_pattern=False, binary=False, threshold=128):
    """Performs OCR asynchronously on detected sections."""
    loop = asyncio.get_event_loop()
    results = {}

    async def ocr_task(cropped, cls_number):
        if cropped is None or cropped.size == 0:
            return ""
        
        section_type = next((key for key, (label, _) in SECTION_MAP.items() if label == cls_number), None)
        if section_type and not processed_section[section_type]:
            return ""

        _, model = SECTION_MAP[section_type]
        config = f'--oem 1 --psm 6 -l {model}'
        ocr_task = partial(pytesseract.image_to_string, config=config)
        return await loop.run_in_executor(None, ocr_task, cropped)
    
    async def try_ocr(cropped, cls_number, de_pattern):
        if de_pattern:
            cropped = dePatterns(cropped)
        result = await ocr_task(cropped, cls_number)
        return result.split("\n")[0]

    async def try_ocr_with_validation(cropped, cls_number, de_pattern):
        # First attempt with current de_pattern
        result = await try_ocr(cropped, cls_number, de_pattern)
        
        # Check validation based on section type
        section_type = next((key for key, (label, _) in SECTION_MAP.items() if label == cls_number), None)
        is_valid = True
        
        if section_type == "hiragana":
            is_valid = is_valid_hiragana(result)
        elif section_type == "classification":
            is_valid = is_valid_classification(result)
        elif section_type == "number":
            is_valid = is_valid_number(result)
        
        # If validation fails, try with opposite de_pattern
        if not is_valid:
            retry_result = await try_ocr(cropped, cls_number, not de_pattern)
            # Return the retry result if original was invalid
            return retry_result if not is_valid else result
            
        return result

    tasks = []
    for cls_number, x1, y1, x2, y2 in sections:
        if x1 >= x2 or y1 >= y2:
            continue
        
        cropped = image[y1:y2, x1:x2]
        task = asyncio.create_task(try_ocr_with_validation(cropped, cls_number, de_pattern))
        tasks.append((task, cls_number))

    # Gather OCR results
    completed_tasks = [task[0] for task in tasks]
    texts = await asyncio.gather(*completed_tasks)

    # Map results to labels
    for idx, (_, cls_number) in enumerate(tasks):
        cleaned_text = texts[idx].strip()
        if cleaned_text:
            section_type = next((key for key, (label, _) in SECTION_MAP.items() if label == cls_number), None)
            if section_type:
                results[section_type] = cleaned_text

    return results
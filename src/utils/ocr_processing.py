import cv2
import pytesseract
import asyncio
from functools import partial


def convert_to_binary(image, threshold=128):
    """Converts an image to binary (black and white) based on a threshold."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary

label = {
    "region": 0,
    "hiragana": 1,
    "classification": 2,
    "number": 3,
}


async def perform_ocr(tessdata, image, sections, binary=False, threshold=128):
    """Performs OCR asynchronously on detected sections."""
    loop = asyncio.get_event_loop()
    results = {}

    async def ocr_task(cropped, cls_number):
        if cropped is None or cropped.size == 0:
            return ""

        if cls_number == label["classification"]:
            model = "digits1"
        elif cls_number == label["number"]:
            model = "tlp"
        elif cls_number == label["hiragana"]:
            model = "hiragana"
        elif cls_number == label["region"]:
            model = "region"

        # Configure Tesseract OCR settings
        config = f'--oem 1 --psm 6 -l {model}'

        ocr_task = partial(pytesseract.image_to_string, config=config)
        return await loop.run_in_executor(None, ocr_task, cropped)

    tasks = []
    # Iterate over sections and run OCR for each
    for cls_number, x1, y1, x2, y2 in sections:
        if x1 >= x2 or y1 >= y2:
            continue
        cropped = image[y1:y2, x1:x2]
        tasks.append((ocr_task(cropped, cls_number), cls_number))  # Store cls_number with OCR task

    # Gather OCR results
    texts = await asyncio.gather(*[task[0] for task in tasks])

    # Map the OCR results to labels and store them in the results dictionary
    for idx, (task, cls_number) in enumerate(tasks):
        cleaned_text = texts[idx].strip()
        if cleaned_text:  # Only include non-empty results
            results[list(label.keys())[list(label.values()).index(cls_number)]] = cleaned_text

    return results


# Example Usage:
# sections = [(label['hiragana'], 0, 0, 100, 100), (label['classification'], 100, 100, 200, 200)]
# tessdata = "path/to/tessdata"  # Provide the correct path
# result = await perform_ocr(tessdata, image, sections)
# print(result)

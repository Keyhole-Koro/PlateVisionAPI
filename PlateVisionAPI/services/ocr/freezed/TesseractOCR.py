from services.recognition.ocr_interface import OCREngine

import pytesseract

import platform
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

class TesseractOCR(OCREngine):
    def __init__(self, model='jpn'):
        self.model = model

    async def recognize_text(self, image):
        config = f'--oem 1 --psm 6 -l {self.model}'
        return self._format_result(pytesseract.image_to_string(image, config=config))
        
    def _format_result(self, result):
        return result.split("\n")[0]
from abc import ABC, abstractmethod
import pytesseract
from paddleocr import PaddleOCR
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")
PADDLE_DIR = os.path.join(MODEL_DIR, "paddle")

class OCREngine(ABC):
    @abstractmethod
    async def recognize_text(self, image, lang_model, config=None):
        pass

class TesseractOCR(OCREngine):
    def __init__(self, model='jpn'):
        self.model = model

    async def recognize_text(self, image):
        config = f'--oem 1 --psm 6 -l {self.model}'
        return self._format_result(pytesseract.image_to_string(image, config=config))
        
    def _format_result(self, result):
        return result.split("\n")[0]

class PaddleOCREngine(OCREngine):
    def __init__(self, rec_model, lang='japan', use_gpu=False):
        self.ocr = PaddleOCR(
            lang=lang,
            use_gpu=use_gpu,
            model_dir=os.path.join(PADDLE_DIR, rec_model),
            rec_model_dir=os.path.join(PADDLE_DIR, rec_model),
            det=False,
            cls=False,
            use_angle_cls=False
            )
        
    async def recognize_text(self, image):
        return self._format_result(self.ocr.ocr(image, cls=False, det=False))
        
    def _format_result(self, result):
        if result and result[0]:
            return result[0][0][0] # Get text from first detection
        return ""

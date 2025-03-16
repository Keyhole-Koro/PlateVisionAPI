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
    async def recognize_text(self, image, lang_model):
        config = f'--oem 1 --psm 6 -l {lang_model}'
        return pytesseract.image_to_string(image, config=config)

class PaddleOCREngine(OCREngine):
    def __init__(self, rec_model, lang='japan'):
        self.rec_model = rec_model
        self.ocr = PaddleOCR(
            lang=lang,
            use_gpu=False,
            model_dir=os.path.join(PADDLE_DIR, rec_model),
            rec_model_dir=os.path.join(PADDLE_DIR, rec_model),
            det=False,
            cls=False,
            use_angle_cls=False
            )
        
    async def recognize_text(self, image, config=None):
        return self.ocr.ocr(image, cls=False, det=False)
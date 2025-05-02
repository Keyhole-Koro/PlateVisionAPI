from services.recognition.ocr_interface import OCREngine, MODEL_DIR

from paddleocr import PaddleOCR
from torchvision import transforms
import os

PADDLE_DIR = os.path.join(MODEL_DIR, "paddle")

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
        print(result)
        if result and result[0]:
            return result[0][0][0] # Get text from first detection
        return ""

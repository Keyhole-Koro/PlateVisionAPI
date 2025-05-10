from services.ocr.paddle_onnx_handler import PaddleONNXHandler
from services.ocr.interface import OCREngine
from services.ocr.interface import get_model, set_model

class PaddleonnxEngine(OCREngine):
    def __init__(self, model_path, dict_path=None):
        self.model_path = model_path
        self.dict_path = dict_path
        self.model = None
        self.annotated_images = None
        self.ocr_result = None

    async def load_model(self):
        if await self.maybe_load_cached_model():
            self.model = get_model(self.model_path).get('model')
            return 
        
        # Load the PaddleOCR model
        self.model = PaddleONNXHandler(self.model_path, self.dict_path)

        set_model(self.model_path, {"model": self.model})

    async def recognize_text(self, image):
        self.ocr_result = self._format_result(self.model.recognize(image))
        return self.ocr_result
        
    def _format_result(self, result):
        if result and result[0]:
            return result[0][0].replace(' ', '') # Get text from first detection
        return ""

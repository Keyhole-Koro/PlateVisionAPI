from abc import ABC, abstractmethod
import pytesseract
from paddleocr import PaddleOCR
import os
import torch
from torchvision import transforms
from PIL import Image  # Add this import

from utils.char_CNN import CharCNN

import platform
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")
PADDLE_DIR = os.path.join(MODEL_DIR, "paddle")
CHAR_CNN_DIR = os.path.join(MODEL_DIR, "char_cnn")

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
        print(result)
        if result and result[0]:
            return result[0][0][0] # Get text from first detection
        return ""

class CharCNNEngine(OCREngine):
    hiragana_chars = [
        'あ', 'い', 'う', 'え', 'お',
        'か', 'き', 'く', 'け', 'こ',
        'さ', 'し', 'す', 'せ', 'そ',
        'た', 'ち', 'つ', 'て', 'と',
        'な', 'に', 'ぬ', 'ね', 'の',
        'は', 'ひ', 'ふ', 'へ', 'ほ',
        'ま', 'み', 'む', 'め', 'も',
        'や',       'ゆ',       'よ',
        'ら', 'り', 'る', 'れ', 'ろ',
        'わ',                   'を',
                  'ん'
    ]

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert NumPy array to tensor
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64))
    ])
    
    def __init__(self, model_name="hiragana"):
        model = CharCNN(num_classes=len(self.hiragana_chars))
        model_path = os.path.join(CHAR_CNN_DIR, model_name + ".pth")
        # Load the model on the CPU
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model = model

    async def recognize_text(self, image):
        # Use the transform pipeline directly on the NumPy array
        tensor_img = self.transform(image).unsqueeze(0)  # Add batch dimension
        self.model.eval()
        with torch.no_grad():
            output = self.model(tensor_img)
            pred = torch.argmax(output, dim=1).item()
            return self.hiragana_chars[pred]

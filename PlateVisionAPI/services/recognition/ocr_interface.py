from abc import ABC, abstractmethod
import os

from root import BASE_DIR

MODEL_DIR = os.path.join(BASE_DIR, "models")

class OCREngine():
    @abstractmethod
    async def recognize_text(self, image, lang_model, config=None):
        pass
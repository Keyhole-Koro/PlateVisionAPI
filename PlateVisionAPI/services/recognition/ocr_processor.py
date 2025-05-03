import os
import cv2
import asyncio
#from PlateVisionAPI.services.recognition.models.freezed.charCNN import CharCNNEngine
#from PlateVisionAPI.services.recognition.models.freezed.TesseractOCR import TesseractOCR
from services.recognition.models.PaddleOCR import PaddleOCREngine

def convert_to_binary(image, threshold=128):
    """Converts an image to binary (black and white) based on a threshold."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary

os.environ['FLAGS_use_mkldnn'] = '0'  # Disable MKLDNN
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU mode

async def initialize_ocr_engine(cfg):
    """Initialize a single OCR engine based on the configuration."""
    if cfg["engine"] == "-":
        return None
    elif cfg["engine"] == "tesseract":
        #return TesseractOCR(model=cfg["model"])
        pass
    elif cfg["engine"] == "paddle":
        return PaddleOCREngine(rec_model=cfg["model"], lang=cfg["lang"])
    elif cfg["engine"] == "char_cnn":
        #return CharCNNEngine(model_name=cfg["model"])
        pass
    else:
        print(f"Invalid OCR engine: {cfg['engine']}")
        return None

async def OCRProcessorConfig(config):
    print("Initializing OCRProcessor... {}".format(config))

    ocr_processor_config = {}

    # Create tasks to initialize OCR engines in parallel
    tasks = {
        section: asyncio.create_task(initialize_ocr_engine(cfg))
        for section, cfg in config.items()
    }

    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks.values())

    # Map results back to sections
    for section, engine_instance in zip(tasks.keys(), results):
        ocr_processor_config[section] = {"engine_instance": engine_instance}

    return ocr_processor_config
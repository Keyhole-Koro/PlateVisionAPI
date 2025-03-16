import os
import cv2
import pytesseract
import asyncio
from functools import partial
from utils.dePatterns import dePatterns, visualize_clusters
from utils.result_whitelist import is_valid_hiragana, is_valid_number, is_valid_classification
from utils.ocr_interface import TesseractOCR, PaddleOCREngine

def convert_to_binary(image, threshold=128):
    """Converts an image to binary (black and white) based on a threshold."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary

os.environ['FLAGS_use_mkldnn'] = '0'  # Disable MKLDNN
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU mode

def OCRProcessorConfig(config):
    loop = asyncio.get_event_loop()

    print("Initializing OCRProcessor... {}".format(config))

    ocr_processor_config = {}
    
    # Initialize PaddleOCR models with CPU mode
    for section, cfg in config.items():
        ocr_processor = None
        if (cfg["engine"] == "tesseract"):
            ocr_processor = TesseractOCR(model=cfg["model"])
        elif (cfg["engine"] == "paddle"):
            ocr_processor = PaddleOCREngine(rec_model=cfg["model"], lang=cfg["lang"])
        else:
            print(f"Invalid OCR engine: {cfg['engine']}")
            continue

        ocr_processor_config[section] = {
            "engine_instance": ocr_processor
        }

    return ocr_processor_config
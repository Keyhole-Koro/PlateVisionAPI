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


class OCRProcessor:
    def __init__(self, config):
        self.config = config
        self.paddle_models = {}
        self.loop = asyncio.get_event_loop()

        print("Initializing OCRProcessor... {}".format(config))
        
        # Initialize PaddleOCR models with CPU mode
        for section, cfg in config.items():
            if cfg["engine"] == "paddle":
                try:
                    import os
                    os.environ['FLAGS_use_mkldnn'] = '0'  # Disable MKLDNN
                    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU mode
                    
                    self.paddle_models[section] = PaddleOCREngine(
                        lang=cfg["lang"],
                        rec_model=cfg["model"],
                    )
                except Exception as e:
                    print(f"Failed to initialize PaddleOCR for {section}: {e}")

    async def process_section(self, image, section_name):
        config = self.config[section_name]

        if config["engine"] == "tesseract":
            # Use tesseract processing
            return await self._process_tesseract(image, config["model"])
        elif config["engine"] == "paddle":
            # Use paddle processing
            return await self._process_paddle(image, section_name)
        
        return None

    async def _process_paddle(self, image, section_name):
        model = self.paddle_models[section_name]
        result = await model.recognize_text(image)
        return self._format_paddle_result(result)

    async def _process_tesseract(self, image, model):
        config = f'--oem 1 --psm 6 -l {model}'
        ocr_task = partial(pytesseract.image_to_string, config=config)
        result = await self.loop.run_in_executor(None, ocr_task, image)
        return self._format_tesseract_result(result)

    def _format_paddle_result(self, result):
        if result and result[0]:
            return result[0][0][0] # Get text from first detection
        return ""

    def _format_tesseract_result(self, result):
        return result.split("\n")[0]

'''
    async def perform_ocr(self, image, sections, processed_section, de_pattern=False, binary=False, threshold=128):
        results = {}
        
        async def try_ocr_with_validation(cropped, cls_number, de_pattern):
            if cropped is None or cropped.size == 0:
                return ""
                
            section_type = next((key for key, (label, _) in SECTION_MAP.items() if label == cls_number), None)
            if section_type and not processed_section[section_type]:
                return ""
                
            _, model = SECTION_MAP[section_type]
            
            if de_pattern:
                cropped = dePatterns(cropped)
                
            result = await self.process_section(cropped, section_type)
            
            # Validation logic
            is_valid = True
            if section_type == "hiragana":
                is_valid = is_valid_hiragana(result)
            elif section_type == "classification":
                is_valid = is_valid_classification(result)
            elif section_type == "number":
                is_valid = is_valid_number(result)
                
            if not is_valid:
                cropped_retry = dePatterns(cropped) if not de_pattern else cropped
                retry_result = await self.process_section(cropped_retry, section_type)
                return retry_result if not is_valid else result
                
            return result

        tasks = []
        for cls_number, x1, y1, x2, y2 in sections:
            if x1 >= x2 or y1 >= y2:
                continue
            
            cropped = image[y1:y2, x1:x2]
            task = asyncio.create_task(try_ocr_with_validation(cropped, cls_number, de_pattern))
            tasks.append((task, cls_number))

        # Gather OCR results
        completed_tasks = [task[0] for task in tasks]
        texts = await asyncio.gather(*completed_tasks)

        # Map results to labels
        for idx, (_, cls_number) in enumerate(tasks):
            cleaned_text = texts[idx].strip()
            if cleaned_text:
                section_type = next((key for key, (label, _) in SECTION_MAP.items() if label == cls_number), None)
                if section_type:
                    results[section_type] = cleaned_text

        return results
        '''
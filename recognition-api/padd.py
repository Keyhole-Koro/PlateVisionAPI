import os
from paddleocr import PaddleOCR

# Force CPU mode
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Initialize the OCR model with base language and custom model
ocr = PaddleOCR(
    use_gpu=False,
    lang='japan',  # Use base Japanese model
    det=False,     # Disable detection
    cls=False,     # Disable classification
    model_dir='model/paddle/hiragana',  # Path to custom recognition model
    rec_model_dir='model/paddle/hiragana',  # Path to custom recognition model
    rec_char_dict_path='model/paddle/hiragana_dict.txt',  # Path to custom recognition model
    show_log=True
)

# Test image path
image_path = 'materials/image2.jpg'

# Perform OCR
try:
    result = ocr.ocr(image_path, det=False, cls=False, rec=True)
    
    print(f"OCR results: {result}")
    # Print results
    if result:
        for line in result:
            print(f"Text: {line[0][0]}, Confidence: {line[0][1]}")
except Exception as e:
    print(f"Error during OCR: {e}")
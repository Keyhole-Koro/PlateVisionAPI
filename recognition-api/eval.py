import os
import cv2
from DetRec import detect_and_recognize, load_models
from utils.ocr_processing import OCRProcessor
import asyncio

tesseract_engine = "tesseract"
paddle_engine = "paddle"

def load_eval_dataset(eval_dataset_path):
    images = []
    labels = []
    for root, _, files in os.walk(eval_dataset_path):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                image_path = os.path.join(root, file)
                label_path = os.path.splitext(image_path)[0] + '.txt'
                with open(label_path, 'r', encoding='utf-8') as f:
                    label = f.read().splitlines()
                images.append(image_path)
                labels.append(label)
    return images, labels

def calculate_accuracy(predictions, labels):
    correct_hiragana = 0
    correct_classification = 0
    correct_number = 0
    total = len(labels)
    mismatches = []
    
    for i, (pred, label) in enumerate(zip(predictions, labels)):
        # Check hiragana
        pred_hiragana = pred['hiragana'].split(" ")[0].replace("\n", "")
        if pred_hiragana == label[1]:
            correct_hiragana += 1
        else:
            mismatches.append({
                'index': i,
                'type': 'hiragana',
                'predicted': pred_hiragana,
                'actual': label[1]
            })
            
        # Check classification
        pred_class = pred['classification'].replace(" ", "")
        if pred_class == label[0]:
            correct_classification += 1
        else:
            mismatches.append({
                'index': i,
                'type': 'classification',
                'predicted': pred_class,
                'actual': label[0]
            })
            
        # Check number
        pred_number = pred['number'].replace("-", "").replace("・", "")
        actual_number = label[2].replace("-", "").replace("・", "")
        if pred_number == actual_number:
            correct_number += 1
        else:
            mismatches.append({
                'index': i,
                'type': 'number',
                'predicted': pred_number,
                'actual': actual_number
            })
    
    # Print mismatches
    print("\nMismatches:")
    for mismatch in mismatches:
        print(f"Index {mismatch['index']}: {mismatch['type']}")
        print(f"  Predicted: {mismatch['predicted']}")
        print(f"  Actual: {mismatch['actual']}\n")
    
    accuracy_hiragana = correct_hiragana / total if total > 0 else 0
    accuracy_classification = correct_classification / total if total > 0 else 0
    accuracy_number = correct_number / total if total > 0 else 0
    
    return accuracy_hiragana, accuracy_classification, accuracy_number

async def main():
    eval_dataset_path = 'recognition-api/eval_dataset'  # Replace with the actual path
    images, labels = load_eval_dataset(eval_dataset_path)

    model_splitting_sections, model_LicensePlateDet, classification_model, classification_scaler = await load_models()

    recognition_only: bool = False
    annotated_image: bool = True
    hiragana: bool = True
    classification: bool = True
    number: bool = True

    flags = {
        "recognition_only": recognition_only,
        "annotated_image": annotated_image,
        "hiragana": hiragana,
        "classification": classification,
        "number": number,
        "region": False
    }
    measure = False

    ocr_config = {
        "hiragana": {
            "engine": "paddle",
            "model": "hiragana",
            "lang": "japan"
        },
        "classification": {
            "engine": "paddle",
            "model": "classification",
            "lang": "en"
        },
        "number": {
            "engine": "paddle",
            "model": "number",
            "lang": "en"
        }
    }
    
    ocr_processor = OCRProcessor(config=ocr_config)

    predictions = []
    print('Evaluating images...{}'.format(len(images)))
    for image_path in images:
        print(f'Evaluating {image_path}...')

        image = cv2.imread(image_path)
        if image is None:
            print(f'Failed to load image: {image_path}')
            continue

        # Run the pipeline with flags
        prediction, processed_image = await detect_and_recognize(
            model_LicensePlateDet, 
            model_splitting_sections,
            classification_model,
            classification_scaler, 
            image,
            flags,
            measure,
            ocr_processor
        )

        print(prediction)

        if prediction == []:
            predictions.append({
                "hiragana": '',
                "classification": '',
                "number": ''
            })
        else:
            predictions.append({
                "hiragana": prediction[0].get("hiragana", '').split(" ")[0].replace("\t", ""),
                "classification": prediction[0].get("classification", ''),
                "number": prediction[0].get("number", '')
            })
        
    accuracy_hiragana, accuracy_classification, accuracy_number = calculate_accuracy(predictions, labels)
    print(f'Hiragana Accuracy: {accuracy_hiragana * 100:.2f}%')
    print(f'Classification Accuracy: {accuracy_classification * 100:.2f}%')
    print(f'Number Accuracy: {accuracy_number * 100:.2f}%')

if __name__ == '__main__':
    asyncio.run(main())
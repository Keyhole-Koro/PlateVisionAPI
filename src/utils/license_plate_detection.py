import cv2

def detect_license_plate(model_LicensePlateDet, image, debug=False):
    """ Detects license plates in an image using YOLO and applies padding."""
    results = model_LicensePlateDet.predict(image)
    detections = []

    if results[0].boxes:
        for box in results[0].boxes.data:
            x1, y1, x2, y2 = map(int, box[:4].tolist())
            pad_x, pad_y = 20, 20  # Adding some padding
            x1, y1, x2, y2 = max(0, x1 - pad_x), max(0, y1 - pad_y), x2 + pad_x, y2 + pad_y
            detections.append((x1, y1, x2, y2))
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return detections, image

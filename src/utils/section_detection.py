import cv2

def detect_sections(model_splitting_sections, image):
    """ Detects sections within the transformed license plate."""
    results = model_splitting_sections.predict(image)
    sections = []
    if results[0].boxes:
        for idx, box in enumerate(results[0].boxes.data):
            cls_ = results[0].boxes.cls[idx]
            cls_number = int(cls_.item())
            x1, y1, x2, y2 = map(int, box[:4].tolist())
            sections.append((cls_number, x1, y1, x2, y2))
    return sections


import cv2

def detect_sections(model_splitting_sections, image):
    """ Detects sections within the transformed license plate."""
    results = model_splitting_sections.predict(image)
    sections = []
    LP_cls = None
    if results[0].boxes:
        for idx, box in enumerate(results[0].boxes.data):
            cls_ = results[0].boxes.cls[idx]

            LP_cls_ = LicensePlateClass(int(cls_.item()))
            if LP_cls_ != "":
                LP_cls = LP_cls_
                continue

            cls_number = int(cls_.item())
            x1, y1, x2, y2 = map(int, box[:4].tolist())
            sections.append((cls_number, x1, y1, x2, y2))
    return sections, LP_cls

def LicensePlateClass(cls_number):
    """Returns the class label based on the class number."""
    label_cls = {
        5: "light_private",
        6: "private",
        7: "light_commercial",
        8: "commercial",
        9: "designed"
    }
    return label_cls.get(cls_number, "")

def range_xy_sections(sections):
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    for (cls_number, s_x1, s_y1, s_x2, s_y2) in sections:
        min_x = min(min_x, s_x1)
        min_y = min(min_y, s_y1)
        max_x = max(max_x, s_x2)
        max_y = max(max_y, s_y2)
    return min_x, min_y, max_x, max_y

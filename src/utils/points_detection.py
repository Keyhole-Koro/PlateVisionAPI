import cv2
import numpy as np

def detect_4points(model_4PointsDet, image, cropped_image):
    """ Detects the four corner points of the license plate."""
    results = model_4PointsDet.predict(cropped_image)
    points = []
    if results[0].boxes:
        for box in results[0].boxes.data:
            points.append(tuple(map(int, box[:2].tolist())))
    return points

def apply_affine_transform(image, points):
    """ Applies an affine transform to flatten the license plate."""
    if len(points) != 4:
        return image  # Return original if detection fails
    
    width, height = 200, 50  # Standard size for normalization
    src_pts = np.array(points, dtype=np.float32)
    dst_pts = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(image, matrix, (width, height))


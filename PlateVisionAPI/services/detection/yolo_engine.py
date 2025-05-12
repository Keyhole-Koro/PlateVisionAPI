from ultralytics import YOLO
import cv2

from services.detection.interface import DetectionEngine
from services.detection.interface import get_model, set_model

class YOLOEngine(DetectionEngine):
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.annotated_images = None
        self.detection_result = None

    async def load_model(self):
        if await self.maybe_load_cached_model():
            self.model = get_model(self.model_path).get('model')
            return 
        
        self.model = YOLO(self.model_path)

        set_model(self.model_path, {"model": self.model})

    async def detect(self, image, conf=0.25, iou=0.45):
        """Detect objects in the image using YOLOv8."""
        if self.model is None:
            await self.load_model()

        # Convert image to RGB format
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Perform detection
        results = self.model.predict(image_rgb, conf=conf, iou=iou, agnostic_nms=True)
        # Process results
        detections = []
        for result in results:
            for box, score, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                x1, y1, x2, y2 = box.cpu().numpy()
                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(score),
                    "class_id": int(cls)
                })

        self.detection_result = detections
        return detections

    async def get_annotated_images(self, image):
        """Return the annotated image with detected text."""
        if self.detection_result is None:
            await self.detect(image)

        # Create a copy of the image to draw on
        annotated_images = image.copy()

        # Draw bounding boxes and labels on the image
        for detection in self.detection_result:
            bbox = detection["bbox"]
            confidence = detection["confidence"]
            class_id = detection["class_id"]

            # Draw bounding box
            cv2.rectangle(annotated_images, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

            # Draw label
            label = f"Class {class_id}: {confidence:.2f}"
            cv2.putText(annotated_images, label, (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        self.annotated_images = annotated_images
        return annotated_images    
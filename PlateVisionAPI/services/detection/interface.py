from abc import ABC, abstractmethod

loaded_models = {}
def get_model(model_name):
    """Get the loaded model by name."""
    return loaded_models.get(model_name)
def set_model(model_name, model):
    """Set the loaded model by name."""
    loaded_models[model_name] = model
def clear_model(model_name):
    """Clear the loaded model by name."""
    if model_name in loaded_models:
        del loaded_models[model_name]

class DetectionEngine(ABC):
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.annotated_images = None
        self.detection_result = None

    @abstractmethod
    async def load_model(self):
        """Load the detection model from the specified path."""
        pass

    @abstractmethod
    async def detect(self, image):
        pass

    @abstractmethod
    async def get_annotated_images(self, image):
        """Return the annotated image with detected text."""
        pass

    async def maybe_load_cached_model(self):
        """Load model and optimizer from cache if available."""
        cached = get_model(self.model_path)
        if cached:
            self.model = cached.get('model')
            self.optimizer = cached.get('optimizer')
            return True
        return False
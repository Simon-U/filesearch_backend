from abc import ABC, abstractmethod
from typing import Dict, Any, List
from PIL import Image

class BaseClassificationBackend(ABC):
    """Base class for all classification backends"""
    
    def __init__(self, config):
        self.config = config
        self.classification_model = None
        self.processor = None
        
        # Define common categories across all backends
        self.categories: List[str] = [
            "chart", "diagram", "graph", "technical drawing", "infographic", "data visualisation"

            # Non-substantive/decorative categories
            "photograph of person", "photograph of object", "action scene", 
            "movie still", "cinematic imagery", "product photo", 
            "portrait", "artistic image", "nature scene", "landscape",
            "building exterior", "interior space", "stock photo",
            "marketing material", "promotional image", "illustration", 
            "animation frame", "fictional character", "screenshot of video",
            "entertainment media", "decorative background", "logo"
        ]
    
    @abstractmethod
    def initialize(self):
        """Initialize the classification model and processor"""
        pass
    
    @abstractmethod
    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        """Classify an image
        
        Args:
            image: PIL Image to classify
            
        Returns:
            Dict containing at least {'top_category': str, 'confidence': float}
        """
        pass
        
    def to_device(self, inputs):
        """Helper to move inputs to device with memory optimization"""
        if isinstance(inputs, dict):
            return {k: self.to_device(v) for k, v in inputs.items()}
        elif isinstance(inputs, (list, tuple)):
            return type(inputs)(self.to_device(v) for v in inputs)
        elif hasattr(inputs, 'to'):
            return inputs.to(self.config.device)
        return inputs
        
    def _is_substantive_category(self, category: str) -> bool:
        """Check if a category is considered substantive/informative content"""
        substantive_categories = {
            "chart", "diagram", "graph", "technical drawing",
            "data visualization", "infographic"
        }
        return category.lower() in substantive_categories
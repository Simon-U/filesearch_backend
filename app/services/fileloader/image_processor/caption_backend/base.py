from abc import ABC, abstractmethod
from typing import Dict, Any
from PIL import Image

class BaseCaptioningBackend(ABC):
    """Base class for all captioning backends"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.processor = None
    
    @abstractmethod
    def initialize(self):
        """Initialize the captioning model and processor"""
        pass
    
    @abstractmethod
    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        """Generate caption for an image
        
        Args:
            image: PIL Image to caption
            
        Returns:
            Dict containing at least {'caption': str}
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
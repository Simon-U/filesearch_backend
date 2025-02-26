import torch
from typing import Dict, Any
from PIL import Image
from .base import BaseClassificationBackend
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from contextlib import contextmanager
import gc

@contextmanager
def torch_gc_context():
    """Context manager for proper CUDA memory cleanup"""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

class CLIPClassificationBackend(BaseClassificationBackend):
    """CLIP-based image classification backend"""
    
    def initialize(self):
        with torch_gc_context():
            self.processor = AutoProcessor.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir,
                token=self.config.hf_token
            )
            self.model = AutoModelForZeroShotImageClassification.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir,
                token=self.config.hf_token,
                torch_dtype=torch.float16 if self.config.use_half_precision else torch.float32
            ).to(self.config.device)
            
            if self.config.optimize_memory_usage:
                self.model.eval()  # Ensure eval mode
                
            self.categories = [
                "chart", "diagram", "graph", "logo", "decorative element",
                "technical drawing", "data visualization", "infographic"
            ]

    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        try:
            with torch_gc_context(), torch.no_grad():
                inputs = self.processor(
                    images=image,
                    text=self.categories,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77
                )
                inputs = self.to_device(inputs)
                
                outputs = self.model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1)[0].cpu()

                return {
                    "classifications": [
                        {"category": cat, "confidence": prob.item()}
                        for cat, prob in zip(self.categories, probs)
                    ],
                    "top_category": self.categories[probs.argmax().item()],
                    "confidence": probs.max().item()
                }
        except Exception as e:
            return {
                "classifications": [],
                "top_category": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }
import torch
from typing import Dict, Any
from PIL import Image
from .base import BaseCaptioningBackend
from transformers import BlipProcessor, BlipForConditionalGeneration
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

class BlipCaptioningBackend(BaseCaptioningBackend):
    """BLIP image captioning backend implementation"""
    
    def initialize(self):
        with torch_gc_context():
            model_kwargs = {
                'cache_dir': self.config.cache_dir,
                'torch_dtype': torch.float16 if self.config.use_half_precision else torch.float32
            }
            
            if self.config.hf_token:
                model_kwargs['token'] = self.config.hf_token
                
            self.processor = BlipProcessor.from_pretrained(
                self.config.caption_model,
                **model_kwargs
            )
            self.model = BlipForConditionalGeneration.from_pretrained(
                self.config.caption_model,
                **model_kwargs
            ).to(self.config.device)
            
            if self.config.optimize_memory_usage:
                self.model.eval()

    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        try:
            with torch_gc_context(), torch.no_grad():
                inputs = self.processor(image, return_tensors="pt")
                inputs = self.to_device(inputs)
                
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_caption_length,
                    num_beams=1  # Reduce beam search complexity
                )
                caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
                
                return {
                    "caption": caption,
                    "confidence": 1.0
                }
        except Exception as e:
            return {
                "caption": "",
                "error": str(e)
            }
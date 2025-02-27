import torch
import time
import logging
from typing import Dict, Any
from PIL import Image
from .base import BaseCaptioningBackend
from transformers import BlipProcessor, BlipForConditionalGeneration
from contextlib import contextmanager
import gc

# Configure logger
logger = logging.getLogger("image_processor.blip")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

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
        logger.info(f"Initializing BLIP captioning backend with model: {self.config.caption_model}")
        
        start_time = time.time()
        try:
            with torch_gc_context():
                # CRITICAL FIX: Create a clean processor config that only includes
                # parameters that BlipProcessor accepts
                processor_kwargs = {}
                if hasattr(self.config, 'hf_token') and self.config.hf_token:
                    processor_kwargs['token'] = self.config.hf_token
                if hasattr(self.config, 'cache_dir') and self.config.cache_dir:
                    processor_kwargs['cache_dir'] = self.config.cache_dir
                
                # Explicitly exclude problematic parameters
                # Do NOT pass num_query_tokens to the processor
                
                logger.info(f"Loading BLIP processor with kwargs: {processor_kwargs}")
                self.processor = BlipProcessor.from_pretrained(
                    self.config.caption_model,
                    **processor_kwargs
                )
                
                # Model can have more parameters
                model_kwargs = {}
                if hasattr(self.config, 'hf_token') and self.config.hf_token:
                    model_kwargs['token'] = self.config.hf_token
                if hasattr(self.config, 'cache_dir') and self.config.cache_dir:
                    model_kwargs['cache_dir'] = self.config.cache_dir
                if hasattr(self.config, 'use_half_precision') and self.config.use_half_precision:
                    model_kwargs['torch_dtype'] = torch.float16
                
                logger.info(f"Loading BLIP model with kwargs: {model_kwargs}")
                self.model = BlipForConditionalGeneration.from_pretrained(
                    self.config.caption_model,
                    **model_kwargs
                ).to(self.config.device)
                
                if hasattr(self.config, 'optimize_memory_usage') and self.config.optimize_memory_usage:
                    logger.info("Setting model to evaluation mode for memory optimization")
                    self.model.eval()
                
                logger.info(f"BLIP initialization successful")
        except Exception as e:
            logger.error(f"Error during BLIP model initialization: {str(e)}", exc_info=True)
            raise
        
        elapsed_time = time.time() - start_time
        logger.info(f"BLIP captioning backend initialized in {elapsed_time:.2f} seconds")

    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        start_time = time.time()
        try:
            with torch_gc_context(), torch.no_grad():
                # Process the image
                inputs = self.processor(images=image, return_tensors="pt")
                inputs = self.to_device(inputs)
                
                # Generate caption
                max_length = getattr(self.config, 'max_caption_length', 150)
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    num_beams=1  # Use simple greedy search for efficiency
                )
                
                # Decode the generated caption
                caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
                
                total_elapsed = time.time() - start_time
                return {
                    "caption": caption,
                    "confidence": 1.0,  # BLIP doesn't provide confidence scores
                    "processing_time": total_elapsed
                }
        except Exception as e:
            logger.error(f"Error during image captioning: {str(e)}", exc_info=True)
            return {
                "caption": "",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
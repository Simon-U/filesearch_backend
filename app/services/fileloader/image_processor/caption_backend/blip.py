import torch
import time
import logging
from typing import Dict, Any
from PIL import Image
from .base import BaseCaptioningBackend
from transformers import BlipProcessor, BlipForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration
from contextlib import contextmanager
import gc
import os

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
    """BLIP/BLIP2 image captioning backend implementation"""
    
    def initialize(self):
        logger.info(f"Initializing BLIP captioning backend with model: {self.config.caption_model}")
        logger.info(f"Config: {self.config}")
        
        start_time = time.time()
        try:
            with torch_gc_context():
                # Check if this is a BLIP2 model
                model_name = self.config.caption_model
                is_blip2 = "blip2" in model_name.lower() if isinstance(model_name, str) else False
                
                if is_blip2:
                    logger.info("Detected BLIP2 model, using appropriate classes")
                else:
                    logger.info("Using standard BLIP classes")
                
                # Create clean processor config 
                processor_kwargs = {}
                if hasattr(self.config, 'hf_token') and self.config.hf_token:
                    processor_kwargs['token'] = self.config.hf_token
                if hasattr(self.config, 'cache_dir') and self.config.cache_dir:
                    processor_kwargs['cache_dir'] = self.config.cache_dir
                
                # Create model kwargs
                model_kwargs = {}
                if hasattr(self.config, 'hf_token') and self.config.hf_token:
                    model_kwargs['token'] = self.config.hf_token
                if hasattr(self.config, 'cache_dir') and self.config.cache_dir:
                    model_kwargs['cache_dir'] = self.config.cache_dir
                if hasattr(self.config, 'use_half_precision') and self.config.use_half_precision:
                    model_kwargs['torch_dtype'] = torch.float16
                
                # Initialize with the appropriate class based on model type
                if is_blip2:
                    logger.info(f"Loading BLIP2 processor with kwargs: {processor_kwargs}")
                    self.processor = Blip2Processor.from_pretrained(
                        self.config.caption_model,
                        **processor_kwargs
                    )
                    
                    logger.info(f"Loading BLIP2 model with kwargs: {model_kwargs}")
                    self.model = Blip2ForConditionalGeneration.from_pretrained(
                        self.config.caption_model,
                        **model_kwargs
                    ).to(self.config.device)
                else:
                    logger.info(f"Loading BLIP processor with kwargs: {processor_kwargs}")
                    self.processor = BlipProcessor.from_pretrained(
                        self.config.caption_model,
                        **processor_kwargs
                    )
                    
                    logger.info(f"Loading BLIP model with kwargs: {model_kwargs}")
                    self.model = BlipForConditionalGeneration.from_pretrained(
                        self.config.caption_model,
                        **model_kwargs
                    ).to(self.config.device)
                
                if hasattr(self.config, 'optimize_memory_usage') and self.config.optimize_memory_usage:
                    logger.info("Setting model to evaluation mode for memory optimization")
                    self.model.eval()
                
                logger.info(f"Model initialization successful")
        except Exception as e:
            logger.error(f"Error during model initialization: {str(e)}", exc_info=True)
            raise
        
        elapsed_time = time.time() - start_time
        logger.info(f"Captioning backend initialized in {elapsed_time:.2f} seconds")

    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        start_time = time.time()
        logger.info(f"Starting image analysis with model type: {type(self.model).__name__}")
        
        try:
            with torch_gc_context(), torch.no_grad():
                # Check if we're using BLIP2
                is_blip2 = isinstance(self.model, Blip2ForConditionalGeneration)
                
                if is_blip2:
                    # BLIP2 needs different processing and generation
                    # For BLIP2-FLAN-T5, we need to provide a prompt
                    prompt = "Describe this image in detail."
                    
                    inputs = self.processor(images=image, text=prompt, return_tensors="pt")
                    inputs = self.to_device(inputs)
                    
                    max_length = getattr(self.config, 'max_caption_length', 150)
                    
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_length,
                        num_beams=1  # Use simple greedy search for efficiency
                    )
                    
                    caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
                else:
                    # Standard BLIP processing
                    inputs = self.processor(images=image, return_tensors="pt")
                    inputs = self.to_device(inputs)
                    
                    max_length = getattr(self.config, 'max_caption_length', 150)
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_length,
                        num_beams=1
                    )
                    
                    caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
                
                total_elapsed = time.time() - start_time
                logger.info(f"Generated caption: {caption}")
                return {
                    "caption": caption,
                    "confidence": 1.0,
                    "processing_time": total_elapsed
                }
        except Exception as e:
            logger.error(f"Error during image captioning: {str(e)}", exc_info=True)
            return {
                "caption": "",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
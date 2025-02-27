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
        logger.info(f"Device: {self.config.device}, Half precision: {self.config.use_half_precision}")
        
        start_time = time.time()
        try:
            with torch_gc_context():
                # Only include valid parameters for the model and processor
                model_kwargs = {
                    'cache_dir': self.config.cache_dir
                }
                
                # Add token if provided
                if self.config.hf_token:
                    model_kwargs['token'] = self.config.hf_token
                    logger.info("Using provided HuggingFace token for model access")
                
                # Load processor first
                logger.info(f"Loading processor for model: {self.config.caption_model}")
                self.processor = BlipProcessor.from_pretrained(
                    self.config.caption_model,
                    **model_kwargs
                )
                
                # Set up model kwargs, including dtype
                model_kwargs['torch_dtype'] = torch.float16 if self.config.use_half_precision else torch.float32
                
                logger.info(f"Loading model: {self.config.caption_model}")
                self.model = BlipForConditionalGeneration.from_pretrained(
                    self.config.caption_model,
                    **model_kwargs
                ).to(self.config.device)
                
                if self.config.optimize_memory_usage:
                    logger.info("Setting model to evaluation mode for memory optimization")
                    self.model.eval()
                
                # Log model size and memory usage
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
                    memory_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
                    logger.info(f"CUDA memory allocated: {memory_allocated:.2f} MB")
                    logger.info(f"CUDA memory reserved: {memory_reserved:.2f} MB")
                
                # Log number of parameters
                total_params = sum(p.numel() for p in self.model.parameters())
                logger.info(f"Model loaded with {total_params:,} parameters")
        except Exception as e:
            logger.error(f"Error during model initialization: {str(e)}", exc_info=True)
            raise
        
        elapsed_time = time.time() - start_time
        logger.info(f"BLIP captioning backend initialized in {elapsed_time:.2f} seconds")

    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        start_time = time.time()
        logger.info(f"Starting image analysis with BLIP. Image size: {image.size}, mode: {image.mode}")
        
        try:
            with torch_gc_context(), torch.no_grad():
                # Process the image
                inputs = self.processor(images=image, return_tensors="pt")
                inputs = self.to_device(inputs)
                
                # Log input tensor shapes for debugging
                input_shapes = {k: v.shape for k, v in inputs.items() if hasattr(v, 'shape')}
                logger.info(f"Input tensor shapes: {input_shapes}")
                
                # Generate caption
                logger.info(f"Generating caption with max_new_tokens={self.config.max_caption_length}")
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_caption_length,
                    num_beams=1  # Use simple greedy search for efficiency
                )
                
                # Decode the generated caption
                caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
                
                # Log memory usage after generation
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
                    memory_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
                    logger.info(f"CUDA memory allocated: {memory_allocated:.2f} MB")
                    logger.info(f"CUDA memory reserved: {memory_reserved:.2f} MB")
                
                total_elapsed = time.time() - start_time
                logger.info(f"Total processing time: {total_elapsed:.2f} seconds")
                logger.info(f"Generated caption: {caption}")
                
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
import torch
import time
import logging
from typing import Dict, Any
from PIL import Image
from .base import BaseClassificationBackend
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from contextlib import contextmanager
import gc

# Configure logger
logger = logging.getLogger("image_processor.clip")
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

class CLIPClassificationBackend(BaseClassificationBackend):
    """CLIP-based image classification backend"""
    
    def initialize(self):
        logger.info(f"Initializing CLIP classification backend with model: {self.config.classification_model}")
        logger.info(f"Device: {self.config.device}, Half precision: {self.config.use_half_precision}")
        
        start_time = time.time()
        try:
            with torch_gc_context():
                logger.info(f"Loading CLIP processor for model: {self.config.classification_model}")
                self.processor = AutoProcessor.from_pretrained(
                    self.config.classification_model,
                    cache_dir=self.config.cache_dir,
                    token=self.config.hf_token
                )
                logger.info("Processor loaded successfully")
                
                model_start_time = time.time()
                logger.info(f"Loading CLIP model: {self.config.classification_model}")
                model_dtype = torch.float16 if self.config.use_half_precision else torch.float32
                logger.info(f"Using model data type: {model_dtype}")
                
                self.model = AutoModelForZeroShotImageClassification.from_pretrained(
                    self.config.classification_model,
                    cache_dir=self.config.cache_dir,
                    token=self.config.hf_token,
                    torch_dtype=model_dtype
                ).to(self.config.device)
                
                model_load_time = time.time() - model_start_time
                logger.info(f"Model loaded in {model_load_time:.2f} seconds")
                
                if self.config.optimize_memory_usage:
                    logger.info("Setting model to evaluation mode for memory optimization")
                    self.model.eval()  # Ensure eval mode
                
                # Log model size and memory usage
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
                    memory_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
                    logger.info(f"CUDA memory allocated: {memory_allocated:.2f} MB")
                    logger.info(f"CUDA memory reserved: {memory_reserved:.2f} MB")
                
                # Log number of parameters
                total_params = sum(p.numel() for p in self.model.parameters())
                logger.info(f"Model loaded with {total_params:,} parameters")
                
                # Set up categories for zero-shot classification
                self.categories = [
                    "chart", "diagram", "graph", "logo", "decorative element",
                    "technical drawing", "data visualization", "infographic"
                ]
                logger.info(f"Initialized with {len(self.categories)} categories: {', '.join(self.categories)}")
        except Exception as e:
            logger.error(f"Error during model initialization: {str(e)}", exc_info=True)
            raise
        
        elapsed_time = time.time() - start_time
        logger.info(f"CLIP classification backend initialized in {elapsed_time:.2f} seconds")

    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            with torch_gc_context(), torch.no_grad():
                # Prepare inputs
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
                

                # Get top category and confidence
                top_idx = probs.argmax().item()
                top_category = self.categories[top_idx]
                top_confidence = probs[top_idx].item()
                
                # Create classifications list
                classifications = [
                    {"category": cat, "confidence": prob.item()}
                    for cat, prob in zip(self.categories, probs)
                ]
                
                # Sort by confidence (highest first)
                classifications.sort(key=lambda x: x["confidence"], reverse=True)

                output = {
                    "classifications": classifications,
                    "top_category": top_category,
                    "confidence": top_confidence,
                }
                logger.info(f"Clip output: {output}")
                return output
        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"Error during image classification: {str(e)}", exc_info=True)
            return {
                "classifications": [],
                "top_category": "unknown",
                "confidence": 0.0,
                "error": str(e),
                "processing_time": error_time
            }
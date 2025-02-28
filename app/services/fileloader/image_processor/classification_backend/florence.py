import torch
import time
import logging
from typing import Dict, Any
from PIL import Image
from .base import BaseClassificationBackend
from transformers import AutoProcessor, AutoModelForImageClassification
from contextlib import contextmanager
import gc

# Configure logger
logger = logging.getLogger("image_processor.florence2")
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

class Florence2ClassificationBackend(BaseClassificationBackend):
    """Florence2-based image classification backend"""
    
    def initialize(self):
        logger.info(f"Initializing Florence2 classification backend with model: {self.config.classification_model}")
        logger.info(f"Device: {self.config.device}, Half precision: {self.config.use_half_precision}")
        
        start_time = time.time()
        try:
            with torch_gc_context():
                logger.info(f"Loading Florence2 processor for model: {self.config.classification_model}")
                self.processor = AutoProcessor.from_pretrained(
                    self.config.classification_model,
                    cache_dir=self.config.cache_dir,
                    token=self.config.hf_token,
                    trust_remote_code=True,
                )
                logger.info("Processor loaded successfully")
                
                model_start_time = time.time()
                logger.info(f"Loading Florence2 model: {self.config.classification_model}")
                model_dtype = torch.float16 if self.config.use_half_precision else torch.float32
                logger.info(f"Using model data type: {model_dtype}")
                
                # Florence2 uses AutoModelForImageClassification for zero-shot classification
                self.model = AutoModelForImageClassification.from_pretrained(
                    self.config.classification_model,
                    cache_dir=self.config.cache_dir,
                    token=self.config.hf_token,
                    torch_dtype=model_dtype,
                    trust_remote_code=True
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
                
                # Log the categories defined in the base class
                logger.info(f"Initialized with {len(self.categories)} categories: {', '.join(self.categories)}")
                
                # Map Florence2 outputs to our categories if needed
                # This will depend on the specific Florence2 model's output labels
                self.output_mapping = self._setup_output_mapping()
                
        except Exception as e:
            logger.error(f"Error during model initialization: {str(e)}", exc_info=True)
            raise
        
        elapsed_time = time.time() - start_time
        logger.info(f"Florence2 classification backend initialized in {elapsed_time:.2f} seconds")
    
    def _setup_output_mapping(self):
        """Set up mapping between Florence2 model outputs and our categories.
        This may need customization based on the specific Florence2 model used."""
        # Default implementation assumes direct mapping
        # For a custom Florence2 model, you might need to map model-specific labels to your categories
        return {}
    
    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            with torch_gc_context(), torch.no_grad():
                # Prepare inputs
                inputs = self.processor(
                    images=image,
                    return_tensors="pt"
                )
                inputs = self.to_device(inputs)
                
                # Forward pass through the model
                outputs = self.model(**inputs)
                
                # Get logits and convert to probabilities
                # For Florence2, the output format may be different from CLIP
                # Adjust as needed based on the specific model
                if hasattr(outputs, "logits"):
                    probs = torch.softmax(outputs.logits[0], dim=0).cpu()
                elif hasattr(outputs, "image_embeds"):
                    # If the model returns embeddings, handle accordingly
                    # This is just a placeholder - implementation will depend on model specifics
                    logger.warning("Model returns embeddings rather than classification logits")
                    probs = torch.ones(len(self.categories)) / len(self.categories)
                else:
                    logger.warning("Unexpected model output format")
                    probs = torch.ones(len(self.categories)) / len(self.categories)
                
                # Map output indices to our categories if needed
                if self.output_mapping:
                    # Handle mapping logic here
                    pass
                
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
                
                processing_time = time.time() - start_time
                output = {
                    "classifications": classifications,
                    "top_category": top_category,
                    "confidence": top_confidence,
                    "processing_time": processing_time
                }
                
                logger.info(f"Florence2 top category: {top_category} with confidence: {top_confidence:.4f}")
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
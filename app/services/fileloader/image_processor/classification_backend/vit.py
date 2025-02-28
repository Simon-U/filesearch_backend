import torch
import time
import logging
from typing import Dict, Any
from PIL import Image
from .base import BaseClassificationBackend
from transformers import ViTImageProcessor, ViTForImageClassification
from contextlib import contextmanager
import gc

# Configure logger
logger = logging.getLogger("image_processor.vit")
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

class ViTClassificationBackend(BaseClassificationBackend):
    """Vision Transformer (ViT)-based image classification backend"""
    
    def initialize(self):
        logger.info(f"Initializing ViT classification backend with model: {self.config.classification_model}")
        logger.info(f"Device: {self.config.device}, Half precision: {self.config.use_half_precision}")
        
        start_time = time.time()
        try:
            with torch_gc_context():
                logger.info(f"Loading ViT processor for model: {self.config.classification_model}")
                self.processor = ViTImageProcessor.from_pretrained(
                    self.config.classification_model,
                    cache_dir=self.config.cache_dir,
                    token=self.config.hf_token
                )
                logger.info("Processor loaded successfully")
                
                model_start_time = time.time()
                logger.info(f"Loading ViT model: {self.config.classification_model}")
                model_dtype = torch.float16 if self.config.use_half_precision else torch.float32
                logger.info(f"Using model data type: {model_dtype}")
                
                # Load ViT model for image classification
                self.model = ViTForImageClassification.from_pretrained(
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
                
                # Log the categories defined in the base class
                logger.info(f"Initialized with {len(self.categories)} categories: {', '.join(self.categories)}")
                
                # Set up class mapping for ViT model's ImageNet classes to our categories
                # We'll use this to map ViT's ImageNet predictions to our technical graphics categories
                self._setup_category_mapping()
                
        except Exception as e:
            logger.error(f"Error during model initialization: {str(e)}", exc_info=True)
            raise
        
        elapsed_time = time.time() - start_time
        logger.info(f"ViT classification backend initialized in {elapsed_time:.2f} seconds")

    def _setup_category_mapping(self):
        """
        Set up mapping between ImageNet classes and our technical graphics categories.
        This is needed because ViT is pre-trained on ImageNet, which doesn't directly 
        align with our categories (chart, diagram, graph, etc.)
        
        We'll use embeddings similarity in the analyze method.
        """
        logger.info("Setting up category mapping for ViT model")
        self.category_mapping = {}
        
        # For ViT, we'll use a different approach since the pre-trained ImageNet
        # classes don't directly map to our technical graphics categories.
        # Instead, we'll compute embeddings for our categories in the analyze method.

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
                
                # Get image embeddings from the ViT model
                outputs = self.model(**inputs, output_hidden_states=True)
                image_embedding = outputs.hidden_states[-1][:, 0]  # CLS token embedding
                
                # Get classification logits
                logits = outputs.logits
                
                # Since ViT is trained on ImageNet and not our specific categories,
                # we need to adapt its outputs to our classification task.
                # Approach: Compute similarity between image embedding and text embeddings
                # of our categories
                
                # For now, we'll use a simplified approach: map the top ImageNet classes
                # to our categories based on some heuristics
                
                # Get top predicted ImageNet classes
                top_k_indices = torch.topk(logits[0], k=5).indices.cpu().numpy()
                top_k_classes = [self.model.config.id2label[idx] for idx in top_k_indices]
                
                # Map ImageNet predictions to our categories
                # This is a simplified mapping and should be improved with proper zero-shot or fine-tuning
                category_scores = {category: 0.0 for category in self.categories}
                
                # Look for specific keywords in the top ImageNet predictions
                # that might correlate with our technical graphics categories
                for i, class_name in enumerate(top_k_classes):
                    score = 1.0 - (i * 0.15)  # Simple decay for lower-ranked predictions
                    
                    # Chart-related keywords
                    if any(keyword in class_name.lower() for keyword in ["chart", "graph", "plot", "diagram", "display", "screen", "monitor"]):
                        for cat in ["chart", "graph", "data visualization"]:
                            if cat in self.categories:
                                category_scores[cat] += score
                    
                    # Technical drawings
                    if any(keyword in class_name.lower() for keyword in ["technical", "blueprint", "schematic", "engineering", "mechanical", "drawing", "sketch"]):
                        for cat in ["technical drawing", "diagram"]:
                            if cat in self.categories:
                                category_scores[cat] += score
                    
                    # Infographics
                    if any(keyword in class_name.lower() for keyword in ["information", "poster", "presentation", "slide", "projection"]):
                        for cat in ["infographic", "data visualization"]:
                            if cat in self.categories:
                                category_scores[cat] += score
                    
                    # Logos
                    if any(keyword in class_name.lower() for keyword in ["logo", "brand", "symbol", "emblem", "trademark", "icon"]):
                        if "logo" in self.categories:
                            category_scores["logo"] += score
                    
                    # Decorative elements
                    if any(keyword in class_name.lower() for keyword in ["art", "decoration", "ornament", "pattern", "design", "artistic"]):
                        if "decorative element" in self.categories:
                            category_scores["decorative element"] += score
                
                # Ensure minimum score for all categories
                min_score = 0.05
                for cat in category_scores:
                    if category_scores[cat] < min_score:
                        category_scores[cat] = min_score
                
                # Normalize scores to sum to 1
                total_score = sum(category_scores.values())
                if total_score > 0:
                    for cat in category_scores:
                        category_scores[cat] /= total_score
                
                # Create classifications list
                classifications = [
                    {"category": cat, "confidence": score}
                    for cat, score in category_scores.items()
                ]
                
                # Sort by confidence (highest first)
                classifications.sort(key=lambda x: x["confidence"], reverse=True)
                
                # Get top category
                top_category = classifications[0]["category"]
                top_confidence = classifications[0]["confidence"]
                
                processing_time = time.time() - start_time
                output = {
                    "classifications": classifications,
                    "top_category": top_category,
                    "confidence": top_confidence,
                    "processing_time": processing_time,
                    "imagenet_classes": top_k_classes[:3]  # Include top ImageNet classes for debugging
                }
                
                logger.info(f"ViT top category: {top_category} with confidence: {top_confidence:.4f}")
                logger.info(f"Top ImageNet classes: {', '.join(top_k_classes[:3])}")
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
import torch
import time
import logging
from typing import Dict, Any, List
from PIL import Image
from .base import BaseClassificationBackend
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from contextlib import contextmanager
import gc

# Configure logger
logger = logging.getLogger("image_processor.deit")
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

class DEiTClassificationBackend(BaseClassificationBackend):
    """DeiT-based image classification backend"""
    
    def initialize(self):
        start_time = time.time()
        logger.info("Initializing DEiT classification backend")
        
        try:
            with torch_gc_context():
                # Set default model if not specified
                model_name = self.config.model_name
                if "deit" not in model_name.lower():
                    # Default to DeiT base model if a specific DeiT model isn't specified
                    model_name = "facebook/deit-base-patch16-224"
                    logger.info(f"No DeiT model specified, using default: {model_name}")
                else:
                    logger.info(f"Using specified model: {model_name}")
                
                logger.info(f"Device: {self.config.device}, Half precision: {self.config.use_half_precision}")
                
                # Load processor
                logger.info(f"Loading feature extractor for model: {model_name}")
                processor_start = time.time()
                self.processor = AutoFeatureExtractor.from_pretrained(
                    model_name,
                    cache_dir=self.config.cache_dir,
                    token=self.config.hf_token
                )
                processor_time = time.time() - processor_start
                logger.info(f"Feature extractor loaded in {processor_time:.2f} seconds")
                
                # Load model
                logger.info(f"Loading model: {model_name}")
                model_start = time.time()
                model_dtype = torch.float16 if self.config.use_half_precision else torch.float32
                logger.info(f"Using model data type: {model_dtype}")
                
                self.model = AutoModelForImageClassification.from_pretrained(
                    model_name,
                    cache_dir=self.config.cache_dir,
                    token=self.config.hf_token,
                    torch_dtype=model_dtype
                ).to(self.config.device)
                
                model_time = time.time() - model_start
                logger.info(f"Model loaded in {model_time:.2f} seconds")
                
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
                
                # Map ImageNet classes to document-relevant categories
                # These are the categories we care about for document analysis
                self.target_categories = [
                    "chart", "diagram", "graph", "logo", "decorative element",
                    "technical drawing", "data visualization", "infographic"
                ]
                logger.info(f"Target categories initialized: {', '.join(self.target_categories)}")
                
                # Initialize a mapping from model outputs to our target categories
                logger.info("Initializing class mapping")
                mapping_start = time.time()
                self.class_mapping = self._initialize_class_mapping()
                mapping_time = time.time() - mapping_start
                logger.info(f"Class mapping initialized in {mapping_time:.2f} seconds with {len(self.class_mapping)} mappings")
                
                init_time = time.time() - start_time
                logger.info(f"DEiT classification backend initialization completed in {init_time:.2f} seconds")
        
        except Exception as e:
            logger.error(f"Error during DEiT initialization: {str(e)}", exc_info=True)
            raise
    
    def _initialize_class_mapping(self) -> Dict[int, str]:
        """Map model's output classes to our target categories"""
        # This is a simplified mapping - in production, you'd want a more comprehensive one
        # based on the specific model's classes
        
        # For DeiT, we'll map certain ImageNet classes to our document categories
        # Note: These are approximate mappings and should be refined based on testing
        mapping = {}
        
        # Examples of mappings from ImageNet classes to document categories
        # Map "bar code", "digital display", "monitor" etc. to "chart" or "graph"
        chart_graph_classes = [716, 782, 664, 508, 720]  # Example ImageNet class indices
        for idx in chart_graph_classes:
            mapping[idx] = "chart"
        
        # Map "projection screen", "web site", "menu" etc. to "diagram"
        diagram_classes = [482, 920, 922]  # Example ImageNet class indices
        for idx in diagram_classes:
            mapping[idx] = "diagram"
        
        # Map "notebook", "book jacket", "envelope" etc. to "decorative element"
        decorative_classes = [747, 921, 549]  # Example ImageNet class indices
        for idx in decorative_classes:
            mapping[idx] = "decorative element"
        
        logger.debug(f"Created mappings: chart/graph({len(chart_graph_classes)}), " + 
                    f"diagram({len(diagram_classes)}), decorative({len(decorative_classes)})")
        
        return mapping
        
    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        start_time = time.time()
        logger.info(f"Starting DEiT image classification. Image size: {image.size}, mode: {image.mode}")
        
        try:
            with torch_gc_context(), torch.no_grad():
                # Prepare image
                preprocess_start = time.time()
                logger.info("Preprocessing image")
                inputs = self.processor(images=image, return_tensors="pt")
                inputs = self.to_device(inputs)
                
                # Log input tensor shapes for debugging
                input_shapes = {k: v.shape for k, v in inputs.items() if hasattr(v, 'shape')}
                logger.info(f"Input tensor shapes: {input_shapes}")
                
                preprocess_time = time.time() - preprocess_start
                logger.info(f"Image preprocessing completed in {preprocess_time:.2f} seconds")
                
                # Run inference
                inference_start = time.time()
                logger.info("Running model inference")
                outputs = self.model(**inputs)
                
                # Get probabilities
                probs = outputs.logits.softmax(dim=1)[0].cpu()
                predicted_class_idx = probs.argmax().item()
                top_confidence = probs[predicted_class_idx].item()
                
                inference_time = time.time() - inference_start
                logger.info(f"Model inference completed in {inference_time:.2f} seconds")
                logger.info(f"Top predicted class index: {predicted_class_idx}, confidence: {top_confidence:.4f}")
                
                # Map to our document categories using our mapping
                postprocess_start = time.time()
                
                if predicted_class_idx in self.class_mapping:
                    top_category = self.class_mapping[predicted_class_idx]
                    logger.info(f"Mapped class index {predicted_class_idx} to category: {top_category}")
                else:
                    # If not in our mapping, check if it's a substantive image based on threshold
                    substantive_threshold = 0.6  # Confidence threshold for images to be considered substantive
                    logger.info(f"Class index {predicted_class_idx} not in mapping, using confidence threshold: {substantive_threshold}")
                    
                    if top_confidence > substantive_threshold:
                        # High confidence in a class we didn't map - likely an image with content
                        top_category = "data visualization"  # Default to a substantive category
                        logger.info(f"High confidence ({top_confidence:.4f}) unmarked class, assigning to: {top_category}")
                    else:
                        # Low confidence - likely a decorative image
                        top_category = "decorative element"
                        logger.info(f"Low confidence ({top_confidence:.4f}) unmarked class, assigning to: {top_category}")
                
                # Create a list of classifications for our target categories
                logger.info("Creating classification results for target categories")
                classifications = []
                for category in self.target_categories:
                    # Assign confidence based on whether it matches the top category
                    if category == top_category:
                        confidence = top_confidence
                    else:
                        confidence = 0.1  # Assign low confidence to non-matched categories
                    
                    classifications.append({
                        "category": category,
                        "confidence": confidence
                    })
                
                postprocess_time = time.time() - postprocess_start
                logger.info(f"Post-processing completed in {postprocess_time:.2f} seconds")
                
                # Log memory usage after inference
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
                    memory_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
                    logger.info(f"CUDA memory allocated: {memory_allocated:.2f} MB")
                    logger.info(f"CUDA memory reserved: {memory_reserved:.2f} MB")
                
                total_time = time.time() - start_time
                logger.info(f"Total DEiT classification completed in {total_time:.2f} seconds")
                output  = {
                    "classifications": classifications,
                    "top_category": top_category,
                    "confidence": top_confidence,
                    "model_class_idx": predicted_class_idx,
                    "processing_time": {
                        "total": total_time,
                        "preprocessing": preprocess_time,
                        "inference": inference_time,
                        "postprocessing": postprocess_time
                    }
                }
                logger.info(f"Output: {output}")
                return output
                
        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"DEiT classification error: {str(e)}", exc_info=True)
            return {
                "classifications": [],
                "top_category": "unknown",
                "confidence": 0.0,
                "error": str(e),
                "processing_time": error_time
            }
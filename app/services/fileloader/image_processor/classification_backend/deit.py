import torch
from typing import Dict, Any, List
from PIL import Image
from .base import BaseClassificationBackend
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
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

class DEiTClassificationBackend(BaseClassificationBackend):
    """DeiT-based image classification backend"""
    
    def initialize(self):
        with torch_gc_context():
            # Set default model if not specified
            model_name = self.config.model_name
            if "deit" not in model_name.lower():
                # Default to DeiT base model if a specific DeiT model isn't specified
                model_name = "facebook/deit-base-patch16-224"
            
            self.processor = AutoFeatureExtractor.from_pretrained(
                model_name,
                cache_dir=self.config.cache_dir,
                token=self.config.hf_token
            )
            
            self.model = AutoModelForImageClassification.from_pretrained(
                model_name,
                cache_dir=self.config.cache_dir,
                token=self.config.hf_token,
                torch_dtype=torch.float32 if not self.config.use_half_precision else torch.float16
            ).to(self.config.device)
            
            if self.config.optimize_memory_usage:
                self.model.eval()
            
            # Map ImageNet classes to document-relevant categories
            # These are the categories we care about for document analysis
            self.target_categories = [
                "chart", "diagram", "graph", "logo", "decorative element",
                "technical drawing", "data visualization", "infographic"
            ]
            
            # Initialize a mapping from model outputs to our target categories
            # This is a simplified approach - in production, you'd want a more
            # sophisticated mapping based on semantically similar ImageNet classes
            self.class_mapping = self._initialize_class_mapping()
    
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
        
        return mapping
        
    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        try:
            with torch_gc_context(), torch.no_grad():
                # Prepare image
                inputs = self.processor(images=image, return_tensors="pt")
                inputs = self.to_device(inputs)
                
                # Run inference
                outputs = self.model(**inputs)
                
                # Get probabilities
                probs = outputs.logits.softmax(dim=1)[0].cpu()
                predicted_class_idx = probs.argmax().item()
                
                # Map to our document categories using our mapping
                if predicted_class_idx in self.class_mapping:
                    top_category = self.class_mapping[predicted_class_idx]
                else:
                    # If not in our mapping, check if it's a substantive image based on threshold
                    substantive_threshold = 0.6  # Confidence threshold for images to be considered substantive
                    
                    if probs[predicted_class_idx].item() > substantive_threshold:
                        # High confidence in a class we didn't map - likely an image with content
                        top_category = "data visualization"  # Default to a substantive category
                    else:
                        # Low confidence - likely a decorative image
                        top_category = "decorative element"
                
                # Create a list of classifications for our target categories
                # This is a simplified approach - in production, you'd map model outputs to category scores
                classifications = []
                for category in self.target_categories:
                    # Assign confidence based on whether it matches the top category
                    if category == top_category:
                        confidence = probs[predicted_class_idx].item()
                    else:
                        confidence = 0.1  # Assign low confidence to non-matched categories
                    
                    classifications.append({
                        "category": category,
                        "confidence": confidence
                    })
                
                return {
                    "classifications": classifications,
                    "top_category": top_category,
                    "confidence": probs[predicted_class_idx].item(),
                    "model_class_idx": predicted_class_idx
                }
                
        except Exception as e:
            import logging
            logging.getLogger("fileloader").error(f"DEiT classification error: {str(e)}")
            return {
                "classifications": [],
                "top_category": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }
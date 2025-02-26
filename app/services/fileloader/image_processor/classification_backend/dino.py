import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple
from PIL import Image
import numpy as np
from .base import BaseClassificationBackend
from transformers import ViTImageProcessor, AutoModel
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

class DINOClassificationBackend(BaseClassificationBackend):
    """DINO V2-based image classification backend"""
    
    def initialize(self):
        with torch_gc_context():
            # Set default model if not specified
            model_name = self.config.model_name
            if "dinov2" not in model_name.lower():
                # Default to DINO V2 base model if a specific DINO model isn't specified
                model_name = "facebook/dinov2-base"
            
            # Use ViTImageProcessor instead of AutoFeatureExtractor for DINO V2
            # This is more reliable as DINO V2 is based on the ViT architecture
            self.processor = ViTImageProcessor.from_pretrained(
                "facebook/dinov2-base",  # Use ViT processor which is compatible
                cache_dir=self.config.cache_dir,
                token=self.config.hf_token
            )
            
            self.model = AutoModel.from_pretrained(
                model_name,
                cache_dir=self.config.cache_dir,
                token=self.config.hf_token,
                torch_dtype=torch.float32 if not self.config.use_half_precision else torch.float16
            ).to(self.config.device)
            
            if self.config.optimize_memory_usage:
                self.model.eval()
            
            # Define categories we're interested in for document analysis
            self.categories = [
                "chart", "diagram", "graph", "logo", "decorative element",
                "technical drawing", "data visualization", "infographic"
            ]
            
            # Since DINO is a feature extractor and doesn't have built-in categories,
            # we use pre-defined feature centroids for each category
            # In a real implementation, these would be learned from a dataset
            self.feature_centroids = self._initialize_feature_centroids()
    
    def _initialize_feature_centroids(self) -> Dict[str, torch.Tensor]:
        """
        Initialize feature centroids for each category.
        
        In a real implementation, these would be learned from a dataset of images
        for each category. For this example, we're using random initializations.
        """
        # This is just a placeholder - in production, you would use actual centroids
        # learned from a dataset of document images
        feature_dim = self.model.config.hidden_size
        centroids = {}
        
        # Generate random centroids for demonstration purposes
        # In production, these would be replaced with actual learned centroids
        for category in self.categories:
            # Create a random vector for each category - this is just a placeholder
            # In a real implementation, this would be a centroid learned from examples
            centroids[category] = torch.randn(feature_dim).to(self.config.device)
            
            # Normalize the centroid
            centroids[category] = F.normalize(centroids[category], p=2, dim=0)
        
        return centroids
    
    def _compute_similarity(self, image_features: torch.Tensor) -> Dict[str, float]:
        """Compute cosine similarity between image features and category centroids"""
        similarities = {}
        
        # Normalize image features
        image_features = F.normalize(image_features, p=2, dim=0)
        
        for category, centroid in self.feature_centroids.items():
            # Compute cosine similarity
            similarity = torch.dot(image_features, centroid).item()
            similarities[category] = similarity
        
        return similarities
    
    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        try:
            with torch_gc_context(), torch.no_grad():
                # Prepare image
                inputs = self.processor(images=image, return_tensors="pt")
                inputs = self.to_device(inputs)
                
                # Extract features using DINO V2
                outputs = self.model(**inputs)
                
                # Get the CLS token embedding as the image representation
                image_features = outputs.last_hidden_state[:, 0, :].squeeze(0)
                
                # Compute similarity to each category
                similarities = self._compute_similarity(image_features)
                
                # Sort categories by similarity
                sorted_categories = sorted(
                    similarities.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                # Get top category and confidence
                top_category, confidence = sorted_categories[0]
                
                # Scale confidence to [0,1] range
                # This is a simplified approach - in a real implementation,
                # you would use a more sophisticated confidence calculation
                min_conf = min(similarities.values())
                max_conf = max(similarities.values())
                range_conf = max_conf - min_conf if max_conf > min_conf else 1.0
                
                # Normalize confidences to [0,1]
                normalized_similarities = {
                    cat: (sim - min_conf) / range_conf
                    for cat, sim in similarities.items()
                }
                
                # Create classifications list
                classifications = [
                    {"category": cat, "confidence": normalized_similarities[cat]}
                    for cat in self.categories
                ]
                
                return {
                    "classifications": classifications,
                    "top_category": top_category,
                    "confidence": normalized_similarities[top_category],
                    "raw_similarities": similarities
                }
                
        except Exception as e:
            import logging
            logging.getLogger("fileloader").error(f"DINO classification error: {str(e)}")
            return {
                "classifications": [],
                "top_category": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def finetune(self, category_examples: Dict[str, List[Image.Image]]):
        """
        Fine-tune the feature centroids using example images for each category.
        
        Args:
            category_examples: Dictionary mapping categories to lists of example images
        
        Note: This is a simplified implementation. In production, you would use
        a more sophisticated approach to update the centroids.
        """
        with torch_gc_context():
            for category, examples in category_examples.items():
                if category not in self.categories:
                    continue
                
                if not examples:
                    continue
                
                # Extract features for all examples
                features = []
                for img in examples:
                    inputs = self.processor(images=img, return_tensors="pt")
                    inputs = self.to_device(inputs)
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        # Get CLS token embedding
                        cls_features = outputs.last_hidden_state[:, 0, :].squeeze(0)
                        features.append(cls_features)
                
                if not features:
                    continue
                
                # Compute mean of features
                mean_features = torch.stack(features).mean(dim=0)
                
                # Normalize
                mean_features = F.normalize(mean_features, p=2, dim=0)
                
                # Update centroid
                self.feature_centroids[category] = mean_features
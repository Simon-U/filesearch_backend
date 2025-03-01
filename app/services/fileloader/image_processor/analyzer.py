from typing import Optional, Dict, Any, Type
import os
import torch
import gc
import logging
from PIL import Image
from dataclasses import dataclass
from contextlib import contextmanager

from .base import ImageAnalyzer, ImageAnalysisResult, ImageRelevance
from docling_core.types.doc import DoclingDocument, PictureItem

# Import the backend factories
from .caption_backend import CaptioningBackendFactory
from .classification_backend import ClassificationBackendFactory

# Import the model cache manager
try:
    from .model_cache_manager import ModelCacheManager
    HAS_CACHE_MANAGER = True
except ImportError:
    HAS_CACHE_MANAGER = False
    
# Set up logging for analyzer
logger = logging.getLogger("image_processor.analyzer")
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

@dataclass
class AnalyzerConfig:
    """Configuration for image analyzer"""
    
    model_type: str = "transformer"
    hf_token: Optional[str] = None  # Token for HuggingFace model access
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    confidence_threshold: float = 0.7
    cache_dir: Optional[str] = None
    max_batch_size: int = 1  # Process one image at a time by default
    
    # Classification configuration
    classification_backend_type: str = "clip"  # New field to select classification backend
    classification_model: str = "openai/clip-vit-base-patch32"
    
    # Captioning configuration
    enable_captioning: bool = True
    caption_model: str = "Salesforce/blip-image-captioning-base"
    caption_backend_type: str = "blip"  # Field to select captioning backend
    max_caption_length: int = 150
    
    # Memory management options
    use_half_precision: bool = True  # Use FP16 for models
    optimize_memory_usage: bool = True  # Enable memory optimization features

class ModelCacheManagerFactory:
    """Factory for getting or creating a model cache manager instance"""
    
    _instance = None
    
    @classmethod
    def get_instance(cls) -> Optional['ModelCacheManager']:
        """Get the model cache manager instance"""
        if not HAS_CACHE_MANAGER:
            logger.warning("Model cache manager not available. Models will be downloaded directly.")
            return None
            
        if cls._instance is None:
            # Get model cache settings from environment
            s3_bucket = os.environ.get('MODEL_CACHE_BUCKET')
            
            # Only create if S3 bucket is configured
            if s3_bucket:
                try:
                    s3_prefix = os.environ.get('MODEL_CACHE_PREFIX', 'cached-models')
                    local_cache_dir = os.environ.get('MODEL_CACHE_DIR', '/opt/ml/models')
                    hf_token = os.environ.get('HF_TOKEN')
                    
                    logger.info(f"Initializing model cache manager with bucket: {s3_bucket}")
                    cls._instance = ModelCacheManager(
                        s3_bucket=s3_bucket,
                        s3_prefix=s3_prefix,
                        local_cache_dir=local_cache_dir,
                        hf_token=hf_token
                    )
                except Exception as e:
                    logger.error(f"Failed to initialize model cache manager: {e}")
                    logger.warning("Models will be downloaded directly.")
                    return None
            else:
                logger.warning("MODEL_CACHE_BUCKET not set. Models will be downloaded directly.")
                return None
                
        return cls._instance

class ImageAnalyzerFactory:
    """Factory for creating image analyzer instances"""
    
    _analyzers: Dict[str, Type[ImageAnalyzer]] = {}
    
    @classmethod
    def register(cls, model_type: str) -> callable:
        """Decorator to register analyzer implementations"""
        def wrapper(analyzer_class: Type[ImageAnalyzer]) -> Type[ImageAnalyzer]:
            cls._analyzers[model_type] = analyzer_class
            return analyzer_class
        return wrapper
    
    @classmethod
    def get_analyzer(cls, config: AnalyzerConfig) -> ImageAnalyzer:
        """Get an analyzer instance based on config"""
        if config.model_type not in cls._analyzers:
            raise ValueError(f"Unknown model type: {config.model_type}")
            
        analyzer_class = cls._analyzers[config.model_type]
        return analyzer_class(config)

@ImageAnalyzerFactory.register("transformer")
class TransformerImageAnalyzer(ImageAnalyzer):
    def __init__(self, config: AnalyzerConfig):
        logger.info(f"The inital config is : {config}")
        self.config = config
        self.classification_backend = None
        self.caption_backend = None
        
        # Try to get model cache manager
        self.cache_manager = ModelCacheManagerFactory.get_instance()
        if self.cache_manager:
            logger.info("Using model cache manager for model loading")
            # Check if we need to modify model paths to use cached versions
            self._maybe_update_model_paths()
        
        self._initialize_backends()
        
    def _maybe_update_model_paths(self):
        """
        Update model paths to use cached versions if available
        """
        if not self.cache_manager:
            return
            
        try:
            # Try to use cached classification model
            if self.cache_manager.is_model_cached(self.config.classification_model):
                logger.info(f"Using cached classification model: {self.config.classification_model}")
                local_path = self.cache_manager.download_model_from_cache(self.config.classification_model)
                self.config.classification_model = local_path
            else:
                logger.info(f"Classification model not cached, will download and cache: {self.config.classification_model}")
                original_model_id = self.config.classification_model
                local_path = self.cache_manager.download_and_cache_model(original_model_id)
                self.config.classification_model = local_path
                
            # Try to use cached caption model if captioning is enabled
            if self.config.enable_captioning:
                if self.cache_manager.is_model_cached(self.config.caption_model):
                    logger.info(f"Using cached caption model: {self.config.caption_model}")
                    local_path = self.cache_manager.download_model_from_cache(self.config.caption_model)
                    self.config.caption_model = local_path
                else:
                    logger.info(f"Caption model not cached, will download and cache: {self.config.caption_model}")
                    original_model_id = self.config.caption_model
                    local_path = self.cache_manager.download_and_cache_model(original_model_id)
                    self.config.caption_model = local_path
        except Exception as e:
            logger.error(f"Error using model cache: {e}")
            logger.warning("Falling back to direct model download")
    
    def _initialize_backends(self):
        with torch_gc_context():
            # Initialize classification backend using the factory
            logger.info(f"The iniatal config before init {self.config}")
            try:
                self.classification_backend = ClassificationBackendFactory.get_backend(
                    self.config.classification_backend_type,
                    self.config
                )
                self.classification_backend.initialize()
            except Exception as e:
                logger.error(
                    f"Failed to initialize classification backend '{self.config.classification_backend_type}': {e}"
                )
                raise  # Classification is essential, so raise the error
            # Initialize captioning backend using the factory
            if self.config.enable_captioning:
                try:
                    self.caption_backend = CaptioningBackendFactory.get_backend(
                        self.config.caption_backend_type,
                        self.config
                    )
                    self.caption_backend.initialize()
                except Exception as e:
                    logger.error(
                        f"Failed to initialize captioning backend '{self.config.caption_backend_type}': {e}"
                    )
                    self.caption_backend = None
    
    def _is_substantive_category(self, category: str) -> bool:
        substantive_categories = {
            "chart", "diagram", "graph", "technical drawing",
            "data visualization", "infographic"
        }
        return category.lower() in substantive_categories
    
    def analyze_image(self, picture: PictureItem, document: DoclingDocument) -> ImageAnalysisResult:
        image = picture.get_image(document)
        if image is None:
            return ImageAnalysisResult(
                relevance=ImageRelevance.UNKNOWN,
                description="Could not load image",
                confidence=0.0,
                metadata={}
            )

        with torch_gc_context():
            # Classification
            classification = self.classification_backend.analyze(image)
            if "error" in classification:
                return ImageAnalysisResult(
                    relevance=ImageRelevance.UNKNOWN,
                    description=f"Classification failed: {classification['error']}",
                    confidence=0.0,
                    metadata=classification
                )

            top_category = classification["top_category"]
            confidence = classification["confidence"]
            is_substantive = self._is_substantive_category(top_category)

            # Captioning only if needed
            caption = ""
            logger.info(f"Check conditions, is substantive: {is_substantive}, confidence is {confidence} with threshold {self.config.confidence_threshold} and {confidence >= self.config.confidence_threshold}")
            if (is_substantive and 
                confidence >= self.config.confidence_threshold and 
                self.config.enable_captioning and 
                self.caption_backend is not None):
                try:
                    caption_result = self.caption_backend.analyze(image)
                    caption = caption_result.get("caption", "")
                except Exception as e:
                    logger.error(f"Caption generation failed: {e}")

        return ImageAnalysisResult(
            relevance=ImageRelevance.SUBSTANTIVE if is_substantive else ImageRelevance.DECORATIVE,
            description=caption if caption else f"Image classified as {top_category}",
            confidence=confidence,
            metadata={
                "classifications": classification.get("classifications", []),
                "category": top_category,
                "caption": caption
            }
        )
    
    def is_relevant(self, picture: PictureItem, document: DoclingDocument) -> bool:
        analysis = self.analyze_image(picture, document)
        return (
            analysis.relevance == ImageRelevance.SUBSTANTIVE and 
            analysis.confidence >= self.config.confidence_threshold
        )
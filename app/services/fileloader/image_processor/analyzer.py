from typing import Optional, Dict, Any, Type
import torch
import gc
from PIL import Image
from dataclasses import dataclass
from contextlib import contextmanager

from .base import ImageAnalyzer, ImageAnalysisResult, ImageRelevance
from docling_core.types.doc import DoclingDocument, PictureItem

# Import the backend factories
from .caption_backend import CaptioningBackendFactory
from .classification_backend import ClassificationBackendFactory

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
    model_name: str
    model_type: str = "transformer"
    hf_token: Optional[str] = None  # Token for HuggingFace model access
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    confidence_threshold: float = 0.5
    cache_dir: Optional[str] = None
    max_batch_size: int = 1  # Process one image at a time by default
    
    # Classification configuration
    classification_backend_type: str = "clip"  # New field to select classification backend
    
    # Captioning configuration
    enable_captioning: bool = True
    caption_model: str = "Salesforce/blip-image-captioning-base"
    caption_backend_type: str = "blip"  # Field to select captioning backend
    max_caption_length: int = 150
    
    # Memory management options
    use_half_precision: bool = True  # Use FP16 for models
    optimize_memory_usage: bool = True  # Enable memory optimization features

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
        self.config = config
        self.classification_backend = None
        self.caption_backend = None
        self._initialize_backends()
        
    def _initialize_backends(self):
        with torch_gc_context():
            # Initialize classification backend using the factory
            try:
                self.classification_backend = ClassificationBackendFactory.get_backend(
                    self.config.classification_backend_type,
                    self.config
                )
                self.classification_backend.initialize()
            except Exception as e:
                import logging
                logger = logging.getLogger("fileloader")
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
                    import logging
                    logger = logging.getLogger("fileloader")
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
            if (is_substantive and 
                confidence >= self.config.confidence_threshold and 
                self.config.enable_captioning and 
                self.caption_backend is not None):
                try:
                    caption_result = self.caption_backend.analyze(image)
                    caption = caption_result.get("caption", "")
                except Exception as e:
                    import logging
                    logger = logging.getLogger("fileloader")
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
from typing import Optional, Dict, Any, Type
import torch
import gc
from PIL import Image
from dataclasses import dataclass
from contextlib import contextmanager

from .base import ImageAnalyzer, ImageAnalysisResult, ImageRelevance
from docling_core.types.doc import DoclingDocument, PictureItem

from transformers import (
    AutoProcessor, 
    AutoModelForZeroShotImageClassification,
    BlipProcessor,
    BlipForConditionalGeneration
)



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
    enable_captioning: bool = True
    caption_model: str = "Salesforce/blip-image-captioning-base"
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
    
class BaseImageBackend:
    def __init__(self, config: AnalyzerConfig):
        self.config = config
        self.model = None
        self.processor = None
        
    def to_device(self, inputs):
        """Helper to move inputs to device with memory optimization"""
        if isinstance(inputs, dict):
            return {k: self.to_device(v) for k, v in inputs.items()}
        elif isinstance(inputs, (list, tuple)):
            return type(inputs)(self.to_device(v) for v in inputs)
        elif hasattr(inputs, 'to'):
            return inputs.to(self.config.device)
        return inputs

class CLIPImageBackend(BaseImageBackend):
    def initialize(self):
        with torch_gc_context():
            self.processor = AutoProcessor.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir,
                token=self.config.hf_token
            )
            self.model = AutoModelForZeroShotImageClassification.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir,
                token=self.config.hf_token,
                torch_dtype=torch.float16 if self.config.use_half_precision else torch.float32
            ).to(self.config.device)
            
            if self.config.optimize_memory_usage:
                self.model.eval()  # Ensure eval mode
                
            self.categories = [
                "chart", "diagram", "graph", "logo", "decorative element",
                "technical drawing", "data visualization", "infographic"
            ]

    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        try:
            with torch_gc_context(), torch.no_grad():
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

                return {
                    "classifications": [
                        {"category": cat, "confidence": prob.item()}
                        for cat, prob in zip(self.categories, probs)
                    ],
                    "top_category": self.categories[probs.argmax().item()],
                    "confidence": probs.max().item()
                }
        except Exception as e:
            return {
                "classifications": [],
                "top_category": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }

class CaptioningImageBackend(BaseImageBackend):
    def initialize(self):
        with torch_gc_context():
            model_kwargs = {
                'cache_dir': self.config.cache_dir,
                'torch_dtype': torch.float16 if self.config.use_half_precision else torch.float32
            }
            
            model_kwargs['token'] = self.config.hf_token
            self.processor = BlipProcessor.from_pretrained(
                self.config.caption_model,
                **model_kwargs
            )
            self.model = BlipForConditionalGeneration.from_pretrained(
                self.config.caption_model,
                **model_kwargs
            ).to(self.config.device)
            
            if self.config.optimize_memory_usage:
                self.model.eval()

    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        try:
            with torch_gc_context(), torch.no_grad():
                inputs = self.processor(image, return_tensors="pt")
                inputs = self.to_device(inputs)
                
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_caption_length,
                    num_beams=1  # Reduce beam search complexity
                )
                caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
                
                return {
                    "caption": caption,
                    "confidence": 1.0
                }
        except Exception as e:
            return {
                "caption": "",
                "error": str(e)
            }

@ImageAnalyzerFactory.register("transformer")
class TransformerImageAnalyzer(ImageAnalyzer):
    def __init__(self, config: AnalyzerConfig):
        self.config = config
        self.classification_backend = None
        self.caption_backend = None
        self._initialize_backends()
        
    def _initialize_backends(self):
        with torch_gc_context():
            self.classification_backend = CLIPImageBackend(self.config)
            self.classification_backend.initialize()
            
            if self.config.enable_captioning:
                self.caption_backend = CaptioningImageBackend(self.config)
                self.caption_backend.initialize()
    
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
                    print(f"Caption generation failed: {e}")

        return ImageAnalysisResult(
            relevance=ImageRelevance.SUBSTANTIVE if is_substantive else ImageRelevance.DECORATIVE,
            description=caption if caption else f"Image classified as {top_category}",
            confidence=confidence,
            metadata={
                "classifications": classification["classifications"],
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
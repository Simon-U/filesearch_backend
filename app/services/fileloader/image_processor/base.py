from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

from docling_core.types.doc import DoclingDocument, PictureItem
from docling.models.base_model import BaseEnrichmentModel

class ImageRelevance(Enum):
    """Classification of image relevance in document context"""
    SUBSTANTIVE = "substantive"    # Charts, diagrams, important visuals
    DECORATIVE = "decorative"      # Logos, design elements
    UNKNOWN = "unknown"            # Not yet classified

@dataclass
class ImageAnalysisResult:
    """Results from image analysis"""
    relevance: ImageRelevance
    description: str
    confidence: float
    caption: Optional[str] = None 
    metadata: Dict[str, Any] = None
    
class ImageAnalyzer(ABC):
    """Base class for image analysis implementations"""
    
    @abstractmethod
    def analyze_image(self, picture: PictureItem, document: DoclingDocument) -> ImageAnalysisResult:
        """Analyze a single image within document context"""
        pass
    
    @abstractmethod
    def is_relevant(self, picture: PictureItem, document: DoclingDocument) -> bool:
        """Determine if an image is relevant for document understanding"""
        pass

class DocumentImageProcessor:
    """Coordinates image processing across entire document"""
    
    def __init__(self, analyzer: ImageAnalyzer):
        self.analyzer = analyzer
    
    def process_document(self, document: DoclingDocument) -> DoclingDocument:
        """Process all images in a document"""
        for picture in document.pictures:
            if self.analyzer.is_relevant(picture, document):
                analysis = self.analyzer.analyze_image(picture, document)
                self._enrich_picture(picture, analysis)
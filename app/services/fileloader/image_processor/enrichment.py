from typing import Iterable, Any, Optional
from docling.models.base_model import BaseEnrichmentModel
from docling_core.types.doc import (
    DoclingDocument,
    NodeItem,
    PictureItem,
    PictureClassificationData,
    PictureClassificationClass,
    TextItem,
    DocItemLabel,
    RefItem
)

from .analyzer import ImageAnalyzerFactory, AnalyzerConfig
from .base import ImageRelevance

class DocumentImageEnrichmentModel(BaseEnrichmentModel):
    """
    Enrichment model for processing images in documents.
    Integrates with docling's enrichment system to add image analysis results
    as annotations to PictureItems.
    """
    
    def __init__(self, config: Optional[AnalyzerConfig] = None, enabled: bool = True):
        """
        Initialize the enrichment model.
        
        Args:
            config: Configuration for the image analyzer. If None, uses default config.
            enabled: Whether this enrichment model is enabled.
        """
        self.enabled = enabled
        
        # Set up default config if none provided
        if config is None:
            config = AnalyzerConfig(
                model_type="transformer",
                model_name="openai/clip-vit-base-patch32"
            )
            
        # Initialize the image analyzer
        self.analyzer = ImageAnalyzerFactory.get_analyzer(config)
    
    def is_processable(self, doc: DoclingDocument, element: NodeItem) -> bool:
        """
        Check if an element can be processed by this enrichment model.
        
        Args:
            doc: The document containing the element
            element: The element to check
            
        Returns:
            bool: True if the element is a PictureItem and model is enabled
        """
        return self.enabled and isinstance(element, PictureItem)
    
    def _create_caption_text_item(self, doc: DoclingDocument, caption: str) -> TextItem:
        """Create a TextItem for the caption with appropriate references."""
        # Create a unique reference for the caption
        caption_ref = f"#/texts/{len(doc.texts)}"
        
        # Create the caption text item
        caption_item = TextItem(
            self_ref=caption_ref,
            label=DocItemLabel.CAPTION,
            text=caption,
            orig=caption
        )
        return caption_item
        
        
    def __call__(
        self,
        doc: DoclingDocument,
        element_batch: Iterable[NodeItem]
    ) -> Iterable[Any]:
        """
        Process a batch of elements and add enrichments.
        
        Args:
            doc: The document containing the elements
            element_batch: Batch of elements to process
            
        Yields:
            Processed elements with added annotations
        """
        if not self.enabled:
            return
            
        for element in element_batch:
            if not isinstance(element, PictureItem):
                continue
                
            # Analyze the image and check relevance
            if not self.analyzer.is_relevant(element, doc):
                yield element
                continue
                
            analysis = self.analyzer.analyze_image(element, doc)
            
            # Handle classification
            classification_data = PictureClassificationData(
                provenance=f"image_analyzer-{self.analyzer.__class__.__name__}",
                predicted_classes=[
                    PictureClassificationClass(
                        class_name=analysis.metadata.get("category", "unknown"),
                        confidence=analysis.confidence
                    )
                ]
            )
            
            # Add relevance classification
            if analysis.relevance != ImageRelevance.UNKNOWN:
                classification_data.predicted_classes.append(
                    PictureClassificationClass(
                        class_name=analysis.relevance.value,
                        confidence=analysis.confidence
                    )
                )

            # Add the classification to picture's annotations
            element.annotations.append(classification_data)
            
            # If we have a caption, create it properly as a TextItem
            if analysis.metadata.get("caption"):
                # Create a new TextItem for the caption
                caption_text = analysis.metadata["caption"]
                formatted_caption = f"[IMG_DESCRIPTION: {caption_text}]"
                
                # Add the caption text to document's texts list
                caption_item = TextItem(
                    label="caption",
                    text=formatted_caption,
                    orig=formatted_caption,
                    self_ref=f"#/texts/{len(doc.texts)}"
                )
                doc.texts.append(caption_item)
                
                # Create a reference to the caption
                ref_item = RefItem(cref=caption_item.self_ref)
                
                # Add the reference to the picture's captions
                element.captions.append(ref_item)
            
            yield element
            
    @classmethod
    def get_default_options(cls) -> dict:
        """Get default options for this enrichment model."""
        return {
            "enabled": True,
            "config": None
        }
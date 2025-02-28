from .factory import ClassificationBackendFactory
from .base import BaseClassificationBackend
from .clip import CLIPClassificationBackend
from .deit import DEiTClassificationBackend
from .dino import DINOClassificationBackend
from .vit import ViTClassificationBackend

# Register backends with factory
ClassificationBackendFactory.register("clip")(CLIPClassificationBackend)
ClassificationBackendFactory.register("deit")(DEiTClassificationBackend)
ClassificationBackendFactory.register("dino")(DINOClassificationBackend)
ClassificationBackendFactory.register("vit")(ViTClassificationBackend)

__all__ = [
    "ClassificationBackendFactory",
    "BaseClassificationBackend",
    "CLIPClassificationBackend",
    "DEiTClassificationBackend",
    "DINOClassificationBackend",
    "ViTClassificationBackend"
]
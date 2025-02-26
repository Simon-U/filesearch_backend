from .factory import ClassificationBackendFactory
from .base import BaseClassificationBackend
from .clip import CLIPClassificationBackend
from .deit import DEiTClassificationBackend
from .dino import DINOClassificationBackend

# Register backends with factory
ClassificationBackendFactory.register("clip")(CLIPClassificationBackend)
ClassificationBackendFactory.register("deit")(DEiTClassificationBackend)
ClassificationBackendFactory.register("dino")(DINOClassificationBackend)

__all__ = [
    "ClassificationBackendFactory",
    "BaseClassificationBackend",
    "CLIPClassificationBackend",
    "DEiTClassificationBackend",
    "DINOClassificationBackend"
]
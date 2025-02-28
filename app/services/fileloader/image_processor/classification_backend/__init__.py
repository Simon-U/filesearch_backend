from .factory import ClassificationBackendFactory
from .base import BaseClassificationBackend
from .clip import CLIPClassificationBackend
from .deit import DEiTClassificationBackend
from .dino import DINOClassificationBackend
from .florence import Florence2ClassificationBackend

# Register backends with factory
ClassificationBackendFactory.register("clip")(CLIPClassificationBackend)
ClassificationBackendFactory.register("deit")(DEiTClassificationBackend)
ClassificationBackendFactory.register("dino")(DINOClassificationBackend)
ClassificationBackendFactory.register("florence")(Florence2ClassificationBackend)

__all__ = [
    "ClassificationBackendFactory",
    "BaseClassificationBackend",
    "CLIPClassificationBackend",
    "DEiTClassificationBackend",
    "DINOClassificationBackend",
    "Florence2ClassificationBackend"
]
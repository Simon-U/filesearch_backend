from .factory import CaptioningBackendFactory
from .base import BaseCaptioningBackend
from .blip import BlipCaptioningBackend
from .smolvlm import SmolVLMCaptioningBackend
from .pali import PaligemmaCaptioningBackend

# Register backends with factory
CaptioningBackendFactory.register("blip")(BlipCaptioningBackend)
CaptioningBackendFactory.register("smolvlm")(SmolVLMCaptioningBackend)
CaptioningBackendFactory.register("pali")(PaligemmaCaptioningBackend)

__all__ = [
    "CaptioningBackendFactory",
    "BaseCaptioningBackend",
    "BlipCaptioningBackend", 
    "PaligemmaCaptioningBackend",
    "SmolVLMCaptioningBackend"
]
from .factory import CaptioningBackendFactory
from .base import BaseCaptioningBackend
from .blip import BlipCaptioningBackend
from .smolvlm import SmolVLMCaptioningBackend

# Register backends with factory
CaptioningBackendFactory.register("blip")(BlipCaptioningBackend)
CaptioningBackendFactory.register("smolvlm")(SmolVLMCaptioningBackend)

__all__ = [
    "CaptioningBackendFactory",
    "BaseCaptioningBackend",
    "BlipCaptioningBackend", 
    "SmolVLMCaptioningBackend"
]
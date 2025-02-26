from typing import Dict, Type, Optional
from .base import BaseCaptioningBackend

class CaptioningBackendFactory:
    """Factory for creating captioning backend instances"""
    
    _backends: Dict[str, Type[BaseCaptioningBackend]] = {}
    
    @classmethod
    def register(cls, backend_type: str):
        """Decorator to register captioning backend implementations"""
        def wrapper(backend_class: Type[BaseCaptioningBackend]) -> Type[BaseCaptioningBackend]:
            cls._backends[backend_type] = backend_class
            return backend_class
        return wrapper
    
    @classmethod
    def get_backend(cls, backend_type: str, config) -> Optional[BaseCaptioningBackend]:
        """Get a captioning backend instance based on type"""
        if backend_type not in cls._backends:
            raise ValueError(f"Unknown captioning backend: {backend_type}")
        return cls._backends[backend_type](config)
from typing import Dict, Type, Optional
from .base import BaseClassificationBackend

class ClassificationBackendFactory:
    """Factory for creating classification backend instances"""
    
    _backends: Dict[str, Type[BaseClassificationBackend]] = {}
    
    @classmethod
    def register(cls, backend_type: str):
        """Decorator to register classification backend implementations"""
        def wrapper(backend_class: Type[BaseClassificationBackend]) -> Type[BaseClassificationBackend]:
            cls._backends[backend_type] = backend_class
            return backend_class
        return wrapper
    
    @classmethod
    def get_backend(cls, backend_type: str, config) -> Optional[BaseClassificationBackend]:
        """Get a classification backend instance based on type"""
        if backend_type not in cls._backends:
            raise ValueError(f"Unknown classification backend: {backend_type}")
        return cls._backends[backend_type](config)
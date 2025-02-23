from abc import ABC, abstractmethod
from ..models import FileLocation, FileMetadata


class StorageConnector(ABC):
    """Abstract base class for storage connectors"""
    
    @abstractmethod
    def get_file_content(self, location: FileLocation) -> bytes:
        """Retrieve file content as bytes"""
        pass
    
    @abstractmethod
    def get_metadata(self, location: FileLocation) -> FileMetadata:
        """Retrieve file metadata"""
        pass
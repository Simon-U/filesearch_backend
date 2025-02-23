from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum
from datetime import datetime


class StorageType(Enum):
    LOCAL = "local"
    S3 = "s3"
    SHAREPOINT = "sharepoint"


@dataclass
class UserInfo:
    """Unified user information structure"""
    name: Optional[str] = None
    email: Optional[str] = None
    system_id: Optional[str] = None  # For storage-specific IDs


@dataclass
class FileLocation:
    """Unified file location structure"""
    path: str
    storage_type: StorageType
    version: Optional[str] = None

    @classmethod
    def from_local(cls, path: str) -> "FileLocation":
        return cls(path=path, storage_type=StorageType.LOCAL)
    
    @classmethod
    def from_s3(cls, path: str, version: Optional[str] = None) -> "FileLocation":
        return cls(
            path=path,
            storage_type=StorageType.S3,
            version=version
        )
    
    @classmethod
    def from_sharepoint(cls, path: str) -> "FileLocation":
        return cls(
            path=path,
            storage_type=StorageType.SHAREPOINT,
        )


@dataclass
class FileMetadata:
    """Unified metadata structure"""
    # Core file information
    file_path: str
    file_type: str  # File extension (.pdf, .docx, etc)
    content_type: str  # MIME type (application/pdf, etc)
    size_bytes: int
    
    # Access information
    universal_url: str  # URL or path to access the file
    storage_type: StorageType
    
    # Timestamps
    created_at: datetime
    modified_at: datetime
    
    # User information
    created_by: Optional[UserInfo] = None
    modified_by: Optional[UserInfo] = None
    
    # Version information
    version: Optional[str] = None
    version_label: Optional[str] = None  # Human readable version label
    
    # Additional metadata
    tags: Dict[str, str] = None  # For custom categorization
    additional_info: Dict[str, Any] = None  # Storage-specific metadata

    def __post_init__(self):
        # Convert string timestamps to datetime if needed
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at.replace('Z', '+00:00'))
        if isinstance(self.modified_at, str):
            self.modified_at = datetime.fromisoformat(self.modified_at.replace('Z', '+00:00'))


# Keep existing config classes
@dataclass
class SharePointConfig:
    """SharePoint connection configuration"""
    tenant_id: str
    client_id: str
    client_secret: str
    site_url: str


@dataclass
class S3Config:
    """S3 connection configuration"""
    bucket_name: str
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    region_name: Optional[str] = None
    endpoint_url: Optional[str] = None
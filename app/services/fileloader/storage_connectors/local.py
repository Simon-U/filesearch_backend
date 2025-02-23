import os
import pwd
import mimetypes
from pathlib import Path
from datetime import datetime
from .base import StorageConnector
from ..models import FileLocation, FileMetadata, UserInfo, StorageType


class LocalStorageConnector(StorageConnector):
    def get_file_content(self, location: FileLocation) -> bytes:
        with open(location.path, 'rb') as f:
            return f.read()
    
    def get_metadata(self, location: FileLocation) -> FileMetadata:
        path = Path(location.path)
        stat = path.stat()
        
        # Get file owner information if possible
        try:
            owner = pwd.getpwuid(stat.st_uid)
            created_by = UserInfo(
                name=owner.pw_gecos.split(',')[0] if owner.pw_gecos else owner.pw_name,
                system_id=str(stat.st_uid)
            )
        except (KeyError, ImportError):
            # pwd might not be available on Windows or user might not exist
            created_by = None
            
        # For modification, we use the same user info
        modified_by = created_by
        
        # Get mime type
        content_type, _ = mimetypes.guess_type(str(path))
        if not content_type:
            content_type = 'application/octet-stream'
            
        # Create platform-independent file URL
        if os.name == 'nt':  # Windows
            universal_url = f'file:///{str(path.absolute()).replace(os.sep, "/")}'
        else:  # Unix-like
            universal_url = f'file://{path.absolute()}'
            
        return FileMetadata(
            file_path=str(path),
            file_type=path.suffix.lower(),
            content_type=content_type,
            size_bytes=stat.st_size,
            universal_url=universal_url,
            storage_type=StorageType.LOCAL,
            created_at=datetime.fromtimestamp(stat.st_ctime),
            modified_at=datetime.fromtimestamp(stat.st_mtime),
            created_by=created_by,
            modified_by=modified_by,
            version=None,
            version_label=None,
            tags={},
            additional_info={
                'permissions': oct(stat.st_mode)[-3:],
                'is_symlink': path.is_symlink(),
                'absolute_path': str(path.absolute())
            }
        )
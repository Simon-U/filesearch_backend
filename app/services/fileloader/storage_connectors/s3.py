import boto3
import os
import mimetypes
from datetime import datetime, timedelta
from .base import StorageConnector
from ..models import FileLocation, FileMetadata, UserInfo, StorageType, S3Config


class S3StorageConnector(StorageConnector):
    def __init__(self, config: S3Config):
        self.config = config
        self.s3_client = self._initialize_s3_client(config)
        # Default presigned URL expiration (24 hours)
        self.url_expiration = 86400
    
    def _initialize_s3_client(self, config: S3Config):
        client_kwargs = {
            'region_name': config.region_name
        }
        if config.aws_access_key_id and config.aws_secret_access_key:
            client_kwargs.update({
                'aws_access_key_id': config.aws_access_key_id,
                'aws_secret_access_key': config.aws_secret_access_key
            })
        if config.endpoint_url:
            client_kwargs['endpoint_url'] = config.endpoint_url
            
        return boto3.client('s3', **client_kwargs)

    def get_file_content(self, location: FileLocation) -> bytes:
        response = self.s3_client.get_object(
            Bucket=self.config.bucket_name,
            Key=location.path,
            VersionId=location.version
        )
        return response['Body'].read()

    def get_metadata(self, location: FileLocation) -> FileMetadata:
        # Get object metadata
        kwargs = {
            'Bucket': self.config.bucket_name,
            'Key': location.path
        }
        if location.version:
            kwargs['VersionId'] = location.version
            
        response = self.s3_client.head_object(**kwargs)
        
        # Generate presigned URL for access
        url_params = {
            'Bucket': self.config.bucket_name,
            'Key': location.path,
            'ExpiresIn': self.url_expiration
        }
        if location.version:
            url_params['VersionId'] = location.version
            
        universal_url = self.s3_client.generate_presigned_url(
            'get_object',
            Params=url_params
        )
        
        # Extract user information from metadata if available
        metadata = response.get('Metadata', {})
        created_by = UserInfo(
            name=metadata.get('created-by-name'),
            email=metadata.get('created-by-email'),
            system_id=metadata.get('created-by-id')
        ) if any(k.startswith('created-by-') for k in metadata) else None
        
        modified_by = UserInfo(
            name=metadata.get('modified-by-name'),
            email=metadata.get('modified-by-email'),
            system_id=metadata.get('modified-by-id')
        ) if any(k.startswith('modified-by-') for k in metadata) else None
        
        # Get content type
        content_type = response.get('ContentType')
        if not content_type:
            content_type, _ = mimetypes.guess_type(location.path)
            content_type = content_type or 'application/octet-stream'
            
        return FileMetadata(
            file_path=location.path,
            file_type=os.path.splitext(location.path)[1].lower(),
            content_type=content_type,
            size_bytes=response.get('ContentLength', 0),
            universal_url=universal_url,
            storage_type=StorageType.S3,
            created_at=datetime.now(),  # S3 doesn't provide creation time
            modified_at=response.get('LastModified'),
            created_by=created_by,
            modified_by=modified_by,
            version=response.get('VersionId'),
            version_label=metadata.get('version-label'),
            tags=response.get('TagSet', {}),
            additional_info={
                'etag': response.get('ETag'),
                'storage_class': response.get('StorageClass'),
                'server_side_encryption': response.get('ServerSideEncryption'),
                'metadata': metadata,
                'content_encoding': response.get('ContentEncoding'),
                'cache_control': response.get('CacheControl'),
                'content_disposition': response.get('ContentDisposition'),
                'replication_status': response.get('ReplicationStatus')
            }
        )
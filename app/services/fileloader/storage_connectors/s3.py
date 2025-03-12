import boto3
import os
import json
import mimetypes
from datetime import datetime
from typing import List, Dict, Any, Optional
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

    def get_file_content(self, location: FileLocation, is_json: bool = False) -> bytes:
        if is_json:
            return self.load_file_content_as_json(location.path)
        response = self.s3_client.get_object(
            Bucket=self.config.bucket_name,
            Key=location.path
        )
        return response['Body'].read()

    def get_metadata(self, location: FileLocation) -> FileMetadata:
        # Get object metadata
        response = self.s3_client.head_object(
            Bucket=self.config.bucket_name,
            Key=location.path
        )
        
        # Generate presigned URL for access
        universal_url = self.s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': self.config.bucket_name,
                'Key': location.path,
                'ExpiresIn': self.url_expiration
            }
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
        
    def list_files_in_directory(self, subdirectory: str) -> List[str]:
        """
        List all files in the specified subdirectory.
        
        Args:
            subdirectory (str): The subdirectory within the bucket to search for files.
            
        Returns:
            List[str]: A list of file keys in the directory.
        """
        prefix = f"{subdirectory.rstrip('/')}/" if subdirectory else ""
        
        paginator = self.s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=self.config.bucket_name, Prefix=prefix)
        
        file_keys = []
        for page in page_iterator:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    # Skip the directory itself and any hidden files
                    if (not key.endswith('/') and 
                        not os.path.basename(key).startswith('.') and 
                        key != prefix):
                        file_keys.append(key)
        
        return file_keys
    
    def load_file_content_as_json(self, file_key: str) -> Optional[Dict[str, Any]]:
        """
        Load and parse the content of a file from S3 as JSON.
        
        Args:
            file_key (str): The key of the file to load.
            
        Returns:
            Optional[Dict[str, Any]]: The parsed file content as a dictionary, or None if an error occurs.
        """
        try:
            # Skip directory paths (they typically end with a slash)
            if file_key.endswith('/'):
                return None
                
            response = self.s3_client.get_object(Bucket=self.config.bucket_name, Key=file_key)
            content = response['Body'].read().decode('utf-8')
            
            # Check if content is empty or whitespace
            if not content or content.isspace():
                print(f"File {file_key} is empty")
                return None
                
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error in file {file_key}: {str(e)}")
            return None
        except Exception as e:
            print(f"Error loading file {file_key}: {str(e)}")
            return None
    
    def process_output_files(self, subdirectory: str) -> List[Dict[str, Any]]:
        """
        Process all .out files in the subdirectory and extract document and metadata
        from files with 'success' status.
        
        Args:
            subdirectory (str): The subdirectory within the bucket to search for files.
            
        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing document and metadata.
        """
        results = []
        
        # Get all files in the subdirectory
        file_keys = self.list_files_in_directory(subdirectory)
        
        # Process each file (focusing on .out files)
        for file_key in file_keys:
            # Ensure the file has a .out extension
            if not file_key.endswith('.out'):
                continue
                
            content = self.load_file_content_as_json(file_key)
            
            if content and isinstance(content, dict):
                # Check for required fields
                if all(key in content for key in ['status', 'document', 'metadata']):
                    if content['status'] == 'success':
                        results.append({
                            'file_key': file_key,
                            'document': content['document'],
                            'metadata': content['metadata'],
                        })
                else:
                    print(f"File {file_key} is missing required fields: status, document, or metadata")
        
        return results
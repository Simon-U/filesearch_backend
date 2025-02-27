import os
import boto3
import logging
import torch
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
from huggingface_hub import snapshot_download, hf_hub_download

# Set up logging
logger = logging.getLogger("model_cache_manager")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class ModelCacheManager:
    """
    Manages model caching to and from S3 to avoid redundant downloads.
    """
    
    def __init__(
        self,
        s3_bucket: str,
        s3_prefix: str = "models",
        local_cache_dir: str = "/opt/ml/models",
        hf_token: Optional[str] = None
    ):
        """
        Initialize the model cache manager.
        
        Args:
            s3_bucket: S3 bucket name for storing cached models
            s3_prefix: Prefix path in the S3 bucket
            local_cache_dir: Local directory to store downloaded models
            hf_token: Optional HuggingFace token for private models
        """
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.local_cache_dir = Path(local_cache_dir)
        self.hf_token = hf_token
        
        # Create local cache directory if it doesn't exist
        os.makedirs(self.local_cache_dir, exist_ok=True)
        
        # Initialize S3 client
        self.s3_client = boto3.client('s3')
        
        logger.info(f"Initialized ModelCacheManager with bucket: {s3_bucket}, prefix: {s3_prefix}")
        logger.info(f"Local cache directory: {local_cache_dir}")
        
        # Create a manifest file to track cached models
        self.manifest_path = self.local_cache_dir / "manifest.json"
        self.manifest = self._load_manifest()
    
    def _load_manifest(self) -> Dict[str, Any]:
        """Load the manifest from the local cache or initialize a new one."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load manifest: {e}. Creating new manifest.")
        
        # Create initial manifest
        manifest = {
            "models": {},
            "version": "1.0"
        }
        self._save_manifest(manifest)
        return manifest
    
    def _save_manifest(self, manifest: Dict[str, Any]) -> None:
        """Save the manifest to the local cache."""
        with open(self.manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def _get_model_s3_key(self, model_id: str) -> str:
        """Generate the S3 key for a model."""
        # Clean model ID to be usable as path (replace '/' with '--')
        clean_model_id = model_id.replace('/', '--')
        return f"{self.s3_prefix}/{clean_model_id}"
    
    def _get_model_local_path(self, model_id: str) -> Path:
        """Generate the local path for a model."""
        clean_model_id = model_id.replace('/', '--')
        return self.local_cache_dir / clean_model_id
    
    def list_cached_models(self) -> List[str]:
        """List all models in the cache manifest."""
        return list(self.manifest["models"].keys())
    
    def _list_s3_models(self) -> List[str]:
        """List all models in S3 bucket"""
        models = []
        try:
            # List all prefixes in the bucket
            paginator = self.s3_client.get_paginator('list_objects_v2')
            result = paginator.paginate(
                Bucket=self.s3_bucket, 
                Prefix=self.s3_prefix + '/',
                Delimiter='/'
            )
            
            # Extract model names
            for page in result:
                if 'CommonPrefixes' in page:
                    for prefix in page['CommonPrefixes']:
                        # Extract just the model name from the prefix
                        prefix_path = prefix.get('Prefix', '')
                        if prefix_path.endswith('/'):
                            prefix_path = prefix_path[:-1]
                            
                        # Get the last part after the prefix
                        model_name = prefix_path.split('/')[-1]
                        # Convert back from S3 format (-- to /)
                        original_name = model_name.replace('--', '/')
                        models.append(original_name)
                        
            logger.info(f"Found {len(models)} models in S3: {models}")
            return models
        except Exception as e:
            logger.error(f"Error listing S3 models: {e}")
            return []
    
    def is_model_cached(self, model_id: str) -> bool:
        """Check if a model is in the cache (both S3 and manifest)."""
        try:
            # Check if local path exists
            local_path = self._get_model_local_path(model_id)
            if local_path.exists():
                logger.info(f"Model {model_id} found in local cache at {local_path}")
                return True
            
            # Check S3 directly
            s3_key = self._get_model_s3_key(model_id)
            try:
                # First try checking if the directory exists
                response = self.s3_client.list_objects_v2(
                    Bucket=self.s3_bucket,
                    Prefix=s3_key + '/',
                    MaxKeys=1
                )
                if 'Contents' in response and len(response['Contents']) > 0:
                    logger.info(f"Model {model_id} found in S3 cache at {s3_key}/")
                    
                    # Update manifest if needed
                    if model_id not in self.manifest["models"]:
                        self.manifest["models"][model_id] = {
                            "local_path": str(local_path),
                            "s3_key": s3_key,
                            "downloaded_at": get_timestamp()
                        }
                        self._save_manifest(self.manifest)
                        
                    return True
                
                # Try with specific common file
                # Most models should have a config.json, model.safetensors, pytorch_model.bin, etc.
                for common_file in ['config.json', 'model.safetensors', 'pytorch_model.bin']:
                    try:
                        self.s3_client.head_object(
                            Bucket=self.s3_bucket, 
                            Key=f"{s3_key}/{common_file}"
                        )
                        logger.info(f"Model {model_id} found in S3 cache with {common_file}")
                        
                        # Update manifest if needed
                        if model_id not in self.manifest["models"]:
                            self.manifest["models"][model_id] = {
                                "local_path": str(local_path),
                                "s3_key": s3_key,
                                "downloaded_at": get_timestamp()
                            }
                            self._save_manifest(self.manifest)
                            
                        return True
                    except Exception:
                        continue
                
                logger.info(f"Model {model_id} not found in S3 cache")
            except Exception as e:
                logger.warning(f"Error checking S3 for model {model_id}: {e}")
            
            return False
        except Exception as e:
            logger.error(f"Error checking model cache: {e}")
            return False
    
    def download_model_from_cache(self, model_id: str) -> str:
        """
        Download a model from the S3 cache to the local directory.
        
        Returns:
            Local path to the model
        """
        if not self.is_model_cached(model_id):
            raise ValueError(f"Model {model_id} not found in cache")
        
        local_path = self._get_model_local_path(model_id)
        
        # If already in local cache, no need to download
        if local_path.exists():
            logger.info(f"Model {model_id} already exists in local cache")
            return str(local_path)
        
        s3_key = self._get_model_s3_key(model_id)
        
        try:
            # Download the model from S3
            logger.info(f"Downloading model {model_id} from S3 cache at {s3_key}/")
            os.makedirs(local_path, exist_ok=True)
            
            # List all objects in the model directory
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.s3_bucket, Prefix=s3_key + '/')
            
            downloaded_files = 0
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        # Extract relative path
                        relative_path = obj['Key'][len(s3_key)+1:]
                        if not relative_path:  # Skip directory marker
                            continue
                        
                        # Create subdirectories if needed
                        file_path = local_path / relative_path
                        os.makedirs(file_path.parent, exist_ok=True)
                        
                        # Download file
                        logger.debug(f"Downloading {obj['Key']} to {file_path}")
                        self.s3_client.download_file(
                            self.s3_bucket, 
                            obj['Key'], 
                            str(file_path)
                        )
                        downloaded_files += 1
            
            if downloaded_files == 0:
                raise ValueError(f"No files found for model {model_id} in S3 cache")
                
            logger.info(f"Successfully downloaded {downloaded_files} files for model {model_id} from S3 cache")
            return str(local_path)
        except Exception as e:
            logger.error(f"Error downloading model {model_id} from S3: {e}")
            if local_path.exists():
                shutil.rmtree(local_path)
            raise
    
    def download_and_cache_model(self, model_id: str, **kwargs) -> str:
        """
        Download a model from HuggingFace and cache it in S3.
        
        Args:
            model_id: HuggingFace model ID
            **kwargs: Additional arguments to pass to snapshot_download
            
        Returns:
            Local path to the downloaded model
        """
        # Check if model is already cached
        if self.is_model_cached(model_id):
            try:
                return self.download_model_from_cache(model_id)
            except Exception as e:
                logger.warning(f"Failed to use cached model {model_id}: {e}. Will download from source.")
        
        local_path = self._get_model_local_path(model_id)
        s3_key = self._get_model_s3_key(model_id)
        
        # Remove any existing local files to avoid conflicts
        if local_path.exists():
            shutil.rmtree(local_path)
        
        try:
            # Download from HuggingFace
            logger.info(f"Downloading model {model_id} from HuggingFace")
            download_kwargs = {
                "cache_dir": str(self.local_cache_dir / "hf_cache"),
                "local_dir": str(local_path),
                "local_dir_use_symlinks": False
            }
            
            # Add token if provided
            if self.hf_token:
                download_kwargs["token"] = self.hf_token
                
            # Add any additional kwargs
            download_kwargs.update(kwargs)
            
            # Download the model
            snapshot_download(repo_id=model_id, **download_kwargs)
            
            # Upload to S3
            logger.info(f"Uploading model {model_id} to S3 cache")
            self._upload_model_to_s3(model_id, local_path, s3_key)
            
            # Update manifest
            self.manifest["models"][model_id] = {
                "local_path": str(local_path),
                "s3_key": s3_key,
                "downloaded_at": get_timestamp()
            }
            self._save_manifest(self.manifest)
            
            logger.info(f"Successfully downloaded and cached model {model_id}")
            return str(local_path)
        except Exception as e:
            logger.error(f"Error downloading model {model_id}: {e}")
            if local_path.exists():
                shutil.rmtree(local_path)
            raise
    
    def _upload_model_to_s3(self, model_id: str, local_path: Path, s3_key: str) -> None:
        """Upload a model to S3."""
        try:
            # Upload all files in the directory
            files_uploaded = 0
            for file_path in local_path.glob('**/*'):
                if file_path.is_file():
                    relative_path = file_path.relative_to(local_path)
                    object_key = f"{s3_key}/{relative_path}"
                    logger.debug(f"Uploading {file_path} to s3://{self.s3_bucket}/{object_key}")
                    self.s3_client.upload_file(
                        str(file_path),
                        self.s3_bucket,
                        object_key
                    )
                    files_uploaded += 1
            logger.info(f"Successfully uploaded model {model_id} to S3 ({files_uploaded} files)")
        except Exception as e:
            logger.error(f"Error uploading model {model_id} to S3: {e}")
            raise

def get_timestamp() -> str:
    """Get the current timestamp as a string."""
    from datetime import datetime
    return datetime.now().isoformat()
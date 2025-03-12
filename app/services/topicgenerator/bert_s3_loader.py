import tempfile
import os
import shutil
import boto3

class BertTopicS3Loader:
    """
    Helper class to download BERTopic models from S3 to local filesystem.
    Returns paths to local files rather than loading the model directly.
    """
    
    @staticmethod
    def _get_s3_client(bucket_name, region_name="us-west-2", aws_access_key_id=None, aws_secret_access_key=None, endpoint_url=None):
        """
        Create an S3 client with the provided credentials.
        """
        s3_kwargs = {"region_name": region_name}
        
        if aws_access_key_id and aws_secret_access_key:
            s3_kwargs.update({
                "aws_access_key_id": aws_access_key_id,
                "aws_secret_access_key": aws_secret_access_key
            })
            
        if endpoint_url:
            s3_kwargs["endpoint_url"] = endpoint_url
        
        s3_client = boto3.client("s3", **s3_kwargs)
        return s3_client, bucket_name
    
    @classmethod
    def download_model_from_s3(cls, s3_path, bucket_name, region_name="us-west-2", 
                          aws_access_key_id=None, aws_secret_access_key=None, 
                          endpoint_url=None):
        """
        Download a BERTopic model from S3 to a local temporary directory.
        
        :param s3_path: Path to the model file in S3
        :param bucket_name: S3 bucket name
        :param region_name: AWS region
        :param aws_access_key_id: AWS access key (optional, can use IAM role)
        :param aws_secret_access_key: AWS secret key (optional, can use IAM role)
        :param endpoint_url: Custom S3 endpoint URL (optional)
        :return: Tuple of (local_path, temp_dir) where local_path is the path to use for loading the model
                and temp_dir is the directory to clean up later
        """
        # Get S3 client
        s3_client, bucket = cls._get_s3_client(
            bucket_name=bucket_name,
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            endpoint_url=endpoint_url
        )
        
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Download the file
            local_file_path = os.path.join(temp_dir, os.path.basename(s3_path))
            s3_client.download_file(bucket, s3_path, local_file_path)
            
            return local_file_path, temp_dir
            
        except Exception as e:
            print(e, flush=True)
            # Clean up on any error
            cls.cleanup_temp_dir(temp_dir)
            raise e
    
    @staticmethod
    def cleanup_temp_dir(temp_dir):
        """
        Clean up a temporary directory.
        
        :param temp_dir: Path to the temporary directory to remove
        """
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    @classmethod
    def save_model_to_s3_(cls, local_model_path, s3_path, bucket_name, region_name="us-west-2", 
                        aws_access_key_id=None, aws_secret_access_key=None, 
                        endpoint_url=None):
        """
        Save a local BERTopic model file to S3.
        
        :param local_model_path: Path to the local model file
        :param s3_path: Path in S3 where to save the model
        :param bucket_name: S3 bucket name
        :param region_name: AWS region
        :param aws_access_key_id: AWS access key (optional, can use IAM role)
        :param aws_secret_access_key: AWS secret key (optional, can use IAM role)
        :param endpoint_url: Custom S3 endpoint URL (optional)
        :return: True if successful
        """
        # Get S3 client
        s3_client, bucket = cls._get_s3_client(
            bucket_name=bucket_name,
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            endpoint_url=endpoint_url
        )
        
        # Upload to S3
        s3_client.upload_file(local_model_path, bucket, s3_path)
        return True
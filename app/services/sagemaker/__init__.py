import boto3
import sagemaker
from sagemaker.transformer import Transformer
from sagemaker.model import Model
import time
from typing import List, Dict, Tuple, Optional
import json
import os
from contextlib import contextmanager

class SageMakerController:
    """Controls SageMaker infrastructure lifecycle"""
    
    def __init__(
        self,
        role_arn: str,
        model_name: str = 'fileloader-model',
        instance_type: str = 'ml.m5.xlarge',
        instance_count: int = 1
    ):
        self.sagemaker_session = sagemaker.Session()
        self.role = role_arn
        self.model_name = model_name
        self.instance_type = instance_type
        self.instance_count = instance_count
        
        # Initialize clients
        self.sagemaker_client = boto3.client('sagemaker')
        self.s3 = boto3.client('s3')
        
        # Track resources for cleanup
        self.resources_to_cleanup = []
        
    def _create_model(self, image_uri: str, model_data_url: str) -> None:
        """Create SageMaker model"""
        model = Model(
            image_uri=image_uri,
            model_data=model_data_url,
            role=self.role,
            name=self.model_name,
            sagemaker_session=self.sagemaker_session
        )
        model.create(
            instance_type=self.instance_type,
            environment={
                'SHAREPOINT_TENANT_ID': os.getenv('SHAREPOINT_TENANT_ID'),
                'SHAREPOINT_CLIENT_ID': os.getenv('SHAREPOINT_CLIENT_ID'),
                'SHAREPOINT_CLIENT_SECRET': os.getenv('SHAREPOINT_CLIENT_SECRET'),
                'SHAREPOINT_SITE_URL': os.getenv('SHAREPOINT_SITE_URL')
            }
        )
        self.resources_to_cleanup.append(('model', self.model_name))

    def _create_transform_job(
        self,
        input_location: str,
        output_location: str
    ) -> str:
        """Create and start batch transform job"""
        job_name = f"{self.model_name}-{int(time.time())}"
        
        transformer = Transformer(
            model_name=self.model_name,
            instance_count=self.instance_count,
            instance_type=self.instance_type,
            output_path=output_location,
            sagemaker_session=self.sagemaker_session
        )
        
        transformer.transform(
            data=input_location,
            content_type='application/json',
            split_type='Line'
        )
        
        self.resources_to_cleanup.append(('transform-job', job_name))
        return job_name

    def _wait_for_transform_job(self, job_name: str, timeout: int = 3600) -> bool:
        """Wait for batch transform job completion"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            response = self.sagemaker_client.describe_transform_job(
                TransformJobName=job_name
            )
            status = response['TransformJobStatus']
            
            if status == 'Completed':
                return True
            elif status in ['Failed', 'Stopped']:
                raise Exception(f"Job failed with status {status}")
                
            time.sleep(30)
            
        raise TimeoutError("Transform job timed out")

    def _cleanup_resources(self) -> None:
        """Clean up all created resources"""
        for resource_type, resource_name in self.resources_to_cleanup:
            try:
                if resource_type == 'model':
                    self.sagemaker_client.delete_model(ModelName=resource_name)
                elif resource_type == 'transform-job':
                    try:
                        self.sagemaker_client.stop_transform_job(
                            TransformJobName=resource_name
                        )
                    except:
                        pass  # Job might be already completed
            except Exception as e:
                print(f"Error cleaning up {resource_type} {resource_name}: {str(e)}")

    @contextmanager
    def process_files(
        self,
        file_paths: List[str],
        image_uri: str,
        model_data_url: str,
        timeout: int = 3600
    ) -> Tuple[List[str], List[Dict]]:
        """
        Process files using SageMaker with automatic cleanup
        
        Args:
            file_paths: List of file paths to process
            image_uri: URI of the Docker image containing FileLoader
            model_data_url: S3 URL of model artifacts
            timeout: Maximum time to wait for completion
            
        Returns:
            Tuple of (document_contents, metadata_list)
        """
        try:
            # Create model
            self._create_model(image_uri, model_data_url)
            
            # Prepare input data
            input_location = self._prepare_input(file_paths)
            output_location = f"s3://{self.sagemaker_session.default_bucket()}/output/"
            
            # Start transform job
            job_name = self._create_transform_job(input_location, output_location)
            
            # Wait for completion
            self._wait_for_transform_job(job_name, timeout)
            
            # Get results
            results = self._get_results(output_location, job_name)
            
            yield results
            
        finally:
            # Cleanup all resources
            self._cleanup_resources()
            
    def _prepare_input(self, file_paths: List[str]) -> str:
        """Prepare input data in S3"""
        bucket = self.sagemaker_session.default_bucket()
        key = f"input/batch_{int(time.time())}.jsonl"
        
        # Create JSONL file with file paths
        content = '\n'.join(json.dumps({'file_path': path}) for path in file_paths)
        
        # Upload to S3
        self.s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=content.encode('utf-8')
        )
        
        return f"s3://{bucket}/{key}"
        
    def _get_results(
        self,
        output_location: str,
        job_name: str
    ) -> Tuple[List[str], List[Dict]]:
        """Get and parse job results"""
        # Download results from S3
        bucket, prefix = self._parse_s3_url(output_location)
        response = self.s3.get_object(
            Bucket=bucket,
            Key=f"{prefix}/{job_name}.out"
        )
        
        # Parse results
        results = response['Body'].read().decode('utf-8').splitlines()
        documents = []
        metadata = []
        
        for result in results:
            parsed = json.loads(result)
            documents.append(parsed['document'])
            metadata.append(parsed['metadata'])
            
        return documents, metadata
        
    def _parse_s3_url(self, url: str) -> Tuple[str, str]:
        """Parse S3 URL into bucket and prefix"""
        parts = url.replace('s3://', '').split('/')
        return parts[0], '/'.join(parts[1:])
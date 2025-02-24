import boto3
import sagemaker
from sagemaker.model import Model
import time
from typing import List, Dict, Tuple, Optional, Any, Union
import json
import os
from contextlib import contextmanager
from ..fileloader.models import FileLocation, StorageType

class SageMakerFileLoader:
    """Loads and processes files using SageMaker inference endpoints"""
    
    def __init__(
        self,
        role_arn: str,
        model_name: str = 'fileloader-model',
        endpoint_name: Optional[str] = None,
        instance_type: str = 'ml.m5.xlarge',
        instance_count: int = 1,
        sagemaker_session = None  # Add this parameter
    ):
        """
        Initialize the SageMaker file loader
        
        Args:
            role_arn: IAM role ARN with SageMaker permissions
            model_name: Name for the SageMaker model
            endpoint_name: Name for the SageMaker endpoint (defaults to model_name + timestamp)
            instance_type: EC2 instance type to use
            instance_count: Number of instances to deploy
        """
        self.sagemaker_session = sagemaker_session or sagemaker.Session()
        self.role = role_arn
        self.model_name = model_name
        self.endpoint_name = endpoint_name or f"{model_name}-{int(time.time())}"
        self.endpoint_config_name = f"{self.endpoint_name}-config"
        self.instance_type = instance_type
        self.instance_count = instance_count
        
        # Initialize clients
        region = self.sagemaker_session.boto_region_name
        self.sagemaker_client = boto3.client('sagemaker', region_name=region)
        self.sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=region)
        self.s3 = boto3.client('s3', region_name=region)
        
        # Track resources for cleanup
        self.resources_to_cleanup = []
        
    def _create_model(self,SHAREPOINT_SITE_URL: str, image_uri: str ='767828763507.dkr.ecr.us-west-2.amazonaws.com/my-sagemaker-image:5973531',
                      model_data_url: Optional[str] = None) -> None:
        """
        Create SageMaker model
        
        Args:
            image_uri: URI of the Docker image containing the inference code
            model_data_url: S3 URL of model artifacts (optional)
        """
        print(f"Creating model: {self.model_name}")
        
        # Set up environment variables
        environment = {
            'SHAREPOINT_TENANT_ID': os.getenv('SHAREPOINT_TENANT_ID', ''),
            'SHAREPOINT_CLIENT_ID': os.getenv('SHAREPOINT_CLIENT_ID', ''),
            'SHAREPOINT_CLIENT_SECRET': os.getenv('SHAREPOINT_CLIENT_SECRET', ''),
            'SHAREPOINT_SITE_URL': SHAREPOINT_SITE_URL,
            'HF_TOKEN': os.getenv('HF_TOKEN', ''),
            'ENABLE_CAPTIONING': os.getenv('ENABLE_CAPTIONING', 'true'),
            'MODEL_TYPE': os.getenv('MODEL_TYPE', 'transformer'),
            'CONFIDENCE_THRESHOLD': os.getenv('CONFIDENCE_THRESHOLD', '0.4')
        }
        
        model = Model(
            image_uri=image_uri,
            model_data=model_data_url,
            role=self.role,
            name=self.model_name,
            sagemaker_session=self.sagemaker_session,
            env=environment
        )
        
        model.create(instance_type=self.instance_type)
        self.resources_to_cleanup.append(('model', self.model_name))

    def _create_endpoint_config(self) -> None:
        """Create endpoint configuration"""
        print(f"Creating endpoint configuration: {self.endpoint_config_name}")
        
        self.sagemaker_client.create_endpoint_config(
            EndpointConfigName=self.endpoint_config_name,
            ProductionVariants=[
                {
                    'VariantName': 'AllTraffic',
                    'ModelName': self.model_name,
                    'InstanceType': self.instance_type,
                    'InitialInstanceCount': self.instance_count,
                }
            ]
        )
        
        self.resources_to_cleanup.append(('endpoint-config', self.endpoint_config_name))

    def _create_endpoint(self) -> None:
        """Create and deploy endpoint"""
        print(f"Creating endpoint: {self.endpoint_name}")
        
        self.sagemaker_client.create_endpoint(
            EndpointName=self.endpoint_name,
            EndpointConfigName=self.endpoint_config_name
        )
        
        self.resources_to_cleanup.append(('endpoint', self.endpoint_name))

    def _wait_for_endpoint(self, timeout: int = 1800) -> bool:
        """
        Wait for endpoint to become available
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            bool: True if endpoint is available, raises exception otherwise
        """
        print(f"Waiting for endpoint {self.endpoint_name} to become available...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            response = self.sagemaker_client.describe_endpoint(
                EndpointName=self.endpoint_name
            )
            status = response['EndpointStatus']
            
            if status == 'InService':
                print(f"Endpoint {self.endpoint_name} is now available")
                return True
            elif status in ['Failed', 'OutOfService']:
                failure_reason = response.get('FailureReason', 'Unknown reason')
                raise Exception(f"Endpoint creation failed: {failure_reason}")
                
            print(f"Endpoint status: {status}. Waiting...")
            time.sleep(30)
            
        raise TimeoutError("Endpoint creation timed out")

    def _invoke_endpoint(self, file_location: Union[Dict, FileLocation]) -> Dict:
        """
        Invoke the endpoint with file location
        
        Args:
            file_location: FileLocation object or dict with storage_type and path
            
        Returns:
            Dict: Response from the endpoint
        """
        # Convert file_location to the format expected by the endpoint
        if isinstance(file_location, FileLocation):
            # Convert Enum to string value for JSON serialization
            payload = {
                "file_location": {
                    "path": file_location.path,
                    "storage_type": file_location.storage_type.value,
                    "version": file_location.version
                }
            }
        else:
            # Assume it's already in the right format
            payload = {"file_location": file_location}
            
        print(f"Invoking endpoint with payload: {payload}")
        
        # Invoke the endpoint
        response = self.sagemaker_runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        
        # Parse and return the response
        result = json.loads(response['Body'].read().decode())
        return result

    def _cleanup_resources(self) -> None:
        """Clean up all created resources"""
        print("Cleaning up resources...")
        
        # Process resources in reverse to handle dependencies correctly
        for resource_type, resource_name in reversed(self.resources_to_cleanup):
            try:
                if resource_type == 'endpoint':
                    print(f"Deleting endpoint: {resource_name}")
                    self.sagemaker_client.delete_endpoint(EndpointName=resource_name)
                    # Wait for endpoint to be deleted
                    wait_time = 0
                    while wait_time < 300:  # 5 minutes max wait
                        try:
                            self.sagemaker_client.describe_endpoint(EndpointName=resource_name)
                            time.sleep(10)
                            wait_time += 10
                        except self.sagemaker_client.exceptions.ClientError:
                            # Endpoint not found, it's been deleted
                            break
                            
                elif resource_type == 'endpoint-config':
                    print(f"Deleting endpoint configuration: {resource_name}")
                    self.sagemaker_client.delete_endpoint_config(
                        EndpointConfigName=resource_name
                    )
                elif resource_type == 'model':
                    print(f"Deleting model: {resource_name}")
                    self.sagemaker_client.delete_model(ModelName=resource_name)
            except Exception as e:
                print(f"Error cleaning up {resource_type} {resource_name}: {str(e)}")

    @contextmanager
    def deploy_model(
        self,
        sharepoint_site_uri: str,
        image_uri: str,
        model_data_url: Optional[str] = None,
        timeout: int = 1200
    ) -> str:
        """
        Deploy a model to SageMaker endpoint with automatic cleanup
        
        Args:
            image_uri: URI of the Docker image containing the inference code
            model_data_url: S3 URL of model artifacts (optional)
            timeout: Maximum time to wait for endpoint to be available
            
        Returns:
            str: Endpoint name to use for invocations
        """
        try:
            # Create model
            self._create_model(sharepoint_site_uri, image_uri, model_data_url)
            
            # Create endpoint configuration
            self._create_endpoint_config()
            
            # Create and deploy endpoint
            self._create_endpoint()
            
            # Wait for endpoint to become available
            self._wait_for_endpoint(timeout)
            
            yield self.endpoint_name
            
        finally:
            # Cleanup all resources
            self._cleanup_resources()
    
    def process_files(
        self,
        file_locations: List[Union[Dict, FileLocation]],
        image_uri: str,
        model_data_url: Optional[str] = None,
        timeout: int = 1800
    ) -> List[Dict]:
        """
        Process multiple files using a SageMaker endpoint
        
        Args:
            file_locations: List of FileLocation objects or dicts to process
            image_uri: URI of the Docker image containing the inference code
            model_data_url: S3 URL of model artifacts (optional)
            timeout: Maximum time to wait for endpoint availability
            
        Returns:
            List[Dict]: Results from processing each file
        """
        results = []
        
        with self.deploy_model(image_uri, model_data_url, timeout) as endpoint_name:
            print(f"Processing {len(file_locations)} files using endpoint: {endpoint_name}")
            
            for i, file_location in enumerate(file_locations):
                print(f"Processing file {i+1}/{len(file_locations)}")
                result = self._invoke_endpoint(file_location)
                results.append(result)
                
        return results
    
    def process_single_file(
        self,
        file_location: Union[Dict, FileLocation],
        image_uri: str, 
        model_data_url: Optional[str] = None,
        timeout: int = 1800
    ) -> Dict:
        """
        Process a single file using a SageMaker endpoint
        
        Args:
            file_location: FileLocation object or dict to process
            image_uri: URI of the Docker image containing the inference code 
            model_data_url: S3 URL of model artifacts (optional)
            timeout: Maximum time to wait for endpoint availability
            
        Returns:
            Dict: Result from processing the file
        """
        with self.deploy_model(image_uri, model_data_url, timeout) as endpoint_name:
            print(f"Processing file using endpoint: {endpoint_name}")
            return self._invoke_endpoint(file_location)
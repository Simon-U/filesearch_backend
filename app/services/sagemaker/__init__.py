import boto3
import sagemaker
from botocore.config import Config
from sagemaker.model import Model
import time
import uuid
from typing import List, Dict, Tuple, Optional, Any, Union
import json
import os
import logging
from contextlib import contextmanager
from ..fileloader.models import FileLocation, StorageType

# Configure logger
logger = logging.getLogger("sagemaker_fileloader")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class SageMakerFileLoader:
    """Loads and processes files using SageMaker async inference endpoints"""
    
    def __init__(
        self,
        role_arn: str,
        model_name: str = 'fileloader-model',
        endpoint_name: Optional[str] = None,
        instance_type: str = 'ml.g4dn.2xlarge',
        instance_count: int = 1,
        sagemaker_session = None,
        s3_bucket: str = 'fileloader',
        s3_input_path: str = 'input',
        s3_output_path: str = 'output',
        max_concurrent_invocations: int = 4
    ):
        """
        Initialize the SageMaker file loader for async processing
        
        Args:
            role_arn: IAM role ARN with SageMaker permissions
            model_name: Name for the SageMaker model
            endpoint_name: Name for the SageMaker endpoint (defaults to model_name + timestamp)
            instance_type: EC2 instance type to use
            instance_count: Number of instances to deploy
            sagemaker_session: SageMaker session to use
            s3_bucket: S3 bucket for async inference I/O (default: 'fileloader')
            s3_input_path: S3 prefix for async inference input (default: 'input')
            s3_output_path: S3 prefix for async inference output (default: 'output')
            max_concurrent_invocations: Maximum concurrent invocations per instance
        """
        # Initialize SageMaker session and variables
        self.sagemaker_session = sagemaker_session or sagemaker.Session()
        self.role = role_arn
        self.model_name = model_name
        self.endpoint_name = endpoint_name or f"{model_name}-{int(time.time())}"
        self.endpoint_config_name = f"{self.endpoint_name}-config"
        self.instance_type = instance_type
        self.instance_count = instance_count
        
        # S3 paths configuration
        self.s3_bucket = s3_bucket
        self.s3_input_path = s3_input_path
        self.s3_output_path = s3_output_path
        self.max_concurrent_invocations = max_concurrent_invocations
        
        # Initialize clients with appropriate timeouts for async processing
        region = self.sagemaker_session.boto_region_name
        self.sagemaker_client = boto3.client('sagemaker', region_name=region)
        
        boto_config = Config(
            read_timeout=3600,  # 1 hour
            connect_timeout=60,
            retries={'max_attempts': 0}
        )
        
        self.sagemaker_runtime = boto3.client(
            'sagemaker-runtime', 
            region_name=region,
            config=boto_config
        )
        self.s3 = boto3.client('s3', region_name=region)
        
        # Track resources for cleanup
        self.resources_to_cleanup = []
        
        logger.info(f"Initialized SageMakerFileLoader with bucket: {s3_bucket}, endpoint: {self.endpoint_name}")
        
    def _create_model(self, sharepoint_site_uri: str, image_uri: str, model_data_url: Optional[str] = None) -> None:
        """
        Create SageMaker model
        
        Args:
            sharepoint_site_uri: URL of the SharePoint site
            image_uri: URI of the Docker image containing the inference code
            model_data_url: S3 URL of model artifacts (optional)
        """
        logger.info(f"Creating model: {self.model_name}")
        
        # Set up environment variables
        environment = {
            'SHAREPOINT_TENANT_ID': os.getenv('SHAREPOINT_TENANT_ID', ''),
            'SHAREPOINT_CLIENT_ID': os.getenv('SHAREPOINT_CLIENT_ID', ''),
            'SHAREPOINT_CLIENT_SECRET': os.getenv('SHAREPOINT_CLIENT_SECRET', ''),
            'SHAREPOINT_SITE_URL': sharepoint_site_uri,
            'HF_TOKEN': os.getenv('HF_TOKEN', ''),
            'ENABLE_CAPTIONING': os.getenv('ENABLE_CAPTIONING', 'true'),
            'MODEL_TYPE': os.getenv('MODEL_TYPE', 'transformer'),
            'CONFIDENCE_THRESHOLD': os.getenv('CONFIDENCE_THRESHOLD', '0.4'),
            'SAGEMAKER_MODEL_SERVER_TIMEOUT': '1800',  # 30 minute timeout
            'SAGEMAKER_MODEL_SERVER_WORKERS': '1'
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
        logger.info(f"Model {self.model_name} created successfully")

    def _create_endpoint_config(self) -> None:
        """Create asynchronous endpoint configuration"""
        logger.info(f"Creating asynchronous endpoint configuration: {self.endpoint_config_name}")
        
        # Set up async-specific configuration
        s3_output_path = f"s3://{self.s3_bucket}/{self.s3_output_path}"
        
        config_args = {
            'EndpointConfigName': self.endpoint_config_name,
            'ProductionVariants': [
                {
                    'VariantName': 'AllTraffic',
                    'ModelName': self.model_name,
                    'InstanceType': self.instance_type,
                    'InitialInstanceCount': self.instance_count,
                }
            ],
            'AsyncInferenceConfig': {
                'OutputConfig': {
                    'S3OutputPath': s3_output_path,
                    # Optional notification config if you want SNS notifications
                    # 'NotificationConfig': {
                    #     'SuccessTopic': 'arn:aws:sns:region:account:success-topic',
                    #     'ErrorTopic': 'arn:aws:sns:region:account:error-topic'
                    # }
                },
                'ClientConfig': {
                    'MaxConcurrentInvocationsPerInstance': self.max_concurrent_invocations
                }
            }
        }
        
        self.sagemaker_client.create_endpoint_config(**config_args)
        self.resources_to_cleanup.append(('endpoint-config', self.endpoint_config_name))
        logger.info(f"Endpoint configuration {self.endpoint_config_name} created successfully")

    def _create_endpoint(self) -> None:
        """Create and deploy endpoint"""
        logger.info(f"Creating endpoint: {self.endpoint_name}")
        
        self.sagemaker_client.create_endpoint(
            EndpointName=self.endpoint_name,
            EndpointConfigName=self.endpoint_config_name
        )
        
        self.resources_to_cleanup.append(('endpoint', self.endpoint_name))
        logger.info(f"Endpoint {self.endpoint_name} creation initiated")

    def _wait_for_endpoint(self, timeout: int = 1800) -> bool:
        """
        Wait for endpoint to become available
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            bool: True if endpoint is available, raises exception otherwise
        """
        logger.info(f"Waiting for endpoint {self.endpoint_name} to become available...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            response = self.sagemaker_client.describe_endpoint(
                EndpointName=self.endpoint_name
            )
            status = response['EndpointStatus']
            
            if status == 'InService':
                logger.info(f"Endpoint {self.endpoint_name} is now available")
                return True
            elif status in ['Failed', 'OutOfService']:
                failure_reason = response.get('FailureReason', 'Unknown reason')
                logger.error(f"Endpoint creation failed: {failure_reason}")
                raise Exception(f"Endpoint creation failed: {failure_reason}")
                
            logger.info(f"Endpoint status: {status}. Waiting...")
            time.sleep(30)
            
        logger.error(f"Endpoint creation timed out after {timeout} seconds")
        raise TimeoutError("Endpoint creation timed out")
        
    def _upload_file_to_s3(self, file_content: str, file_name: str) -> str:
        """
        Upload input file to S3 for async inference
        
        Args:
            file_content: Content of the file to upload
            file_name: Name to give the file in S3
            
        Returns:
            str: S3 URI of the uploaded file
        """
        s3_key = f"{self.s3_input_path}/{file_name}"
        
        # Upload the file
        self.s3.put_object(
            Bucket=self.s3_bucket,
            Key=s3_key,
            Body=file_content
        )
        
        s3_uri = f"s3://{self.s3_bucket}/{s3_key}"
        logger.info(f"Uploaded file to {s3_uri}")
        
        # Return the S3 URI
        return s3_uri

    def _invoke_endpoint_async(self, file_location: Union[Dict, FileLocation], config: Optional[Dict] = None) -> Dict:
        """
        Invoke the endpoint asynchronously
        
        Args:
            file_location: FileLocation object or dict with storage_type and path
            config: Optional configuration parameters to pass to the model
            
        Returns:
            Dict: Response with request ID and output location
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
            
        # Add configuration if provided
        if config:
            payload["config"] = config
            
        logger.info(f"Invoking endpoint asynchronously with payload: {payload}")
        
        # Generate a unique file name for this request
        request_id = str(uuid.uuid4())
        input_file_name = f"{request_id}.json"
        
        # Upload payload to S3
        input_location = self._upload_file_to_s3(
            file_content=json.dumps(payload),
            file_name=input_file_name
        )
        
        # Invoke the endpoint asynchronously
        response = self.sagemaker_runtime.invoke_endpoint_async(
            EndpointName=self.endpoint_name,
            ContentType='application/json',
            InputLocation=input_location
        )
        
        logger.info(f"Async request submitted. Request ID: {response['InferenceId']}")
        
        # Return the response information for checking status later
        return {
            'request_id': response['InferenceId'],
            'output_location': response.get('OutputLocation', ''),
            'status': 'InProgress'
        }

    def check_async_result(self, request_id: str) -> Dict:
        """
        Check the status of an asynchronous inference request
        
        Args:
            request_id: The inference ID returned from invoke_endpoint_async
            
        Returns:
            Dict: Status of the request and result if available
        """
        start_time = time.time()
        logger.info(f"Checking status of async request: {request_id}")
        
        # Parse the output location pattern
        output_prefix = f"s3://{self.s3_bucket}/{self.s3_output_path}"
        output_key_prefix = f"{self.s3_output_path}"
        
        result = {
            'request_id': request_id,
            'status': 'InProgress',
            'output_location': f"{output_prefix}/{request_id}"
        }
        
        try:
            # Check for the output file with the request_id as filename
            output_file_key = f"{output_key_prefix}/{request_id}.out"
            output_failure_key = f"{output_key_prefix}/{request_id}.failure"
            
            # First, check for the .out file (success case)
            try:
                logger.info(f"Checking for output file: {output_file_key}")
                s3_response = self.s3.get_object(
                    Bucket=self.s3_bucket, 
                    Key=output_file_key
                )
                # If we get here, the output file exists
                result['status'] = 'Completed'
                result_content = s3_response['Body'].read().decode()
                result['result'] = json.loads(result_content)
                logger.info(f"Found completed result file, size: {len(result_content)} bytes")
                
                # Add processing time to the result
                result['processing_time'] = time.time() - start_time
                return result
            except self.s3.exceptions.NoSuchKey:
                logger.info(f"Output file {output_file_key} not found yet")
                
            # Then check for the .failure file (error case)
            try:
                #logger.info(f"Checking for failure file: {output_failure_key}")
                s3_response = self.s3.get_object(
                    Bucket=self.s3_bucket, 
                    Key=output_failure_key
                )
                # If we get here, the failure file exists
                result['status'] = 'Failed'
                error_content = s3_response['Body'].read().decode()
                result['failure_reason'] = error_content
                logger.error(f"Found failure file with content: {error_content}")
                
                # Add processing time to the result
                result['processing_time'] = time.time() - start_time
                return result
            except self.s3.exceptions.NoSuchKey:
                logger.info(f"Failure file {output_failure_key} not found")
                
            # If we get here, neither file was found - try listing objects to see if 
            # there's anything with a matching prefix that might indicate progress
            logger.info(f"Listing objects in bucket with prefix: {output_key_prefix}")
            response = self.s3.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix=f"{output_key_prefix}/{request_id}"
            )
            
            if 'Contents' in response and len(response['Contents']) > 0:
                logger.info(f"Found {len(response['Contents'])} related objects")
                for obj in response['Contents']:
                    logger.info(f"Found related object: {obj['Key']}")
            else:
                logger.info("No related objects found")
        
        except Exception as e:
            error_msg = f"Error checking async result: {str(e)}"
            logger.error(error_msg, exc_info=True)
            result['fetch_error'] = error_msg
        
        # Add processing time to the result
        result['processing_time'] = time.time() - start_time
        return result

    def wait_for_async_result(self, request_id: str, check_interval: int = 20, timeout: int = 3600) -> Dict:
        """
        Wait for an asynchronous inference request to complete
        
        Args:
            request_id: The inference ID returned from invoke_endpoint_async
            check_interval: How often to check the status in seconds
            timeout: Maximum time to wait in seconds
            
        Returns:
            Dict: Result of the async inference
        """
        logger.info(f"Waiting for async inference {request_id} to complete...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            result = self.check_async_result(request_id)
            status = result['status']
            
            if status == 'Completed':
                logger.info(f"Async inference {request_id} completed successfully")
                return result
            elif status == 'Failed':
                failure_reason = result.get('failure_reason', 'Unknown reason')
                logger.error(f"Async inference failed: {failure_reason}")
                raise Exception(f"Async inference failed: {failure_reason}")
            
            elapsed = time.time() - start_time
            logger.info(f"Async inference status: {status}. Elapsed: {elapsed:.1f}s. Waiting {check_interval}s...")
            time.sleep(check_interval)
            
        logger.error(f"Async inference {request_id} timed out after {timeout} seconds")
        raise TimeoutError(f"Async inference {request_id} timed out after {timeout} seconds")

    def _cleanup_resources(self) -> None:
        """Clean up all created resources"""
        logger.info("Cleaning up resources...")
        
        # Process resources in reverse to handle dependencies correctly
        for resource_type, resource_name in reversed(self.resources_to_cleanup):
            try:
                if resource_type == 'endpoint':
                    logger.info(f"Deleting endpoint: {resource_name}")
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
                    logger.info(f"Deleting endpoint configuration: {resource_name}")
                    self.sagemaker_client.delete_endpoint_config(
                        EndpointConfigName=resource_name
                    )
                elif resource_type == 'model':
                    logger.info(f"Deleting model: {resource_name}")
                    self.sagemaker_client.delete_model(ModelName=resource_name)
            except Exception as e:
                logger.error(f"Error cleaning up {resource_type} {resource_name}: {str(e)}", exc_info=True)
        
        logger.info("Resource cleanup completed")
        
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
            sharepoint_site_uri: URL of the SharePoint site
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
        sharepoint_site_uri: str,
        image_uri: str,
        model_data_url: Optional[str] = None,
        endpoint_timeout: int = 1800,
        processing_timeout: int = 3600,
        wait_for_results: bool = True,
        config: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Process multiple files using a SageMaker async endpoint
        
        Args:
            file_locations: List of FileLocation objects or dicts to process
            sharepoint_site_uri: URL of the SharePoint site
            image_uri: URI of the Docker image containing the inference code
            model_data_url: S3 URL of model artifacts (optional)
            endpoint_timeout: Maximum time to wait for endpoint availability
            processing_timeout: Maximum time to wait for processing to complete
            wait_for_results: Whether to wait for processing to complete
            config: Optional configuration parameters to pass to the model
            
        Returns:
            List[Dict]: Results from processing each file
        """
        results = []
        
        with self.deploy_model(sharepoint_site_uri, image_uri, model_data_url, endpoint_timeout) as endpoint_name:
            logger.info(f"Processing {len(file_locations)} files using asynchronous endpoint: {endpoint_name}")
            
            for i, file_location in enumerate(file_locations):
                logger.info(f"Processing file {i+1}/{len(file_locations)}")
                
                # Submit the job
                result = self._invoke_endpoint_async(file_location, config)
                
                # If wait_for_results is True, wait for the result
                if wait_for_results and 'request_id' in result:
                    result = self.wait_for_async_result(
                        result['request_id'], 
                        timeout=processing_timeout
                    )
                
                results.append(result)
                
        return results
    
    def process_single_file(
        self,
        file_location: Union[Dict, FileLocation],
        sharepoint_site_uri: str,
        image_uri: str, 
        model_data_url: Optional[str] = None,
        endpoint_timeout: int = 1800,
        processing_timeout: int = 3600,
        wait_for_result: bool = True,
        config: Optional[Dict] = None
    ) -> Dict:
        """
        Process a single file using a SageMaker async endpoint
        
        Args:
            file_location: FileLocation object or dict to process
            sharepoint_site_uri: URL of the SharePoint site
            image_uri: URI of the Docker image containing the inference code 
            model_data_url: S3 URL of model artifacts (optional)
            endpoint_timeout: Maximum time to wait for endpoint availability
            processing_timeout: Maximum time to wait for processing to complete
            wait_for_result: Whether to wait for processing to complete
            config: Optional configuration parameters to pass to the model
            
        Returns:
            Dict: Result from processing the file
        """
        with self.deploy_model(sharepoint_site_uri, image_uri, model_data_url, endpoint_timeout) as endpoint_name:
            logger.info(f"Processing file using asynchronous endpoint: {endpoint_name}")
            
            # Submit the job
            result = self._invoke_endpoint_async(file_location, config)
            
            # If wait_for_result is True, wait for the result
            if wait_for_result and 'request_id' in result:
                result = self.wait_for_async_result(
                    result['request_id'], 
                    timeout=processing_timeout
                )
            
            return result
import os
import json
import torch
import logging
from fileloader.loader import FileLoader
from fileloader.models import SharePointConfig, FileLocation, StorageType
from fileloader.image_processor.analyzer import AnalyzerConfig
from sagemaker_inference import default_inference_handler

# Set up basic logging to stdout
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

class MyHandler(default_inference_handler.DefaultInferenceHandler):
    def get_analyzer_config(self):
        """Get analyzer configuration from environment variables or defaults"""
        return AnalyzerConfig(
            hf_token=os.getenv('HF_TOKEN'),
            model_type=os.getenv('MODEL_TYPE', "transformer"),
            model_name=os.getenv('MODEL_NAME', "openai/clip-vit-base-patch32"),
            confidence_threshold=float(os.getenv('CONFIDENCE_THRESHOLD', "0.4")),
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_half_precision=os.getenv('USE_HALF_PRECISION', 'true').lower() == 'true',
            optimize_memory_usage=True,
            enable_captioning=os.getenv('ENABLE_CAPTIONING', 'true').lower() == 'true',
            caption_model=os.getenv('CAPTION_MODEL', 'Salesforce/blip-image-captioning-base')
        )

    def model_fn(self, model_dir):
        """
        Initialize the FileLoader with configurations.
        This is called by SageMaker when starting the model server.
        """
        # Get SharePoint configuration from environment variables
        sharepoint_config = SharePointConfig(
            tenant_id=os.environ.get('SHAREPOINT_TENANT_ID'),
            client_id=os.environ.get('SHAREPOINT_CLIENT_ID'),
            client_secret=os.environ.get('SHAREPOINT_CLIENT_SECRET'),
            site_url=os.environ.get('SHAREPOINT_SITE_URL')
        )
        
        # Get analyzer configuration
        analyzer_config = self.get_analyzer_config()
        
        # Initialize FileLoader with configurations
        loader = FileLoader(
            config=sharepoint_config,
            num_threads=int(os.getenv('NUM_THREADS', '8')),
            use_cuda=torch.cuda.is_available(),
            do_table_structure=True,
            do_ocr=True,
            do_image_enrichment=True,
            image_analyzer_config=analyzer_config
        )
        
        return loader

    def input_fn(self, request_body, request_content_type):
        """
        Parse input data coming from SageMaker invocations.
        Expects JSON with file_location.
        """
        if request_content_type != 'application/json':
            raise ValueError(f'Unsupported content type: {request_content_type}')
        
        # Parse the incoming JSON request
        input_data = json.loads(request_body)
        
        # Validate input
        if 'file_location' not in input_data:
            raise ValueError("Input must contain 'file_location'")
        
        # Convert dictionary to FileLocation object if needed
        file_location_data = input_data['file_location']
        if isinstance(file_location_data, dict):
            # Convert string storage_type to Enum
            storage_type_str = file_location_data.get('storage_type', 'local')
            storage_type = StorageType(storage_type_str)
            
            file_location = FileLocation(
                path=file_location_data['path'],
                storage_type=storage_type,
                version=file_location_data.get('version')
            )
            return file_location
        
        return input_data['file_location']

    def predict_fn(self, file_location, model):
        """
        Process file using FileLoader.
        Args:
            file_location: FileLocation object or path
            model: Initialized FileLoader instance
        """
        try:
            # Process the file
            document, metadata = model.load(file_location)
            
            # Prepare successful response
            return {
                'status': 'success',
                'document': document,
                'metadata': metadata,
                'error': None
            }
        except Exception as e:
            # Handle any errors during processing
            return {
                'status': 'error',
                'document': None,
                'metadata': None,
                'error': str(e)
            }

    def output_fn(self, prediction_output, accept):
        """
        Format the output to return to SageMaker client.
        Args:
            prediction_output: Dictionary containing results
            accept: Accept header from the client
        """
        if accept == 'application/json':
            return json.dumps(prediction_output), 'application/json'
        
        raise ValueError(f'Unsupported accept header: {accept}')
    
    def __call__(self, request, context):
        # Log the incoming request and context details
        logger.info("Received request: %s", request)
        logger.info("Received context: %s", context)
        try:
            # Attempt to extract the content type; if not available, default to 'application/json'
            content_type = getattr(context, 'content_type', 'application/json')
        except Exception as e:
            logger.error("Error extracting content type: %s", e)
            content_type = 'application/json'
        logger.info("Using content type: %s", content_type)
        # Process input, prediction, and output
        input_data = self.input_fn(request, content_type)
        # Ensure that the model is loaded. This example assumes you have already set self.model.
        prediction = self.predict_fn(input_data, self.model)
        output = self.output_fn(prediction, content_type)
        logger.info("Returning output: %s", output)
        return output

default_handler = MyHandler()

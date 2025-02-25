import os
import json
import torch
import logging
from fileloader.loader import FileLoader
from fileloader.models import SharePointConfig, FileLocation, StorageType
from fileloader.image_processor.analyzer import AnalyzerConfig
from sagemaker_inference import content_types

# Set up basic logging to stdout
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# Global file loader instance to share between functions
_file_loader = None
_initialized = False

def get_analyzer_config():
    """Get analyzer configuration from environment variables or defaults."""
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

def model_fn(model_dir):
    """
    Initialize the FileLoader with configurations.
    This is called by SageMaker when starting the model server.
    
    Args:
        model_dir: Directory where model artifacts are stored
        
    Returns:
        The initialized FileLoader instance
    """
    global _file_loader, _initialized
    
    if _initialized:
        logger.info("FileLoader already initialized")
        return _file_loader
    
    logger.info(f"Initializing FileLoader with model_dir: {model_dir}")
    
    # Get SharePoint configuration from environment variables
    sharepoint_config = SharePointConfig(
        tenant_id=os.environ.get('SHAREPOINT_TENANT_ID'),
        client_id=os.environ.get('SHAREPOINT_CLIENT_ID'),
        client_secret=os.environ.get('SHAREPOINT_CLIENT_SECRET'),
        site_url=os.environ.get('SHAREPOINT_SITE_URL')
    )
    
    # Get analyzer configuration
    analyzer_config = get_analyzer_config()
    
    logger.info("Creating FileLoader instance")
    
    # Initialize FileLoader
    _file_loader = FileLoader(
        config=sharepoint_config,
        num_threads=int(os.getenv('NUM_THREADS', '8')),
        use_cuda=torch.cuda.is_available(),
        do_table_structure=True,
        do_ocr=True,
        do_image_enrichment=True,
        image_analyzer_config=analyzer_config
    )
    
    _initialized = True
    logger.info("FileLoader successfully initialized")
    return _file_loader

def input_fn(request_body, request_content_type):
    """
    Transform input data to a FileLocation object.
    Expects JSON with a 'file_location' key.
    
    Args:
        request_body: The request payload
        request_content_type: The content type of the request
        
    Returns:
        FileLocation object for document processing
    """
    logger.info(f"input_fn called with content_type: {request_content_type}")
    logger.info(f"request_body: {request_body}")
    
    if request_content_type != content_types.JSON:
        raise ValueError(f'Unsupported content type: {request_content_type}')
    
    try:
        # Decode JSON input
        input_data = json.loads(request_body)
        logger.info(f"Parsed JSON input: {input_data}")
        
        if 'file_location' not in input_data:
            raise ValueError("Input must contain 'file_location'")
        
        file_location_data = input_data['file_location']
        
        # Handle file location as dict or direct value
        if isinstance(file_location_data, dict):
            storage_type_str = file_location_data.get('storage_type', 'local')
            storage_type = StorageType(storage_type_str)
            file_location = FileLocation(
                path=file_location_data['path'],
                storage_type=storage_type,
                version=file_location_data.get('version')
            )
            logger.info(f"Created FileLocation: {file_location}")
            return file_location
        
        logger.info(f"Using provided file_location: {file_location_data}")
        return input_data['file_location']
        
    except Exception as e:
        logger.error(f"Error preprocessing request: {e}")
        raise

def predict_fn(file_location, file_loader):
    """
    Process file using FileLoader.
    
    Args:
        file_location: FileLocation object specifying the document to process
        file_loader: The FileLoader instance from model_fn
        
    Returns:
        Dictionary with processing results or error information
    """
    logger.info(f"predict_fn called with file_location: {file_location}")
    
    if file_loader is None:
        error_msg = "FileLoader not initialized, call model_fn first"
        logger.error(error_msg)
        return {
            'status': 'error',
            'document': None,
            'metadata': None,
            'error': error_msg
        }
    
    try:
        logger.info(f"Processing document at: {file_location}")
        document, metadata = file_loader.load(file_location)
        logger.info(f"Document processed successfully, metadata length: {len(str(metadata))}")
        
        return {
            'status': 'success',
            'document': document,
            'metadata': metadata,
            'error': None
        }
    except Exception as e:
        error_msg = f"Error processing document: {str(e)}"
        logger.error(error_msg)
        return {
            'status': 'error',
            'document': None,
            'metadata': None,
            'error': error_msg
        }

def output_fn(prediction, response_content_type):
    """
    Transform prediction output to the expected response format.
    
    Args:
        prediction: Model prediction from predict_fn
        response_content_type: Expected content type of the response
        
    Returns:
        Formatted response
    """
    logger.info(f"output_fn called with response_content_type: {response_content_type}")
    
    # Default to JSON if not specified
    content_type = response_content_type or content_types.JSON
    
    if content_type == content_types.JSON:
        logger.info("Formatting response as JSON")
        return json.dumps(prediction), content_types.JSON
    else:
        logger.warning(f"Unsupported content type: {content_type}, using JSON instead")
        return json.dumps(prediction), content_types.JSON
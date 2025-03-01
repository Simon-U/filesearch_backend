import os
import json
import torch
import logging
from datetime import datetime
from fileloader.loader import FileLoader
from fileloader.models import SharePointConfig, FileLocation, StorageType
from fileloader.image_processor.analyzer import AnalyzerConfig
from sagemaker_inference import content_types
from docling_core.types.doc import ImageRefMode

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

# Custom JSON encoder to handle datetime objects and other non-serializable types
class FileMetadataJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            # Handle custom objects by converting to dict
            return {key: value for key, value in obj.__dict__.items() 
                    if not key.startswith('_')}
        return super().default(obj)

def get_analyzer_config():
    """Get analyzer configuration from environment variables or defaults."""
    # Set S3 bucket if it's not set (for model caching)
    if 'MODEL_CACHE_BUCKET' not in os.environ and 'S3_BUCKET' in os.environ:
        os.environ['MODEL_CACHE_BUCKET'] = os.environ['S3_BUCKET']
        
    # Set model cache directory
    if 'MODEL_CACHE_DIR' not in os.environ:
        os.environ['MODEL_CACHE_DIR'] = '/opt/ml/models'
        
    # Ensure model cache directory exists
    os.makedirs(os.environ['MODEL_CACHE_DIR'], exist_ok=True)
    logger.info(f"Getting CONFIDENCE_THRESHOLD: {float(os.getenv('CONFIDENCE_THRESHOLD', 0.7))}")
    return AnalyzerConfig(
        hf_token=os.getenv('HF_TOKEN'),
        model_type=os.getenv('MODEL_TYPE', "transformer"),
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_half_precision=os.getenv('USE_HALF_PRECISION', 'true').lower() == 'true',
        optimize_memory_usage=True,
        #Classification
        classification_backend_type=os.getenv('CLASSIFICATION_BACKEND_TYPE', 'clip'),
        classification_model=os.getenv('CLASSIFICATION_MODEL', 'openai/clip-vit-base-patch32'),
        confidence_threshold=float(os.getenv('CONFIDENCE_THRESHOLD', 0.7)),
        #Caption backend
        enable_captioning=os.getenv('ENABLE_CAPTIONING', 'true').lower() == 'true',
        caption_backend_type=os.getenv('CAPTION_BACKEND_TYPE', 'blip'),
        caption_model=os.getenv('CAPTION_MODEL', 'Salesforce/blip-image-captioning-base'),
        max_caption_length=int(os.getenv('MAX_CAPTION_LENGTH', '400')),
        # Cache directory
        cache_dir=os.environ.get('MODEL_CACHE_DIR', '/opt/ml/models')
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
    
    logger.info(f"Creating FileLoader instance in model_fn {analyzer_config}")
    
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
    Expects JSON with a 'file_location' key and optional 'config' key.
    
    Args:
        request_body: The request payload
        request_content_type: The content type of the request
        
    Returns:
        FileLocation object for document processing
    """
    logger.info(f"input_fn called with content_type: {request_content_type}")
    logger.info(f"request_body (truncated): {request_body[:500] if len(request_body) > 500 else request_body}")
    
    from huggingface_hub import login
    login(token=os.environ.get('HF_TOKEN'))
    
    global _initialized, _file_loader
    
    if request_content_type != content_types.JSON:
        raise ValueError(f'Unsupported content type: {request_content_type}')
    
    try:
        # Decode JSON input
        input_data = json.loads(request_body)
        logger.info(f"Parsed JSON input (truncated): {str(input_data)[:500] if len(str(input_data)) > 500 else str(input_data)}")
        
        if 'file_location' not in input_data:
            raise ValueError("Input must contain 'file_location'")
        
        # Process config if provided
        if 'config' in input_data:
            config = input_data['config']
            logger.info(f"Found configuration in request: {config}")
            
            # Set configuration in environment variables
            if 'classification' in config:
                os.environ['CLASSIFICATION_BACKEND_TYPE'] = config['classification'].get('backend_type', os.environ.get('CLASSIFICATION_BACKEND_TYPE', 'clip'))
                os.environ['CLASSIFICATION_MODEL'] = config['classification'].get('classification_model', os.environ.get('CLASSIFICATION_MODEL', 'openai/clip-vit-base-patch32'))
                os.environ['MODEL_TYPE'] = config['classification'].get('model_type', os.environ.get('MODEL_TYPE', 'transformer'))
            
            if 'captioning' in config:
                os.environ['CAPTION_BACKEND_TYPE'] = config['captioning'].get('backend_type', os.environ.get('CAPTION_BACKEND_TYPE', 'smolvlm'))
                os.environ['CAPTION_MODEL'] = config['captioning'].get('model', os.environ.get('CAPTION_MODEL', 'HuggingFaceTB/SmolVLM-Instruct'))
                os.environ['ENABLE_CAPTIONING'] = str(config['captioning'].get('enabled', os.environ.get('ENABLE_CAPTIONING', 'true'))).lower()
                os.environ['MAX_CAPTION_LENGTH'] = str(config['captioning'].get('max_length', os.environ.get('MAX_CAPTION_LENGTH', '400')))
            
            if 'general' in config:
                os.environ['CONFIDENCE_THRESHOLD'] = str(config['general'].get('confidence_threshold', os.environ.get('CONFIDENCE_THRESHOLD', 0.7)))
                os.environ['USE_HALF_PRECISION'] = str(config['general'].get('use_half_precision', os.environ.get('USE_HALF_PRECISION', 'false'))).lower()
            
            # Model caching settings
            if 'model_cache' in config:
                os.environ['MODEL_CACHE_BUCKET'] = config['model_cache'].get('bucket', os.environ.get('MODEL_CACHE_BUCKET', ''))
                os.environ['MODEL_CACHE_PREFIX'] = config['model_cache'].get('prefix', os.environ.get('MODEL_CACHE_PREFIX', 'models'))
                
            # If FileLoader already initialized with different config, reset it
            if _initialized:
                _initialized = False
                _file_loader = None
                logger.info("Configuration changed, reinitializing FileLoader")
                _file_loader = model_fn(None) 
        
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
        
        document_markdown = document.document.export_to_markdown(image_placeholder='', image_mode=ImageRefMode.PLACEHOLDER) if hasattr(document, 'document') else str(document)
        
        return {
            'status': 'success',
            'document': document_markdown,
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
        try:
            # Use our custom JSON encoder to handle metadata serialization
            serialized = json.dumps(prediction, cls=FileMetadataJSONEncoder)
            logger.info(f"Successfully serialized response, length: {len(serialized)}")
            return serialized, content_types.JSON
        except Exception as e:
            logger.error(f"Error serializing prediction to JSON: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Return a simplified error response that we know can be serialized
            error_response = {
                'status': 'error',
                'error': f"Error serializing response: {str(e)}",
                'document': None,
                'metadata': None
            }
            return json.dumps(error_response), content_types.JSON
    else:
        logger.warning(f"Unsupported content type: {content_type}, using JSON instead")
        return json.dumps({
            'status': 'error',
            'error': f"Unsupported content type: {content_type}",
            'document': None, 
            'metadata': None
        }), content_types.JSON
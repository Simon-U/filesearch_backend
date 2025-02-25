import os
import json
import torch
import logging
from fileloader.loader import FileLoader
from fileloader.models import SharePointConfig, FileLocation, StorageType
from fileloader.image_processor.analyzer import AnalyzerConfig
from sagemaker_inference import content_types, decoder, encoder
from sagemaker_inference.default_handler_service import DefaultHandlerService
from sagemaker_inference.transformer import Transformer

# Set up basic logging to stdout
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

class DocumentProcessorHandler(Transformer):
    """
    Document processor handler that implements the SageMaker model server interface.
    Processes documents from SharePoint or local storage using FileLoader.
    """
    def __init__(self):
        super(DocumentProcessorHandler, self).__init__()
        self.initialized = False
        self.file_loader = None

    def initialize(self, context):
        """
        Initialize the FileLoader with configurations.
        This is called when the container starts.
        """
        if self.initialized:
            return

        logger.info("Initializing document processor handler")
        
        # Get SharePoint configuration from environment variables
        sharepoint_config = SharePointConfig(
            tenant_id=os.environ.get('SHAREPOINT_TENANT_ID'),
            client_id=os.environ.get('SHAREPOINT_CLIENT_ID'),
            client_secret=os.environ.get('SHAREPOINT_CLIENT_SECRET'),
            site_url=os.environ.get('SHAREPOINT_SITE_URL')
        )
        
        # Get analyzer configuration
        analyzer_config = self._get_analyzer_config()
        
        # Initialize FileLoader
        self.file_loader = FileLoader(
            config=sharepoint_config,
            num_threads=int(os.getenv('NUM_THREADS', '8')),
            use_cuda=torch.cuda.is_available(),
            do_table_structure=True,
            do_ocr=True,
            do_image_enrichment=True,
            image_analyzer_config=analyzer_config
        )
        
        self.initialized = True
        logger.info("Document processor handler initialized")

    def _get_analyzer_config(self):
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

    def transform(self, request_body, content_type, accept_type):
        """
        Transform a request using the document processor.
        This is the implementation of the main SageMaker transform functionality.
        
        Args:
            request_body: The request payload
            content_type: The request content type
            accept_type: The accept content type from the client
            
        Returns:
            Response data and response content type
        """
        # Initialize model if needed
        if not self.initialized:
            self.initialize(None)
            
        # Get the input data
        input_data = self._preprocess(request_body, content_type)
        
        # Process the document
        result = self._inference(input_data)
        
        # Return the output
        return self._postprocess(result, accept_type or content_types.JSON)

    def _preprocess(self, request_body, content_type):
        """
        Preprocess and decode the request body.
        Expects JSON with a 'file_location' key.
        
        Args:
            request_body: Raw request body
            content_type: Request content type
            
        Returns:
            FileLocation object for document processing
        """
        logger.info(f"Preprocessing with content type: {content_type}")
        
        if content_type != content_types.JSON:
            raise ValueError(f'Unsupported content type: {content_type}')
            
        try:
            # Decode JSON input
            input_data = json.loads(request_body)
            
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
                return file_location
            
            return input_data['file_location']
            
        except Exception as e:
            logger.error(f"Error preprocessing request: {e}")
            raise

    def _inference(self, file_location):
        """
        Process file using FileLoader.
        
        Args:
            file_location: FileLocation object specifying the document to process
            
        Returns:
            Dictionary with processing results or error information
        """
        logger.info(f"Processing document at: {file_location}")
        
        if self.file_loader is None:
            raise RuntimeError("FileLoader not initialized, please initialize first")
            
        try:
            document, metadata = self.file_loader.load(file_location)
            return {
                'status': 'success',
                'document': document,
                'metadata': metadata,
                'error': None
            }
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return {
                'status': 'error',
                'document': None,
                'metadata': None,
                'error': str(e)
            }

    def _postprocess(self, inference_output, accept):
        """
        Post-process and encode the inference output.
        
        Args:
            inference_output: Raw inference output
            accept: Client accept content type
            
        Returns:
            Post-processed model output
        """
        logger.info(f"Post-processing to {accept}")
        
        if accept != content_types.JSON:
            raise ValueError(f'Unsupported accept header: {accept}')
            
        return json.dumps(inference_output), accept

# Create the handler instance
_service = DocumentProcessorHandler()

def handler(data, context):
    """
    Call transform for the handler service.
    
    Args:
        data: Input data for transformation
        context: Context object with request details
        
    Returns:
        Transformed data
    """
    return _service.transform(data, context.request_content_type, context.accept_header)

# Create a handler service using our DocumentProcessorHandler
handler_service = DefaultHandlerService(_service)
import os
import json
import torch
import logging
from fileloader.loader import FileLoader
from fileloader.models import SharePointConfig, FileLocation, StorageType
from fileloader.image_processor.analyzer import AnalyzerConfig
from sagemaker_inference import default_inference_handler
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

class MyHandler(default_inference_handler.DefaultInferenceHandler):
    def get_analyzer_config(self):
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

    def model_fn(self, model_dir):
        """
        Initialize the FileLoader with configurations.
        This is called by SageMaker when starting the model server.
        """
        sharepoint_config = SharePointConfig(
            tenant_id=os.environ.get('SHAREPOINT_TENANT_ID'),
            client_id=os.environ.get('SHAREPOINT_CLIENT_ID'),
            client_secret=os.environ.get('SHAREPOINT_CLIENT_SECRET'),
            site_url=os.environ.get('SHAREPOINT_SITE_URL')
        )
        analyzer_config = self.get_analyzer_config()
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

    def input_fn(self, request_body, request_content_type, context=None):
        """
        Parse input data coming from SageMaker invocations.
        Expects JSON with a 'file_location' key.
        """
        if request_content_type != 'application/json':
            raise ValueError(f'Unsupported content type: {request_content_type}')
        input_data = json.loads(request_body)
        if 'file_location' not in input_data:
            raise ValueError("Input must contain 'file_location'")
        file_location_data = input_data['file_location']
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

    def predict_fn(self, file_location, model, context=None):
        """
        Process file using FileLoader.
        """
        try:
            document, metadata = model.load(file_location)
            return {
                'status': 'success',
                'document': document,
                'metadata': metadata,
                'error': None
            }
        except Exception as e:
            return {
                'status': 'error',
                'document': None,
                'metadata': None,
                'error': str(e)
            }

    def output_fn(self, prediction_output, accept, context=None):
        """
        Format the output to return to SageMaker client.
        """
        if accept == 'application/json':
            return json.dumps(prediction_output), 'application/json'
        raise ValueError(f'Unsupported accept header: {accept}')

# Export your handler instance




class HandlerService(DefaultHandlerService):
    def __init__(self):
        transformer = Transformer(default_inference_handler=MyHandler())
        super(HandlerService, self).__init__(transformer=transformer)

    def handle(self, data, context):
        """
        MMS calls this function for inference.
        `data`: The request data.
        `context`: Metadata about the request.
        """
        logger.info(data)
        logger.info(context)
        return self._service.transform(data, context)

# Export handler service for MMS
handler_service = HandlerService()
        
        
default_handler = HandlerService()
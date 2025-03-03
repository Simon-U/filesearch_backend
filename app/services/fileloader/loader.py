import os
import logging
import time
from typing import Optional, Any, Tuple, Union
from io import BytesIO
from docling.document_converter import DocumentConverter, DocumentStream
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
    PipelineOptions
)
from docling.datamodel.settings import settings
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption, WordFormatOption, PowerpointFormatOption

from .models import (
    FileLocation,
    FileMetadata,
    SharePointConfig,
    S3Config,
    StorageType
)
from .storage_connectors.base import StorageConnector
from .storage_connectors.local import LocalStorageConnector
from .storage_connectors.s3 import S3StorageConnector
from .storage_connectors.sharepoint import SharePointStorageConnector
from .exceptions import StorageNotConfiguredError, UnsupportedFileTypeError
from .image_processor import DocumentImageEnrichmentModel, AnalyzerConfig

from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
)
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.backend.mspowerpoint_backend import MsPowerpointDocumentBackend
from docling.backend.msword_backend import MsWordDocumentBackend

# Configure logging
logger = logging.getLogger("fileloader")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class FileLoader:
    def __init__(
        self,
        config: Optional[Union[SharePointConfig, S3Config]] = None,
        num_threads: int = 8,
        use_cuda: bool = True,
        do_table_structure: bool = True,
        do_ocr: bool = True,
        do_image_enrichment: bool = False,
        image_analyzer_config: Optional[AnalyzerConfig] = None
    ):
        """
        Initialize the FileLoader with storage config, docling settings, and enrichment options.
        
        Args:
            config: Configuration for storage (SharePointConfig or S3Config)
            num_threads: Number of threads for document processing
            use_cuda: Whether to use CUDA for acceleration
            do_table_structure: Whether to extract table structure
            do_ocr: Whether to perform OCR
            do_image_enrichment: Whether to enrich images with analysis
            image_analyzer_config: Configuration for image analyzer if enrichment enabled
        """
        logger.info(f"Initializing FileLoader with threads={num_threads}, use_cuda={use_cuda}")
        
        try:
            # Initialize docling settings
            self.accelerator_options = AcceleratorOptions(
                num_threads=num_threads,
                device=AcceleratorDevice.CUDA if use_cuda else AcceleratorDevice.CPU
            )

            # Initialize different pipeline options for different document types
            self.pdf_pipeline_options = PdfPipelineOptions(
                do_table_structure=do_table_structure,
                do_ocr=do_ocr,
                generate_picture_images=True,
                accelerator_options=self.accelerator_options
            )

            # Word and PowerPoint can use simpler pipeline options since they don't need OCR
            self.office_pipeline_options = PipelineOptions(
                accelerator_options=self.accelerator_options
            )
            
            settings.debug.profile_pipeline_timings = True

            # Initialize document converter
            logger.info("Initializing document converter")
            self._init_converter()
            
            # Initialize storage settings
            self.supported_extensions = {".pdf", ".docx", ".pptx"}
            self.file_type_mapping = {
                ".pdf": "PDF Document",
                ".docx": "Word Document",
                ".pptx": "PowerPoint Presentation"
            }
            
            # Initialize storage connectors based on config type
            logger.info("Setting up storage connectors")
            self._init_connectors(config)
            
            # Initialize image enrichment if enabled
            self.do_image_enrichment = do_image_enrichment
            if do_image_enrichment:
                logger.info("Initializing image enrichment model")
                self.enrichment_model = DocumentImageEnrichmentModel(
                    config=image_analyzer_config,
                    enabled=True
                )
            else:
                self.enrichment_model = None
                
            logger.info("FileLoader initialization complete")
        except Exception as e:
            logger.error(f"Error during FileLoader initialization: {str(e)}")
            raise

    def _init_converter(self):
        """Initialize the docling document converter with appropriate settings."""
        try:
            self.converter = DocumentConverter(
                        allowed_formats=[InputFormat.PDF, InputFormat.DOCX, InputFormat.PPTX],
                        format_options={
                            # PDF uses the standard PDF pipeline with full options
                            InputFormat.PDF: PdfFormatOption(
                                pipeline_cls=StandardPdfPipeline,
                                pipeline_options=self.pdf_pipeline_options,
                                backend=PyPdfiumDocumentBackend
                            ),
                            
                            # Word documents use the simple pipeline
                            InputFormat.DOCX: WordFormatOption(
                                pipeline_cls=SimplePipeline,
                                pipeline_options=self.office_pipeline_options,
                                backend=MsWordDocumentBackend
                            ),
                            
                            # PowerPoint presentations use the simple pipeline
                            InputFormat.PPTX: PowerpointFormatOption(
                                pipeline_cls=SimplePipeline,
                                pipeline_options=self.office_pipeline_options,
                                backend=MsPowerpointDocumentBackend
                            )
                        }
                    )
            logger.info("Document converter initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize document converter: {str(e)}")
            raise

    def _init_connectors(self, config: Optional[Union[SharePointConfig, S3Config]]):
        """Initialize storage connectors based on config type."""
        try:
            self.connectors = {
                StorageType.LOCAL: LocalStorageConnector(),
            }
            
            if config is None:
                logger.info("No remote storage configuration provided")
                return
                
            if isinstance(config, SharePointConfig):
                logger.info(f"Initializing SharePoint connector for site: {config.site_url}")
                self.connectors[StorageType.SHAREPOINT] = SharePointStorageConnector(config)
            elif isinstance(config, S3Config):
                logger.info(f"Initializing S3 connector for bucket: {config.bucket_name}")
                self.connectors[StorageType.S3] = S3StorageConnector(config)
            
            logger.info(f"Storage connectors initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize storage connectors: {str(e)}")
            raise

    def _get_file_extension(self, path: str) -> str:
        """Extract file extension from path."""
        return os.path.splitext(path)[1].lower()
    
    def _convert_file(self, stream: DocumentStream, file_extension: str) -> Any:
        """Convert the file using docling converter."""
        start_time = time.time()
        logger.info(f"Starting conversion of {file_extension} file")
        
        if file_extension not in self.supported_extensions:
            logger.error(f"File type '{file_extension}' is not supported")
            raise UnsupportedFileTypeError(f"File type '{file_extension}' is not supported")
        
        try:
            result = self.converter.convert(stream)
            elapsed_time = time.time() - start_time
            logger.info(f"File conversion completed in {elapsed_time:.2f} seconds")
            return result
        except Exception as e:
            logger.error(f"Error during file conversion: {str(e)}")
            raise
    
    def _enrich_document(self, document) -> Any:
        """Apply image enrichment if enabled."""
        if not self.do_image_enrichment:
            return document

        try:
            start_time = time.time()
            logger.info("Starting document image enrichment")
            
            # Process each picture element in the document
            enriched_elements = 0
            for element, _level in document.document.iterate_items():
                if self.enrichment_model.is_processable(document.document, element):
                    # Collect the processed elements from the generator
                    processed_elements = list(self.enrichment_model(document.document, [element]))
                    enriched_elements += len(processed_elements)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Document enrichment completed in {elapsed_time:.2f} seconds, enriched {enriched_elements} elements")
            return document
        except Exception as e:
            logger.error(f"Error during document enrichment: {str(e)}")
            logger.warning("Returning document without enrichment due to error")
            return document

    def load(self, location: FileLocation) -> Tuple[Any, FileMetadata]:
        """
        Load and convert a document from any configured storage location.
        
        Args:
            location: FileLocation object specifying where to load from
            
        Returns:
            Tuple of (converted_document, metadata)
        """
        overall_start_time = time.time()
        logger.info(f"Loading file from {location.storage_type.value}: {location.path}")
        
        try:
            if location.storage_type not in self.connectors:
                error_msg = (
                    f"Storage type {location.storage_type.value} is not configured. "
                    "Please provide appropriate configuration during FileLoader initialization."
                )
                logger.error(error_msg)
                raise StorageNotConfiguredError(error_msg)
            
            file_extension = self._get_file_extension(location.path)
            if file_extension not in self.supported_extensions:
                logger.error(f"File type '{file_extension}' is not supported")
                raise UnsupportedFileTypeError(f"File type '{file_extension}' is not supported")

            connector = self.connectors[location.storage_type]
            
            # Get file content and metadata
            fetch_start_time = time.time()
            logger.info(f"Fetching file content from {location.storage_type.value}")
            content = connector.get_file_content(location)
            metadata = connector.get_metadata(location)
            fetch_elapsed = time.time() - fetch_start_time
            logger.info(f"Retrieved {len(content)} bytes in {fetch_elapsed:.2f} seconds")
            
            # Convert bytes to BytesIO before creating DocumentStream
            content_stream = BytesIO(content)
            
            # Convert document
            logger.info(f"Starting conversion of {file_extension} file")
            document = self._convert_file(
                DocumentStream(name=location.path, stream=content_stream),
                file_extension
            )

            # Apply image enrichment if enabled
            if self.do_image_enrichment:
                logger.info("Starting document enrichment")
                document = self._enrich_document(document)

            overall_elapsed = time.time() - overall_start_time
            logger.info(f"Total processing time: {overall_elapsed:.2f} seconds")
            return document, metadata
            
        except Exception as e:
            logger.error(f"Error processing file {location.path}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
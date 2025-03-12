import json
import uuid
from os import environ
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any
from app.services.fileloader.storage_connectors.s3 import S3StorageConnector
from app.services.fileloader.models import FileLocation, S3Config, StorageType
from app.schemas.file_api import FileRequest, AddDocumentResponse, AddTopicRequest, AddTopicResponse
from app.services.database.vector_db import QuantVectorDB
from app.services.fileloader.document_chunker import DocumentChunker
from app.services.model_manager import get_bert_model

router = APIRouter(prefix="/document", tags=["document_management"])

@router.post("/add_document", response_model=AddDocumentResponse)
async def add_document(request: FileRequest = Body(...)):
    """
    Processes a document from S3 that contains document and metadata keys.
    
    **Parameters:**
    - `path` (str): The S3 path to the document file.
    - `tenant_id` (str): The ID of the tenant for data isolation.
    
    **Processing Steps:**
    1. Download the file from S3 using S3StorageConnector.
    2. Extract document content and metadata.
    3. Process the document content into chunks.
    4. Generate topic distributions for each chunk using BERTopic.
    5. Store chunks with their topic distributions in the vector database.
    
    **Responses:**
    - `200 OK`: Successfully processed and added document.
    - `400 Bad Request`: File loading or processing failed.
    - `500 Internal Server Error`: An unexpected server error occurred.
    
    **Returns:**
    ```json
    {
        "status": "success",
        "message": "Document processed successfully",
        "chunks_processed": 12
    }
    ```
    """
    try:
        # Configure and initialize S3 connector
        from app.config import settings
        s3_config = S3Config(
            bucket_name=settings.S3_BUCKET,
            region_name="us-west-2",
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            endpoint_url=None
        )
        s3_connector = S3StorageConnector(config=s3_config)
        chunker = DocumentChunker()
        
        bert = get_bert_model()

        
        print('loaded model', flush=True)
        
        # Initialize vector database with tenant ID from the request
        vector_db = QuantVectorDB(tenant_id=request.tenant_id)
        
        
        # Create file location object
        file_location = FileLocation(path=request.path, storage_type=StorageType.SHAREPOINT)
        
        # Load the file content as JSON
        content_json = s3_connector.get_file_content(file_location, is_json=True)

        # Extract document and metadata
        if 'document' not in content_json or 'metadata' not in content_json:
            raise HTTPException(status_code=400, detail="Invalid file format: missing 'document' or 'metadata' keys")
        
        document_content = content_json['document']
        file_metadata = content_json['metadata']

        # Get current timestamp in ISO format with timezone
        current_time = datetime.now(timezone.utc).isoformat()
        
        # First make chunks
        doc_chunks = chunker.chunk_doc(document_content)
        if not doc_chunks:
            raise HTTPException(status_code=400, detail="Document chunking failed. No content generated.")
        

        # Process and store each chunk
        chunks_processed = 0
        for i, chunk in enumerate(doc_chunks):
            # Get topic distribution for this chunk
            topic_distribution = bert.get_document_topic_distribution(chunk)

            # Convert topic distribution to a queryable format
            # Format: {"0": 0.996, "4": 0.072, ...}
            topic_probabilities = {}
            for topic_id, probability in topic_distribution:
                topic_probabilities[str(topic_id)] = probability
            
            # Create chunk metadata with timestamps
            # Using 'chunk_' prefix to avoid collisions with file metadata
            chunk_metadata = {
                "position": i,
                "topic_probabilities": topic_probabilities,
                "chunk_created_at": current_time,
                "chunk_updated_at": current_time
            }

            # Add file metadata
            chunk_metadata.update(file_metadata)
            
            # Insert chunk into vector database
            vector_db.insert_chunk(
                chunk_text=chunk,
                metadata=chunk_metadata
            )

            chunks_processed += 1
        
        return {
            "status": "success",
            "message": "Document processed successfully"
        }
        
    except HTTPException as http_exc:
        print(http_exc, flush=True)
        raise http_exc
    except json.JSONDecodeError as js:
        print(js, flush=True)
        raise HTTPException(status_code=400, detail="Invalid JSON format in the S3 file")
    except Exception as e:
        print(e, flush=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    
@router.post("/add_topic", response_model=AddTopicResponse)
async def add_topic(request: AddTopicRequest = Body(...)):
    """
    Adds a new topic to the system.
    
    **Parameters:**
    - `topic_id` (int): Unique ID for the topic.
    - `topic_label` (str): Label for the topic.
    - `topic_description` (str): Description of the topic.
    - `tenant_id` (str): The ID of the tenant for data isolation.
    
    **Processing Steps:**
    1. Create a vector embedding of the topic description.
    2. Store the topic in the topics collection.
    3. Associate it with timestamps for creation and updates.
    
    **Responses:**
    - `200 OK`: Successfully added the topic.
    - `400 Bad Request`: Invalid input or duplicate topic ID.
    - `500 Internal Server Error`: Unexpected errors during processing.
    
    **Returns:**
    ```json
    {
        "status": "success",
        "message": "Topic added successfully",
        "topic": {
            "topic_id": 1,
            "topic_label": "Example Topic",
            "topic_description": "This is an example topic"
        }
    }
    ```
    """
    try:
        # Initialize vector database with tenant ID
        vector_db = QuantVectorDB(tenant_id=request.tenant_id)
        
        # Add the topic
        result = vector_db.insert_topic(
                topic_id=request.topic_id,
                topic_label=request.topic_label,
                topic_description=request.topic_description
            )
            
        if not result:
            raise HTTPException(status_code=400, detail="Failed to add topic. It may already exist.")
        
        return {
            "status": "success",
            "message": "Topic added successfully",
        }
        
    except HTTPException as http_exc:
        print(http_exc)
        raise http_exc
    except Exception as e:
        print(e, flush=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
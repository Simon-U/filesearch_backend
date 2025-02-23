import json
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, Any, Optional

from services.fileloader.fileloader import FileLoader
from services.topicgenerator.topic_generator import TopicGenerator
from services.summarygenerator.summary_generator import Summarizer
from schemas.file_api import FileRequest, QueryRequest
from services.database.vector_db import QuantVectorDB

router = APIRouter(prefix="/document", tags=["documents"])

@router.post("/add_new_document", response_model=Dict[str, Any])
async def prepare_document(request: FileRequest):
    """
    Adds a new document to the vector database after processing.

    **Parameters:**
    - `path` (str): Path to the document.
    - `storage_type` (str): The storage type (local, S3, etc.).
    - `user_id` (str): The ID of the user uploading the document.
    - `tentant_id` (str): The ID of the tenant for data isolation.

    **Processing Steps:**
    1. Load the document from the given path.
    2. Split it into chunks for processing.
    3. Generate topics using `TopicGenerator`.
    4. Summarize the content using `Summarizer`.
    5. Insert extracted data (topics & summary) into the vector database.

    **Responses:**
    - `200 OK`: Successfully processed and added document.
    - `400 Bad Request`: File loading or chunking failed.
    - `404 Not Found`: File not found.
    - `501 Not Implemented`: A required method is not implemented.
    - `500 Internal Server Error`: An unexpected server error occurred.

    **Returns:**
    ```json
    {
        "status": "success",
        "message": "File added successfully",
    }
    ```
    """

    try:
        nr_topics = 5
        path = request.path
        storage_type = request.storage_type
        user_id = request.user_id
        tentant_id = request.tentant_id
        
        
        file_loader = FileLoader()
        summarizer = Summarizer()
        topic_generator = TopicGenerator(method='bertopic', nr_topics=nr_topics)
        vector_db = QuantVectorDB(tenant_id=tentant_id)

        
        doc, file_metadata = file_loader.load(path, storage_type)
        if not doc:
            raise HTTPException(status_code=400, detail="Failed to load document. File may be empty or inaccessible.")
        
        
        #ToDo We need to dynamically set chunk size and skip
        chunk_size = 500
        skip = 1
        
        doc_chunks = file_loader.chunk_doc(doc.document.export_to_markdown(), max_token=chunk_size, skip=skip)
        if not doc_chunks:
            raise HTTPException(status_code=400, detail="Document chunking failed. No content generated.")
        topics = topic_generator.generate(chunk_list=doc_chunks)
        print(f"topics are {topics}")
        summary, llm_count = summarizer.summarize(doc_chunks, skip=10)
        print(summary)
        
        # Worked until here, the issue is 
        #2 validation errors for PointStruct
        #id.int
        #Input should be a valid integer [type=int_type, input_value=None, input_type=NoneType]
        #    For further information visit https://errors.pydantic.dev/2.9/v/int_type
        #id.str
        #Input should be a valid string [type=string_type, input_value=None, input_type=NoneType]
        #    For further information visit https://errors.pydantic.dev/2.9/v/string_type
        
        vector_db.insert_document(topics, summary, file_metadata)
        return {
            "status": "success",
            "message": "File added successfully",
        }

    except HTTPException as http_exc:
        print(http_exc, flush=True)
        raise http_exc  # Let FastAPI handle known HTTP errors

    except FileNotFoundError as e:
        print(e, flush=True)
        raise HTTPException(status_code=404, detail="File not found. Please check the file path.")

    except NotImplementedError as nie:
        raise HTTPException(status_code=501, detail=str(nie))

    except Exception as e:
        print(e, flush=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred. We are looking into it.")
    
@router.get("/search_document", response_model=Dict[str, Any])
async def search_document(
    topic_term: str = Query(..., description="The topic to be searched"),
    summary_term: Optional[str] = Query(None, description="The summary to be searched"),
    tentant_id: int = Query(..., description="The ID of the tenant"),
    meta_data: Optional[str] = Query({}, description="Filtering conditions (JSON string)")
):
    """
    Search for documents in Qdrant using topic and summary vectors with metadata filters.

    **Parameters:**
    - `topic_term` (str): The topic to be searched.
    - `summary_term` (str): The summary to be searched.
    - `meta_data` (dict): A dictionary containing filtering conditions. See example below.
    - `user_id` (str): The ID of the user performing the search.
    - `tentant_id` (str): The ID of the tenant for scoping data access.

    **Example of `meta_data` Format:**
    ```json
    {
        "must": {
            "city": {"match": "London"},
            "price": {"range": {"gte": 100, "lte": 450}},
            "date": {"datetime_range": {"gte": "2023-02-08T10:49:00Z", "lte": "2024-01-31T10:14:31Z"}}
        },
        "should": {
            "color": {"match_any": ["red", "green"]},
            "category": {"full_text_match": "AI research"}
        },
        "must_not": {
            "status": {"match_except": ["inactive", "banned"]}
        },
        "nested": {
            "diet": {  
                "must": {  
                    "food": {"match": "meat"},
                    "likes": {"match": True}
                }
            }
        }
    }
    ```
    
    **Supported Conditions:**
    - `match`: Exact value match.
    - `match_any`: Matches any value from a list (equivalent to SQL `IN`).
    - `match_except`: Matches everything **except** values in the list (`NOT IN`).
    - `full_text_match`: Searches within a text field for a phrase.
    - `range`: Numerical range filtering (supports `gt`, `gte`, `lt`, `lte`).
    - `datetime_range`: Date range filtering (supports RFC 3339 datetime formats).
    - `nested`: Allows filtering inside nested objects.

    **Responses:**
    - `200 OK`: Successfully retrieved search results.
    - `400 Bad Request`: Invalid input or missing parameters.
    - `500 Internal Server Error`: Unexpected errors during processing.

    **Returns:**
    A dictionary containing merged results from both topic and summary searches.
    """
    try:
        vector_db = QuantVectorDB(tenant_id=tentant_id)
        results = vector_db.search_document(topic_term, summary_term, json.loads(meta_data))
        if "error" in results:
            raise HTTPException(status_code=500, detail=results["error"])

        return {"success": True, "data": results}

    except HTTPException as http_err:
        raise http_err

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
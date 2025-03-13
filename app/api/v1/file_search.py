import json
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, Any, Optional, List

from app.services.model_manager import get_bert_model
from app.services.database.vector_db import QuantVectorDB
from app.services.fileloader.document_chunker import DocumentChunker

router = APIRouter(prefix="/search", tags=["document_search"])

    
@router.get("/hybrid_search", response_model=Dict[str, Any])
async def hybrid_search(
    query: str = Query(..., description="The search query text"),
    tenant_id: str = Query(..., description="The ID of the tenant"),
    meta_data: Optional[str] = Query({}, description="Filtering conditions (JSON string)"),
    probability_limit: Optional[float] = Query(0.2, description="The limit under which to not consider topics. Default 0.2"),
):
    """
    Performs a hierarchical search:
    1. First identifies relevant topics based on query text using the topics endpoint
    2. Then searches document chunks within those topics using the chunks endpoint
    
    Returns results organized by topics with their associated chunks.

    **Parameters:**
    - `query` (str): The search query text.
    - `tenant_id` (int): The ID of the tenant for scoping data access.
    - `meta_data` (str, optional): Filtering conditions as a JSON string.
    - `probability_limit` (float, optional): The topic probability threshold. Default 0.2.

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
        }
    }
    ```

    **Responses:**
    - `200 OK`: Successfully retrieved search results organized by topics.
    - `400 Bad Request`: Invalid input or missing parameters.
    - `500 Internal Server Error`: Unexpected errors during processing.

    **Returns:**
    A hierarchical structure of topics and their associated document chunks.
    """
    try:

        # Parse metadata JSON if provided
        metadata = json.loads(meta_data) if meta_data else {}
        
        # Search topics
        topic_results = await search_topics(
            query=query, 
            tenant_id=tenant_id, 
            probability_limit=probability_limit
        )
        
        if not topic_results.get("success", False):
            raise HTTPException(status_code=500, detail="Failed to retrieve topics")
        
        topic_data = topic_results.get("data", [])

        # 2. Extract topic_ids from results
        topic_ids = [item['topic']['topic_id'] for item in topic_data]
        
        # 3. Search for chunks within these topics
        chunk_results = await search_chunks(
            query=query,
            topic_ids=topic_ids,
            tenant_id=tenant_id,
            meta_data=meta_data
        )
        
        
        if not chunk_results.get("success", False):
            raise HTTPException(status_code=500, detail="Failed to retrieve chunks")
        
        points = chunk_results['data']['results'].points
        results = []
        for point in points:
            results.append(
                {'score': point.score,
                 'url': point.payload['universal_url'],
                 'created_at': point.payload['created_at'],
                 'created_by': point.payload['created_by'],
                 'modified_at': point.payload['modified_at'],
                 'modified_by': point.payload['modified_by'],
                 'topics': point.payload['topic_probabilities'],
                 'text': point.payload['chunk_text'],
                 
                 }
            )
        
        return {"success": True, "data": {'topics': topic_data, "chunks": results}}

    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.get("/topics", response_model=Dict[str, Any])
async def search_topics(
    query: str = Query(None, description="The search query for topic descriptions"),
    tenant_id: str = Query(..., description="The ID of the tenant"),
    probability_limit: Optional[float] = Query(0.2, description="The limit under which to not consider topics anymore. Default 0.2"),
):
    """
    Searches for topics by ID/label or using semantic search on topic descriptions.

    **Parameters:**
    - `query` (str, optional): Search query for semantic search on topic descriptions.
    - `tenant_id` (int): The ID of the tenant for scoping data access.
    
    **Responses:**
    - `200 OK`: Successfully retrieved topic search results.
    - `400 Bad Request`: Invalid input or missing parameters.
    - `500 Internal Server Error`: Unexpected errors during processing.

    **Returns:**
    A list of matching topics with their metadata.
    """
    try:
        # Ensure at least one search parameter is provided

        vector_db = QuantVectorDB(tenant_id=tenant_id)
        
        
        bert = get_bert_model()
        topics = bert.get_document_topic_distribution(query)
        topic_prob_map = {topic_id: prob for topic_id, prob in topics}

        # Search topics
        results = vector_db.search_topics(
            topic_query=query,
            topic_ids=list(topic_prob_map.keys()),
            limit=3
        )
        
        
        for item in results['direct_results']:
            topic_id = item['topic']['topic_id']
            item['score'] = topic_prob_map[topic_id]
        
        seen_topic_ids = set()
        best_results = {}
        # Process both result lists
        for result_list in [results['semantic_results'], results['direct_results']]:
            for item in result_list:
                topic_id = item['topic']['topic_id']
                score = item['score']
                
                # If we haven't seen this topic yet, add it
                if topic_id not in seen_topic_ids:
                    seen_topic_ids.add(topic_id)
                    best_results[topic_id] = item
                # If we've seen it before, keep only the version with higher score
                elif score > best_results[topic_id]['score']:
                    best_results[topic_id] = item
        
        combined_results = list(best_results.values())

        # Sort the list by score in descending order
        combined_results.sort(key=lambda x: x['score'], reverse=True)
        combined_results = [item for item in combined_results if item['score'] > probability_limit]
        
        if "error" in results:
            raise HTTPException(status_code=500, detail=results["error"])

        return {"success": True, "data": combined_results}

    except HTTPException as http_err:
        raise http_err

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.get("/chunks", response_model=Dict[str, Any])
async def search_chunks(
    query: str = Query(..., description="The search query for chunk content"),
    topic_ids: Optional[List[int]] = Query(None, description="Optional list of topic IDs to filter chunks"),
    tenant_id: str = Query(..., description="The ID of the tenant"),
    meta_data: Optional[str] = Query({}, description="Filtering conditions (JSON string)")
):
    """
    Searches for document chunks matching the query, optionally filtered by topic IDs.

    **Parameters:**
    - `query` (str): The search query for chunk content.
    - `topic_ids` (List[int], optional): Optional list of topic IDs to filter chunks.
    - `tenant_id` (int): The ID of the tenant for scoping data access.
    - `meta_data` (str, optional): Filtering conditions as a JSON string.

    **Responses:**
    - `200 OK`: Successfully retrieved chunk search results.
    - `400 Bad Request`: Invalid input or missing parameters.
    - `500 Internal Server Error`: Unexpected errors during processing.

    **Returns:**
    A list of matching document chunks with their metadata.
    """
    try:
        vector_db = QuantVectorDB(tenant_id=tenant_id)
        
        # Parse metadata JSON if provided
        metadata = json.loads(meta_data) if meta_data else {}
        
        topic_ids = [str(topic) for topic in topic_ids]
        # Search chunks
        results = vector_db.search_chunks(
            query=query,
            topic_ids=topic_ids,
            metadata=metadata
        )
        if "error" in results:
            raise HTTPException(status_code=500, detail=results["error"])

        return {"success": True, "data": results}

    except HTTPException as http_err:
        raise http_err

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

class FileRequest(BaseModel):
    path: str = Field(..., description="S3 path to the document file")
    tenant_id: str = Field(..., description="Unique identifier for the tenant (company)")

class TopicInfo(BaseModel):
    topic_id: int
    topic_label: str
    topic_description: Optional[str] = None
    score: Optional[float] = None

class ChunkInfo(BaseModel):
    text: str
    document_id: str
    position: int
    score: float

class TopicWithChunks(BaseModel):
    topic_id: int
    topic_label: str
    topic_description: Optional[str] = None
    score: Optional[float] = None
    chunks: List[ChunkInfo]

class HybridSearchResult(BaseModel):
    topics: List[TopicWithChunks]
    total_topics: int
    total_chunks: int

class QueryRequest(BaseModel):
    query: str = Field(..., description="The search query text")
    topic_id: Optional[int] = Field(None, description="Optional topic ID for direct matching")
    topic_label: Optional[str] = Field(None, description="Optional topic label for direct matching")
    tenant_id: str = Field(..., description="Unique identifier for the tenant (company)")
    meta_data: Optional[Dict[str, Any]] = Field({}, description="Filter conditions")

class AddDocumentResponse(BaseModel):
    status: str
    message: str

class SearchResponse(BaseModel):
    success: bool
    data: HybridSearchResult
    
class AddTopicRequest(BaseModel):
    topic_id: int = Field(..., description="Unique ID for the topic")
    topic_label: str = Field(..., description="Label for the topic")
    topic_description: str = Field(..., description="Description of the topic")
    tenant_id: str = Field(..., description="The ID of the tenant for data isolation")

class AddTopicResponse(BaseModel):
    status: str
    message: str
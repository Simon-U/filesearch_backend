from pydantic import BaseModel
from typing import Dict, Any

class FileRequest(BaseModel):
    path: str
    storage_type: str  # Expected values: "local", "s3", "sharepoint"
    tentant_id: str  # Unique identifier for the tenant (company)
    user_id: str  # Unique identifier for the user


class QueryRequest(BaseModel):
    topic_term: str
    summary_term: str
    meta_data: dict
    tenant_id: str  # Unique identifier for the tenant (company)
    user_id: str  # Unique identifier for the user
    


class AddDocumentResponse(BaseModel):
    status: str
    message: str
    file_metadata: Dict[str, Any]
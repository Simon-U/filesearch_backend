# app/routers/admin.py
"""
Admin endpoints for model management and application monitoring.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any
from app.services.model_manager import get_bert_model, get_model_status

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/model/status", response_model=Dict[str, Any])
async def get_model_status_endpoint():
    """
    Get current status of the BERT model.
    
    Returns:
        Dict: Status information about the model
    """
    try:
        status = get_model_status()
        return {"success": True, "data": status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving model status: {str(e)}")

@router.post("/model/reload", response_model=Dict[str, Any])
async def reload_model_endpoint(
    background_tasks: BackgroundTasks,
    wait_for_completion: bool = False
):
    """
    Reload the BERT model, either in the background or synchronously.
    
    Args:
        background_tasks: FastAPI background tasks
        wait_for_completion: If True, wait for model to reload before responding
        
    Returns:
        Dict: Success status and message
    """
    try:
        if wait_for_completion:
            # Synchronous reload - will block until complete
            get_bert_model(force_reload=True)
            return {
                "success": True, 
                "data": {
                    "message": "Model reloaded successfully", 
                    "status": get_model_status()
                }
            }
        else:
            # Asynchronous reload in background
            background_tasks.add_task(get_bert_model, force_reload=True)
            return {
                "success": True, 
                "data": {
                    "message": "Model reload initiated in background", 
                    "status": get_model_status()
                }
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reloading model: {str(e)}")
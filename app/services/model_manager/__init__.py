# app/services/model_manager.py
"""
Centralized model management for the application.
This module provides global access to ML models and utilities to load/reload them.
"""

import os
import time
from functools import lru_cache
from app.config import settings
from app.services.topicgenerator.bert_topic_generator import BertTopicGenerator

# Global model instance
_BERT_MODEL = None
_LAST_LOADED = None

def get_bert_model(force_reload=False):
    """
    Singleton factory function for BERTopic model.
    Returns the same model instance across all calls.
    
    Args:
        force_reload (bool): If True, forces reloading the model even if already loaded
        
    Returns:
        BertTopicGenerator: The loaded BERT model instance
    """
    global _BERT_MODEL, _LAST_LOADED
    
    if _BERT_MODEL is None or force_reload:
        print(f"{'Re' if force_reload else ''}Loading BERT model...")
        start_time = time.time()
        
        _BERT_MODEL = BertTopicGenerator(
            verbose=False, 
            calculate_probabilities=True, 
            min_topic_size=10, 
            min_cluster_size=3
        )
        
        # Load model from S3
        _BERT_MODEL.load_model_from_s3(
            s3_path=settings.BERT_MODEL,
            bucket_name=settings.S3_BUCKET,
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            endpoint_url=None
        )
        
        _LAST_LOADED = time.time()
        load_time = _LAST_LOADED - start_time
        print(f"BERT model {'re' if force_reload else ''}loaded in {load_time:.2f} seconds")
    
    return _BERT_MODEL

def get_model_status():
    """
    Get the current status of the model
    
    Returns:
        dict: Status information about the model
    """
    return {
        "is_loaded": _BERT_MODEL is not None,
        "last_loaded": _LAST_LOADED,
        "model_config": {
            "s3_path": settings.BERT_MODEL,
            "bucket": settings.S3_BUCKET,
            "region": settings.AWS_REGION
        } if _BERT_MODEL else None
    }

def preload_bert_model():
    """
    Preload the BERT model.
    Useful for application startup to eliminate first-request latency.
    """
    get_bert_model()
    print("BERT model preloaded successfully")
from typing import Optional, Any
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

import os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(base_dir, '.env')

class Settings(BaseSettings):
    # API Configuration
    ENV: str = "dev"
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "MyAPI"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "My FastAPI application"
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = False
    WORKERS: int = 1
    
    # Security
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ALGORITHM: str = "HS256"
    
    BACKEND_CORS_ORIGINS: list[str] = ["*"]
    
    OPENAI_API_KEY: str
    ANTHROPIC_API_KEY: str
    
    # AWS Configuration
    aws_access_key_id: Any
    aws_secret_access_key: Any
    AWS_REGION: Any = "us-west-2"
    S3_BUCKET: Any
    
    # Database Configuration
    DB_HOST: str
    DB_PORT: str
    DB_USER: str
    DB_PASS: str
    DB_NAME: str
    
    # Add the missing fields from the error message
    #HTTP_PROXY: Optional[str] = None
    LOCAL_LLM_URL: Optional[str] = None
    SHAREPOINT_TENANT_ID: Optional[str] = None
    SHAREPOINT_CLIENT_ID: Optional[str] = None
    SHAREPOINT_CLIENT_SECRET: Optional[str] = None
    HF_TOKEN: Optional[str] = None
    SAGEMAKER_ARN: Optional[str] = None
    SAGEMAKER_IMAGE_URI: Optional[str] = None
    SHAREPOINT_SITE_URL: Optional[str] = None
    BERT_MODEL: Optional[str] = None
    SEARCH_ENDPOINT: Optional[str] = None

    @property
    def DATABASE_URL(self) -> str:
        """Construct database URL from components."""
        return f"postgresql://{self.DB_USER}:{self.DB_PASS}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
    
    model_config = SettingsConfigDict(
        env_file=env_path,
        env_file_encoding="utf-8",
        case_sensitive=True,
        # You could also add this instead of defining all fields:
        # extra="allow"
    )

@lru_cache()
def get_settings() -> Settings:
    """
    Create cached instance of settings.
    This function returns the same instance of settings throughout the application's lifecycle.
    """
    return Settings()

settings = get_settings()

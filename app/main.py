from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from app.api.v1.router import api_router
from app.config import settings
from app.services.model_manager import preload_bert_model

def create_application() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=settings.DESCRIPTION,
        openapi_url=f"{settings.API_V1_STR}/openapi.json"
    )
    
    # Set up CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.BACKEND_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API router
    app.include_router(
        api_router,
        prefix=settings.API_V1_STR
    )
    
    # Add startup event for model preloading
    @app.on_event("startup")
    async def startup_event():
        """
        Application startup event.
        Preloads models in the background to reduce latency for first requests.
        """
        print("Starting application...")
        
        # Create a background task to preload the model without blocking startup
        asyncio.create_task(preload_model_on_startup())
        
        print("Application startup completed, models loading in background")
    
    async def preload_model_on_startup():
        """
        Asynchronous task to preload models on startup.
        """
        try:
            print("Starting BERT model preloading...")
            preload_bert_model()
            print("BERT model preloaded and ready!")
        except Exception as e:
            print(f"Error during model preloading: {str(e)}")
            # Log the error but don't crash the application
    
    return app
    
app = create_application()

if __name__ == "__main__":
    import uvicorn
    import onnxruntime as ort
    print(ort.get_device())
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        workers=settings.WORKERS,
        reload_dirs=["/home/simon/Documents/Pure_Inference/Projects/filesearch/backend/app"]
    )
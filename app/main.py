from fastapi import FastAPI
import asyncio
import socketio
import logging
from app.api.v1.router import api_router
from app.config import settings
from app.services.model_manager import preload_bert_model
from app.sockets.user_requests import register_socket_events

# Configure logging

# Define allowed origins for Socket.IO
allowed_origins = [
    "http://localhost:5001",
    "http://127.0.0.1:5001",
    # Add any other origins your frontend might use
    "http://localhost:3000",
    "http://127.0.0.1:3000", 
    "http://localhost:8080",
    "http://127.0.0.1:8080"
]

# Create Socket.IO server with explicit allowed origins
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins=allowed_origins,
    ping_timeout=60,  # Increase from default
    ping_interval=25
)

# Create Socket.IO ASGI app
socket_app = socketio.ASGIApp(sio)

def create_application() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=settings.DESCRIPTION,
        openapi_url=f"{settings.API_V1_STR}/openapi.json"
    )
    
    # Include API router
    app.include_router(
        api_router,
        prefix=settings.API_V1_STR
    )
    
    # Register Socket.IO events
    register_socket_events(sio)
    
    # Mount Socket.IO app at root level
    app.mount("/", socket_app)
    
    # Add connection/disconnection debug handlers
    @sio.event
    async def connect(sid, environ, namespace):
        client_origin = environ.get('HTTP_ORIGIN', 'Unknown')
        return True
        
    @sio.event
    async def disconnect(sid):
        print(f"Client disconnected: {sid}")
    
    # Add startup event for model preloading
    @app.on_event("startup")
    async def startup_event():
        """
        Application startup event.
        Preloads models in the background to reduce latency for first requests.
        """
        
        # Create a background task to preload the model without blocking startup
        asyncio.create_task(preload_model_on_startup())
        
    
    async def preload_model_on_startup():
        """
        Asynchronous task to preload models on startup.
        """
        try:
            preload_bert_model()
        except Exception as e:
            print(f"Error during model preloading: {str(e)}")
            # Log the error but don't crash the application
    
    return app
    
app = create_application()

if __name__ == "__main__":
    import uvicorn
    print("Starting uvicorn server...")
    uvicorn.run(
        "app.main:app",  # Make sure this matches your file path structure
        host=settings.HOST, 
        port=settings.PORT,
        reload=settings.RELOAD,
        workers=1,  # Use only 1 worker for Socket.IO testing
    )
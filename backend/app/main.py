import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from .core.config import settings
from .api.endpoints import router as api_router
from .db.database import engine, Base
from .services.scheduler import telemetry_scheduler

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize DB tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set to actual domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Attach API endpoints
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.on_event("startup")
def startup_event():
    logger.info("Starting up FastAPI Backend Service...")
    # Seed default baseline if empty, and load models
    from backend.app.ml.pipeline import pipeline_manager
    logger.info(f"ML Pipeline models status loaded: {pipeline_manager.load_models()}")
    
    # Proactively start background telemetry streaming on startup
    # This guarantees that the dashboard has telemetry to present instantly on load
    telemetry_scheduler.start_stream()

@app.on_event("shutdown")
def shutdown_event():
    logger.info("Shutting down FastAPI Backend Service...")
    telemetry_scheduler.stop_stream()

@app.get("/")
def read_root():
    return {
        "service": settings.PROJECT_NAME,
        "api_v1_docs": "/docs",
        "health": "UP"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)

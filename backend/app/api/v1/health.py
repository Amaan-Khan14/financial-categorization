"""
Health check endpoint
"""
from fastapi import APIRouter, Depends
from app.core.model_loader import ModelLoader
import time

router = APIRouter()

start_time = time.time()


def get_model_loader():
    """Dependency to get model loader"""
    from app.main import model_loader
    return model_loader


@router.get("/health")
async def health_check(model_loader: ModelLoader = Depends(get_model_loader)):
    """Health check endpoint"""
    uptime = time.time() - start_time

    return {
        "status": "healthy",
        "model_loaded": model_loader.is_loaded if model_loader else False,
        "version": "1.0.0",
        "uptime_seconds": int(uptime)
    }

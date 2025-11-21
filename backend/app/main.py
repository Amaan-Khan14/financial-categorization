"""
Main FastAPI Application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from app.api.v1 import predict, batch, feedback, taxonomy, metrics, health, insights
from app.core.model_loader import ModelLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Transaction Categorization API",
    description="AI-powered transaction categorization system with Indian market support",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for model loader
model_loader = None


@app.on_event("startup")
async def startup_event():
    """Load ML models on startup"""
    global model_loader
    logger.info("Starting up application...")
    try:
        model_loader = ModelLoader()
        model_loader.load_models()
        logger.info("✓ Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down application...")


# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(predict.router, prefix="/api/v1/predict", tags=["Prediction"])
app.include_router(batch.router, prefix="/api/v1/batch", tags=["Batch Processing"])
app.include_router(feedback.router, prefix="/api/v1/feedback", tags=["Feedback"])
app.include_router(taxonomy.router, prefix="/api/v1/taxonomy", tags=["Taxonomy"])
app.include_router(metrics.router, prefix="/api/v1/metrics", tags=["Metrics"])
app.include_router(insights.router, prefix="/api/v1/insights", tags=["Insights"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Transaction Categorization API",
        "version": "1.0.0",
        "docs": "/api/docs",
        "health": "/api/v1/health"
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc)
        }
    )

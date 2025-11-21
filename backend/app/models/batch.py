"""
Batch processing-related Pydantic models
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class BatchTransaction(BaseModel):
    """Single transaction in batch"""
    transaction_id: str
    transaction: str
    predicted_category: str
    confidence: float


class BatchSummary(BaseModel):
    """Summary statistics for batch processing"""
    processed: int
    high_confidence: int = Field(..., description="Predictions with confidence > 85%")
    medium_confidence: int = Field(..., description="Predictions with confidence 70-85%")
    low_confidence: int = Field(..., description="Predictions with confidence < 70%")


class BatchResponse(BaseModel):
    """Response model for batch processing"""
    job_id: str
    total_transactions: int
    status: str
    results: List[BatchTransaction]
    summary: BatchSummary
    download_url: Optional[str] = None

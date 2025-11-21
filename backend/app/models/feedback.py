"""
Feedback-related Pydantic models
"""
from pydantic import BaseModel, Field
from typing import Optional


class FeedbackInput(BaseModel):
    """Input model for feedback submission"""
    transaction: str = Field(..., min_length=1, description="Original transaction")
    predicted_category: str = Field(..., description="Category predicted by model")
    correct_category: str = Field(..., description="Correct category provided by user")
    confidence: float = Field(..., ge=0, le=1, description="Original prediction confidence")
    user_id: Optional[str] = Field(default=None, description="Optional user ID")

    class Config:
        json_schema_extra = {
            "example": {
                "transaction": "Nykaa beauty products",
                "predicted_category": "Food & Dining",
                "correct_category": "Shopping",
                "confidence": 0.183,
                "user_id": "user_123"
            }
        }


class FeedbackResponse(BaseModel):
    """Response model for feedback submission"""
    status: str
    message: str
    feedback_id: str

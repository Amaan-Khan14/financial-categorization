"""
Transaction-related Pydantic models
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime


class TransactionInput(BaseModel):
    """Input model for single transaction prediction"""
    transaction: str = Field(..., min_length=1, max_length=500, description="Transaction description")
    metadata: Optional[Dict] = Field(default=None, description="Additional metadata (amount, date, etc.)")

    class Config:
        json_schema_extra = {
            "example": {
                "transaction": "BigBasket grocery delivery",
                "metadata": {
                    "amount": 2500.00,
                    "date": "2025-11-13"
                }
            }
        }


class Alternative(BaseModel):
    """Alternative prediction with confidence"""
    category: str
    confidence: float = Field(..., ge=0, le=1)


class FeatureExplanation(BaseModel):
    """Feature contribution to prediction"""
    feature: str
    weight: float
    impact: Optional[str] = None  # e.g., "strong_positive", "moderate_negative", etc.


class ModelVote(BaseModel):
    """Individual model prediction"""
    category: str
    confidence: float


class DetailedExplanation(BaseModel):
    """Detailed explanation for prediction with LIME and feature importance"""
    top_features: List[FeatureExplanation]
    model_votes: Optional[Dict[str, ModelVote]] = None
    confidence_flag: Optional[str] = None  # HIGH, MEDIUM, LOW
    requires_review: Optional[bool] = False


class Explanation(BaseModel):
    """Explanation for prediction (legacy - simplified)"""
    top_features: List[FeatureExplanation]


class PredictionMetadata(BaseModel):
    """Metadata about prediction"""
    processing_time_ms: float
    model_version: str = "1.0.0"
    timestamp: str


class PredictionResponse(BaseModel):
    """Response model for single transaction prediction"""
    transaction: str
    predicted_category: str
    confidence: float = Field(..., ge=0, le=1)
    alternatives: List[Alternative]
    explanation: Optional[DetailedExplanation] = None
    metadata: PredictionMetadata

    class Config:
        json_schema_extra = {
            "example": {
                "transaction": "BigBasket grocery delivery",
                "predicted_category": "Groceries",
                "confidence": 0.9234,
                "alternatives": [
                    {"category": "Shopping", "confidence": 0.0523},
                    {"category": "Food & Dining", "confidence": 0.0143}
                ],
                "explanation": {
                    "top_features": [
                        {"feature": "bigbasket", "weight": 0.82},
                        {"feature": "grocery", "weight": 0.65}
                    ]
                },
                "metadata": {
                    "processing_time_ms": 45.2,
                    "model_version": "1.0.0",
                    "timestamp": "2025-11-13T12:00:00Z"
                }
            }
        }

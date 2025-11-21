"""
Insights-related Pydantic models
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime


class TransactionWithDetails(BaseModel):
    """Transaction with full details for insights analysis"""
    transaction_id: str
    transaction: str
    category: str
    amount: float = Field(..., gt=0, description="Transaction amount in INR")
    date: str = Field(..., description="Transaction date in YYYY-MM-DD format")
    confidence: Optional[float] = Field(default=None, ge=0, le=1)

    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "txn_0001",
                "transaction": "Swiggy food delivery",
                "category": "Food & Dining",
                "amount": 450.00,
                "date": "2025-11-01",
                "confidence": 0.95
            }
        }


class AnomalyTransaction(BaseModel):
    """Anomalous transaction with explanation"""
    transaction_id: str
    transaction: str
    category: str
    amount: float
    date: str
    anomaly_score: float = Field(..., description="Anomaly score (-1 = anomaly, 1 = normal)")
    reason: str = Field(..., description="Why this is flagged as anomalous")
    severity: str = Field(..., description="HIGH, MEDIUM, or LOW")


class SpendingPattern(BaseModel):
    """Spending pattern analysis"""
    category: str
    total_amount: float
    transaction_count: int
    average_amount: float
    percentage_of_total: float


class DayOfWeekPattern(BaseModel):
    """Spending pattern by day of week"""
    day: str
    average_spending: float
    transaction_count: int


class CategoryForecast(BaseModel):
    """Forecast for a specific category"""
    category: str
    current_month_spending: float
    forecasted_next_month: float
    confidence: str = Field(..., description="HIGH, MEDIUM, or LOW")
    trend: str = Field(..., description="INCREASING, STABLE, or DECREASING")


class Recommendation(BaseModel):
    """Personalized recommendation"""
    type: str = Field(..., description="savings, optimization, alert, or info")
    category: str
    title: str
    message: str
    potential_savings: Optional[float] = None
    priority: str = Field(..., description="HIGH, MEDIUM, or LOW")
    actionable: bool = Field(default=True, description="Whether user can act on this")


class InsightsResponse(BaseModel):
    """Complete insights analysis response"""
    summary: Dict[str, Any]
    anomalies: List[AnomalyTransaction]
    spending_by_category: List[SpendingPattern]
    spending_by_day: List[DayOfWeekPattern]
    forecasts: List[CategoryForecast]
    recommendations: List[Recommendation]
    analysis_period: Dict[str, str]
    total_transactions: int
    total_spending: float


class GenerateDataRequest(BaseModel):
    """Request to generate sample transaction history"""
    num_transactions: int = Field(default=100, ge=10, le=1000, description="Number of transactions to generate")
    days_back: int = Field(default=90, ge=30, le=365, description="How many days of history to generate")
    seed: Optional[int] = Field(default=42, description="Random seed for reproducibility")

    class Config:
        json_schema_extra = {
            "example": {
                "num_transactions": 150,
                "days_back": 90,
                "seed": 42
            }
        }


class GenerateDataResponse(BaseModel):
    """Response from data generation"""
    message: str
    transactions_generated: int
    date_range: Dict[str, str]
    categories: List[str]
    total_amount: float

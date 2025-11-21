"""
Insights API endpoints

Provides AI-powered financial insights:
- Anomaly detection
- Spending pattern analysis
- Forecasting
- Personalized recommendations
"""
from fastapi import APIRouter, HTTPException
from app.models.insights import (
    TransactionWithDetails,
    InsightsResponse,
    GenerateDataRequest,
    GenerateDataResponse
)
from app.services.insights_engine import InsightsEngine
from app.services.sample_data_generator import SampleDataGenerator
from typing import List
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory storage for demo (in production, use a database)
_demo_transactions: List[dict] = []


@router.post("/generate-demo-data", response_model=GenerateDataResponse)
async def generate_demo_data(request: GenerateDataRequest):
    """
    Generate sample transaction history for demo/testing purposes

    This creates realistic transaction data with dates and amounts
    that can be used to demonstrate insights capabilities.
    """
    try:
        global _demo_transactions

        logger.info(f"Generating demo data: {request.num_transactions} transactions, {request.days_back} days back")

        # Generate data
        generator = SampleDataGenerator(seed=request.seed)
        transactions = generator.generate_transaction_history(
            num_transactions=request.num_transactions,
            days_back=request.days_back
        )

        # Store in memory
        _demo_transactions = transactions

        # Get summary
        stats = generator.get_summary_stats(transactions)

        logger.info(f"Generated {len(transactions)} transactions, total amount: ₹{stats['total_amount']}")

        return GenerateDataResponse(
            message=f"Successfully generated {len(transactions)} transactions",
            transactions_generated=len(transactions),
            date_range=stats['date_range'],
            categories=stats['categories'],
            total_amount=stats['total_amount']
        )

    except Exception as e:
        logger.error(f"Error generating demo data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate demo data: {str(e)}")


@router.get("/demo-transactions")
async def get_demo_transactions():
    """
    Get all demo transactions

    Returns the generated transaction history
    """
    global _demo_transactions

    if not _demo_transactions:
        raise HTTPException(
            status_code=404,
            detail="No demo data available. Please generate demo data first using /insights/generate-demo-data"
        )

    return {
        "total": len(_demo_transactions),
        "transactions": _demo_transactions
    }


@router.post("/analyze", response_model=InsightsResponse)
async def analyze_transactions(transactions: List[TransactionWithDetails]):
    """
    Analyze a list of transactions and generate insights

    Provides:
    - Summary statistics
    - Anomaly detection
    - Spending patterns (by category and day)
    - Next-month forecasts
    - Personalized recommendations

    **Note:** If you don't have transaction data with dates/amounts,
    use the `/insights/generate-demo-data` endpoint first.
    """
    try:
        if not transactions or len(transactions) == 0:
            raise HTTPException(status_code=400, detail="No transactions provided")

        logger.info(f"Analyzing {len(transactions)} transactions...")

        # Convert Pydantic models to dicts
        transaction_dicts = [tx.model_dump() for tx in transactions]

        # Initialize insights engine
        engine = InsightsEngine(transaction_dicts)

        # Generate full insights
        insights = engine.get_full_insights()

        logger.info(f"Analysis complete: {len(insights['anomalies'])} anomalies, "
                   f"{len(insights['recommendations'])} recommendations")

        return InsightsResponse(**insights)

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing transactions: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/analyze-demo", response_model=InsightsResponse)
async def analyze_demo_data():
    """
    Analyze the demo data that was previously generated

    This is a convenience endpoint that analyzes the in-memory demo transactions.
    First generate demo data using `/insights/generate-demo-data`, then call this endpoint.
    """
    global _demo_transactions

    if not _demo_transactions:
        raise HTTPException(
            status_code=404,
            detail="No demo data available. Please generate demo data first using /insights/generate-demo-data"
        )

    try:
        logger.info(f"Analyzing {len(_demo_transactions)} demo transactions...")

        # Initialize insights engine
        engine = InsightsEngine(_demo_transactions)

        # Generate full insights
        insights = engine.get_full_insights()

        logger.info(f"Demo analysis complete: {len(insights['anomalies'])} anomalies, "
                   f"{len(insights['recommendations'])} recommendations")

        return InsightsResponse(**insights)

    except Exception as e:
        logger.error(f"Error analyzing demo data: {e}")
        raise HTTPException(status_code=500, detail=f"Demo analysis failed: {str(e)}")

"""
Batch processing endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from app.models.batch import BatchResponse, BatchTransaction, BatchSummary
from app.core.model_loader import ModelLoader
import pandas as pd
import io
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


def get_model_loader():
    """Dependency to get model loader"""
    from app.main import model_loader
    return model_loader


@router.post("/upload", response_model=BatchResponse)
async def process_batch(
    file: UploadFile = File(...),
    model_loader: ModelLoader = Depends(get_model_loader)
):
    """
    Process a batch of transactions from CSV file

    Expected CSV format:
    transaction_id,transaction
    1,BigBasket grocery delivery
    2,Myntra fashion shopping
    """
    try:
        # Validate file type
        if not file.filename.endswith(('.csv', '.xlsx')):
            raise HTTPException(status_code=400, detail="Only CSV or Excel files are supported")

        # Read file
        contents = await file.read()

        # Parse CSV/Excel
        try:
            if file.filename.endswith('.csv'):
                df = pd.read_csv(io.BytesIO(contents))
            else:
                df = pd.read_excel(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")

        # Validate columns
        if 'transaction' not in df.columns:
            raise HTTPException(
                status_code=400,
                detail="CSV must contain 'transaction' column"
            )

        # Add transaction_id if not present
        if 'transaction_id' not in df.columns:
            df['transaction_id'] = [f"txn_{i+1:04d}" for i in range(len(df))]

        # Process transactions
        results = []
        high_conf = 0
        medium_conf = 0
        low_conf = 0
        category_distribution = {}

        logger.info(f"Processing {len(df)} transactions...")

        for _, row in df.iterrows():
            try:
                transaction = str(row['transaction'])
                transaction_id = str(row['transaction_id'])

                # Predict
                prediction = model_loader.predict(transaction)

                # Categorize confidence
                confidence = prediction['confidence']
                if confidence > 0.85:
                    high_conf += 1
                elif confidence > 0.70:
                    medium_conf += 1
                else:
                    low_conf += 1

                # Track category distribution
                predicted_category = prediction['predicted_category']
                category_distribution[predicted_category] = category_distribution.get(predicted_category, 0) + 1

                results.append(BatchTransaction(
                    transaction_id=transaction_id,
                    transaction=transaction,
                    predicted_category=predicted_category,
                    confidence=confidence
                ))

            except Exception as e:
                logger.error(f"Error processing transaction {row.get('transaction_id', 'unknown')}: {e}")
                # Continue processing other transactions

        logger.info(f"Batch processing complete: {len(results)} transactions processed")

        # Create response
        return BatchResponse(
            total_transactions=len(results),
            status="completed",
            results=results,
            summary=BatchSummary(
                processed=len(results),
                high_confidence=high_conf,
                medium_confidence=medium_conf,
                low_confidence=low_conf,
                category_distribution=category_distribution
            )
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

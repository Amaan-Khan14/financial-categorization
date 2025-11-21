"""
Batch processing endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from app.models.batch import BatchResponse, BatchTransaction, BatchSummary
from app.core.model_loader import ModelLoader
import pandas as pd
import io
import uuid
from pathlib import Path
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# Directory for batch results
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
BATCH_RESULTS_DIR = PROJECT_ROOT / "backend" / "batch_results"
BATCH_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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

                results.append(BatchTransaction(
                    transaction_id=transaction_id,
                    transaction=transaction,
                    predicted_category=prediction['predicted_category'],
                    confidence=confidence
                ))

            except Exception as e:
                logger.error(f"Error processing transaction {row.get('transaction_id', 'unknown')}: {e}")
                # Continue processing other transactions

        # Generate job ID
        job_id = f"batch_{uuid.uuid4().hex[:12]}"

        # Save results to CSV
        results_df = pd.DataFrame([
            {
                'transaction_id': r.transaction_id,
                'transaction': r.transaction,
                'predicted_category': r.predicted_category,
                'confidence': r.confidence
            }
            for r in results
        ])

        output_path = BATCH_RESULTS_DIR / f"{job_id}.csv"
        results_df.to_csv(output_path, index=False)

        logger.info(f"Batch processing complete: {len(results)} transactions processed")

        # Create response
        return BatchResponse(
            job_id=job_id,
            total_transactions=len(results),
            status="completed",
            results=results,
            summary=BatchSummary(
                processed=len(results),
                high_confidence=high_conf,
                medium_confidence=medium_conf,
                low_confidence=low_conf
            ),
            download_url=f"/api/v1/batch/{job_id}/download"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")


@router.get("/{job_id}/download")
async def download_batch_results(job_id: str):
    """Download batch processing results as CSV"""
    file_path = BATCH_RESULTS_DIR / f"{job_id}.csv"

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Batch results not found")

    return FileResponse(
        path=file_path,
        filename=f"categorized_{job_id}.csv",
        media_type="text/csv"
    )

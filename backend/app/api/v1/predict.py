"""
Prediction endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from app.models.transaction import TransactionInput, PredictionResponse, Alternative, DetailedExplanation, FeatureExplanation, ModelVote, PredictionMetadata
from app.core.model_loader import ModelLoader
import time
from datetime import datetime
import logging
import io
from PIL import Image
import easyocr

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize OCR reader (lazily loaded on first use)
_ocr_reader = None

def get_ocr_reader():
    """Get or initialize OCR reader"""
    global _ocr_reader
    if _ocr_reader is None:
        logger.info("Initializing EasyOCR reader...")
        _ocr_reader = easyocr.Reader(['en'], gpu=False)
    return _ocr_reader


def get_model_loader():
    """Dependency to get model loader"""
    from app.main import model_loader
    return model_loader


@router.post("/single", response_model=PredictionResponse)
async def predict_single(
    transaction_input: TransactionInput,
    model_loader: ModelLoader = Depends(get_model_loader)
):
    """
    Categorize a single transaction

    Returns prediction with confidence score, alternatives, and explanation
    """
    try:
        start_time = time.time()

        # Get prediction
        result = model_loader.predict(transaction_input.transaction)

        # Get detailed explanation
        try:
            explanation_data = model_loader.explain_prediction(transaction_input.transaction)

            # Format LIME top features with impact
            top_features = []
            if 'explanation' in explanation_data and 'top_features' in explanation_data['explanation']:
                for feat in explanation_data['explanation']['top_features']:
                    top_features.append(FeatureExplanation(
                        feature=feat['feature'],
                        weight=float(feat['weight']),
                        impact=feat.get('impact')
                    ))

            # Format model votes
            model_votes = {}
            if 'explanation' in explanation_data and 'model_votes' in explanation_data['explanation']:
                for model_name, vote_data in explanation_data['explanation']['model_votes'].items():
                    model_votes[model_name] = ModelVote(
                        category=vote_data['category'],
                        confidence=float(vote_data['confidence'])
                    )

            explanation = DetailedExplanation(
                top_features=top_features,
                model_votes=model_votes if model_votes else None,
                confidence_flag=explanation_data.get('confidence_flag'),
                requires_review=explanation_data.get('requires_review', False)
            ) if top_features else None
        except Exception as e:
            logger.warning(f"Could not generate explanation: {e}")
            explanation = None

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        # Format alternatives
        alternatives = [
            Alternative(category=alt['category'], confidence=alt['confidence'])
            for alt in result.get('alternatives', [])
        ]

        # Create response
        response = PredictionResponse(
            transaction=transaction_input.transaction,
            predicted_category=result['predicted_category'],
            confidence=result['confidence'],
            alternatives=alternatives,
            explanation=explanation,
            metadata=PredictionMetadata(
                processing_time_ms=processing_time,
                model_version="1.0.0",
                timestamp=datetime.utcnow().isoformat() + "Z"
            )
        )

        logger.info(f"Predicted '{transaction_input.transaction}' as '{result['predicted_category']}' "
                   f"with {result['confidence']:.2%} confidence in {processing_time:.1f}ms")

        return response

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/image", response_model=PredictionResponse)
async def predict_from_image(
    file: UploadFile = File(...),
    model_loader: ModelLoader = Depends(get_model_loader)
):
    """
    Categorize a transaction from an image (receipt, screenshot, etc.)

    Process:
    1. Extract text from image using OCR
    2. Categorize extracted text
    3. Return prediction with explanation
    """
    try:
        start_time = time.time()

        # Validate file type
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg", "image/webp"]:
            raise HTTPException(
                status_code=400,
                detail="Only image files (JPEG, PNG, WebP) are supported"
            )

        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Extract text using OCR
        logger.info(f"Extracting text from image: {file.filename}")
        ocr_reader = get_ocr_reader()

        # Convert image to numpy array for OCR
        import numpy as np
        image_array = np.array(image)

        # Run OCR
        ocr_results = ocr_reader.readtext(image_array)

        # Extract and combine text
        extracted_text = " ".join([text[1] for text in ocr_results])

        if not extracted_text.strip():
            raise HTTPException(
                status_code=400,
                detail="Could not extract any text from the image. Please upload a clearer image."
            )

        logger.info(f"Extracted text: {extracted_text[:100]}...")

        # Get prediction on extracted text
        result = model_loader.predict(extracted_text)

        # Get detailed explanation
        try:
            explanation_data = model_loader.explain_prediction(extracted_text)

            # Format LIME top features with impact
            top_features = []
            if 'explanation' in explanation_data and 'top_features' in explanation_data['explanation']:
                for feat in explanation_data['explanation']['top_features']:
                    top_features.append(FeatureExplanation(
                        feature=feat['feature'],
                        weight=float(feat['weight']),
                        impact=feat.get('impact')
                    ))

            # Format model votes
            model_votes = {}
            if 'explanation' in explanation_data and 'model_votes' in explanation_data['explanation']:
                for model_name, vote_data in explanation_data['explanation']['model_votes'].items():
                    model_votes[model_name] = ModelVote(
                        category=vote_data['category'],
                        confidence=float(vote_data['confidence'])
                    )

            explanation = DetailedExplanation(
                top_features=top_features,
                model_votes=model_votes if model_votes else None,
                confidence_flag=explanation_data.get('confidence_flag'),
                requires_review=explanation_data.get('requires_review', False)
            ) if top_features else None
        except Exception as e:
            logger.warning(f"Could not generate explanation: {e}")
            explanation = None

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        # Format alternatives
        alternatives = [
            Alternative(category=alt['category'], confidence=alt['confidence'])
            for alt in result.get('alternatives', [])
        ]

        # Create response - use extracted text in response
        response = PredictionResponse(
            transaction=extracted_text,
            predicted_category=result['predicted_category'],
            confidence=result['confidence'],
            alternatives=alternatives,
            explanation=explanation,
            metadata=PredictionMetadata(
                processing_time_ms=processing_time,
                model_version="1.0.0",
                timestamp=datetime.utcnow().isoformat() + "Z"
            )
        )

        logger.info(f"Predicted image text as '{result['predicted_category']}' "
                   f"with {result['confidence']:.2%} confidence in {processing_time:.1f}ms")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

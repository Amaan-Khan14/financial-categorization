"""
Prediction endpoints
"""
from fastapi import APIRouter, Depends, HTTPException
from app.models.transaction import TransactionInput, PredictionResponse, Alternative, DetailedExplanation, FeatureExplanation, ModelVote, PredictionMetadata
from app.core.model_loader import ModelLoader
import time
from datetime import datetime
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


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

"""
Feedback endpoints
"""
from fastapi import APIRouter, HTTPException
from app.models.feedback import FeedbackInput, FeedbackResponse
import json
import uuid
from pathlib import Path
from datetime import datetime
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# Directory for feedback storage
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
FEEDBACK_DIR = PROJECT_ROOT / "backend" / "feedback"
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_FILE = FEEDBACK_DIR / "feedback_log.jsonl"


@router.post("/", response_model=FeedbackResponse)
async def submit_feedback(feedback: FeedbackInput):
    """
    Submit feedback for a prediction

    Users can correct incorrect predictions. This feedback is logged
    for model retraining and improvement.
    """
    try:
        # Generate feedback ID
        feedback_id = f"fb_{uuid.uuid4().hex[:8]}"

        # Create feedback record
        feedback_record = {
            "feedback_id": feedback_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "transaction": feedback.transaction,
            "predicted_category": feedback.predicted_category,
            "correct_category": feedback.correct_category,
            "confidence": feedback.confidence,
            "user_id": feedback.user_id
        }

        # Append to feedback log (JSONL format)
        with open(FEEDBACK_FILE, 'a') as f:
            f.write(json.dumps(feedback_record) + '\n')

        logger.info(f"Feedback recorded: {feedback_id} - {feedback.transaction} "
                   f"({feedback.predicted_category} → {feedback.correct_category})")

        return FeedbackResponse(
            status="success",
            message="Feedback recorded successfully",
            feedback_id=feedback_id
        )

    except Exception as e:
        logger.error(f"Feedback submission error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {str(e)}")


@router.get("/stats")
async def get_feedback_stats():
    """Get feedback statistics"""
    try:
        if not FEEDBACK_FILE.exists():
            return {
                "total_feedback": 0,
                "corrections_by_category": {}
            }

        # Read all feedback
        feedback_records = []
        with open(FEEDBACK_FILE, 'r') as f:
            for line in f:
                feedback_records.append(json.loads(line))

        # Calculate stats
        corrections_by_category = {}
        for record in feedback_records:
            pred_cat = record['predicted_category']
            correct_cat = record['correct_category']
            key = f"{pred_cat} → {correct_cat}"
            corrections_by_category[key] = corrections_by_category.get(key, 0) + 1

        return {
            "total_feedback": len(feedback_records),
            "corrections_by_category": corrections_by_category,
            "recent_feedback": feedback_records[-10:]  # Last 10 feedback items
        }

    except Exception as e:
        logger.error(f"Failed to get feedback stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve feedback stats: {str(e)}")

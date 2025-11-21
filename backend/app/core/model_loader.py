"""
Model Loader - Integrates with existing ML models
"""
import sys
import os
from pathlib import Path
import logging

# Add project root to path to import existing modules
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import TransactionClassifier
from src.preprocessing import TransactionPreprocessor
from src.taxonomy_loader import TaxonomyLoader
from src.explainer import TransactionExplainer
import joblib

logger = logging.getLogger(__name__)


class ModelLoader:
    """Loads and manages ML models"""

    def __init__(self):
        self.classifier = None
        self.preprocessor = None
        self.taxonomy_loader = None
        self.explainer = None
        self.is_loaded = False

        # Paths
        self.models_dir = PROJECT_ROOT / "models"
        self.config_dir = PROJECT_ROOT / "config"

    def load_models(self):
        """Load all required models and components"""
        try:
            logger.info("Loading models...")

            # Load taxonomy
            logger.info("Loading taxonomy...")
            self.taxonomy_loader = TaxonomyLoader(str(self.config_dir / "taxonomy.yaml"))

            # Load preprocessor
            logger.info("Loading preprocessor...")
            preprocessor_path = self.models_dir / "preprocessor.pkl"
            if not preprocessor_path.exists():
                raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")

            preprocessor_data = joblib.load(preprocessor_path)

            # Reconstruct the TransactionPreprocessor from saved dict
            self.preprocessor = TransactionPreprocessor(
                max_features=preprocessor_data['max_features'],
                ngram_range=preprocessor_data['ngram_range'],
                seed=preprocessor_data['seed']
            )
            self.preprocessor.tfidf_vectorizer = preprocessor_data['tfidf_vectorizer']
            self.preprocessor.char_vectorizer = preprocessor_data['char_vectorizer']
            self.preprocessor.is_fitted = preprocessor_data['is_fitted']

            # Load classifier
            logger.info("Loading classifier...")
            self.classifier = TransactionClassifier.load(str(self.models_dir))

            # Initialize explainer
            logger.info("Initializing explainer...")
            categories = self.taxonomy_loader.get_categories()
            self.explainer = TransactionExplainer(
                self.classifier,
                self.preprocessor,
                categories
            )

            self.is_loaded = True
            logger.info("✓ All models loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

    def predict(self, transaction: str):
        """
        Make prediction for a single transaction

        Returns:
            dict with prediction, confidence, alternatives
        """
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        # Preprocess
        features = self.preprocessor.transform([transaction])

        # Get prediction with confidence
        prediction_result = self.classifier.predict_with_confidence(features, top_k=3)[0]

        return prediction_result

    def explain_prediction(self, transaction: str):
        """
        Get explanation for a prediction

        Returns:
            dict with prediction and feature importance
        """
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        explanation = self.explainer.explain_prediction(transaction)
        return explanation

    def get_categories(self):
        """Get list of all categories"""
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        return self.taxonomy_loader.get_categories()

    def get_feature_importance(self):
        """Get feature importance from model"""
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        return self.classifier.get_feature_importance()

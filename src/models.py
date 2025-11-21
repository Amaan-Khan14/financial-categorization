"""
Ensemble Classification Models
Multi-stage classification with Logistic Regression, SVM, and Random Forest
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Tuple, Optional
import joblib
from pathlib import Path


class TransactionClassifier:
    """Ensemble classifier for transaction categorization"""

    def __init__(self, seed=42):
        """
        Initialize ensemble classifier

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.label_encoder = LabelEncoder()

        # Individual models
        self.logistic_model = LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver='lbfgs',
            multi_class='multinomial',
            random_state=seed,
            n_jobs=-1
        )

        self.svm_model = SVC(
            kernel='linear',  # Linear kernel for text classification
            C=1.0,
            probability=True,
            random_state=seed
        )

        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1
        )

        # Voting ensemble with weighted voting
        # Weights based on validation performance: LR (best) > RF > SVM
        self.ensemble = VotingClassifier(
            estimators=[
                ('logistic', self.logistic_model),
                ('svm', self.svm_model),
                ('random_forest', self.rf_model)
            ],
            voting='soft',  # Use probability averaging
            weights=[0.5, 0.2, 0.3],  # LR: 50%, SVM: 20%, RF: 30%
            n_jobs=-1
        )

        self.is_fitted = False
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TransactionClassifier':
        """
        Fit the ensemble classifier

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)

        Returns:
            Self
        """
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_

        # Fit ensemble
        self.ensemble.fit(X, y_encoded)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict categories

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predicted categories
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction")

        y_encoded = self.ensemble.predict(X)
        return self.label_encoder.inverse_transform(y_encoded)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Probability matrix (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction")

        return self.ensemble.predict_proba(X)

    def predict_with_confidence(self, X: np.ndarray, top_k: int = 3) -> List[Dict]:
        """
        Predict with confidence scores and alternatives

        Args:
            X: Feature matrix (n_samples, n_features)
            top_k: Number of top predictions to return

        Returns:
            List of prediction dictionaries
        """
        probas = self.predict_proba(X)
        predictions = []

        for prob_dist in probas:
            # Sort by probability
            top_indices = np.argsort(prob_dist)[::-1][:top_k]
            top_probs = prob_dist[top_indices]
            top_classes = self.label_encoder.inverse_transform(top_indices)

            prediction = {
                'predicted_category': top_classes[0],
                'confidence': float(top_probs[0]),
                'alternatives': [
                    {'category': cat, 'confidence': float(prob)}
                    for cat, prob in zip(top_classes[1:], top_probs[1:])
                ]
            }

            predictions.append(prediction)

        return predictions

    def get_individual_predictions(self, X: np.ndarray) -> Dict[str, Dict]:
        """
        Get predictions from each individual model

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Dictionary with predictions from each model
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction")

        results = {}

        # Logistic Regression
        lr_proba = self.logistic_model.predict_proba(X)
        lr_pred = self.logistic_model.predict(X)
        results['Logistic Regression'] = {
            'predictions': self.label_encoder.inverse_transform(lr_pred),
            'probabilities': lr_proba,
            'confidence': np.max(lr_proba, axis=1)
        }

        # SVM
        svm_proba = self.svm_model.predict_proba(X)
        svm_pred = self.svm_model.predict(X)
        results['SVM'] = {
            'predictions': self.label_encoder.inverse_transform(svm_pred),
            'probabilities': svm_proba,
            'confidence': np.max(svm_proba, axis=1)
        }

        # Random Forest
        rf_proba = self.rf_model.predict_proba(X)
        rf_pred = self.rf_model.predict(X)
        results['Random Forest'] = {
            'predictions': self.label_encoder.inverse_transform(rf_pred),
            'probabilities': rf_proba,
            'confidence': np.max(rf_proba, axis=1)
        }

        return results

    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance from Random Forest

        Returns:
            Feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before getting feature importance")

        return self.rf_model.feature_importances_

    def save(self, model_dir: str):
        """
        Save all models to disk

        Args:
            model_dir: Directory to save models
        """
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save individual models
        joblib.dump(self.logistic_model, model_dir / 'logistic_regression.pkl')
        joblib.dump(self.svm_model, model_dir / 'svm.pkl')
        joblib.dump(self.rf_model, model_dir / 'random_forest.pkl')

        # Save ensemble
        joblib.dump(self.ensemble, model_dir / 'voting_ensemble.pkl')

        # Save label encoder and metadata
        joblib.dump({
            'label_encoder': self.label_encoder,
            'classes': self.classes_,
            'seed': self.seed,
            'is_fitted': self.is_fitted
        }, model_dir / 'metadata.pkl')

    @classmethod
    def load(cls, model_dir: str) -> 'TransactionClassifier':
        """
        Load models from disk

        Args:
            model_dir: Directory containing saved models

        Returns:
            Loaded classifier
        """
        model_dir = Path(model_dir)

        # Load metadata
        metadata = joblib.load(model_dir / 'metadata.pkl')

        # Create classifier
        classifier = cls(seed=metadata['seed'])
        classifier.label_encoder = metadata['label_encoder']
        classifier.classes_ = metadata['classes']
        classifier.is_fitted = metadata['is_fitted']

        # Load ensemble first
        classifier.ensemble = joblib.load(model_dir / 'voting_ensemble.pkl')

        # Extract fitted individual models from the ensemble
        # The VotingClassifier stores fitted estimators in estimators_ attribute
        classifier.logistic_model = classifier.ensemble.estimators_[0]  # logistic
        classifier.svm_model = classifier.ensemble.estimators_[1]  # svm
        classifier.rf_model = classifier.ensemble.estimators_[2]  # random_forest

        return classifier


class RuleBasedFallback:
    """Rule-based fallback for low-confidence predictions"""

    def __init__(self, taxonomy_loader):
        """
        Initialize rule-based fallback

        Args:
            taxonomy_loader: TaxonomyLoader instance
        """
        self.taxonomy_loader = taxonomy_loader
        self.keyword_map = taxonomy_loader.keyword_to_category_map()

    def predict(self, transaction: str, confidence_threshold: float = 0.60) -> Optional[str]:
        """
        Apply rule-based prediction

        Args:
            transaction: Transaction string
            confidence_threshold: Minimum confidence required

        Returns:
            Category or None if no match
        """
        transaction_lower = transaction.lower()

        # Exact keyword matching
        for keyword, category in self.keyword_map.items():
            if keyword in transaction_lower:
                if isinstance(category, list):
                    # Multiple categories contain this keyword, return first
                    return category[0]
                return category

        return None

    def fuzzy_match(self, transaction: str, max_distance: int = 2) -> Optional[str]:
        """
        Apply fuzzy matching using Levenshtein distance

        Args:
            transaction: Transaction string
            max_distance: Maximum edit distance

        Returns:
            Category or None
        """
        from difflib import SequenceMatcher

        transaction_lower = transaction.lower()
        best_match = None
        best_ratio = 0.0

        for keyword, category in self.keyword_map.items():
            ratio = SequenceMatcher(None, transaction_lower, keyword).ratio()
            if ratio > best_ratio and ratio > 0.8:  # 80% similarity threshold
                best_ratio = ratio
                if isinstance(category, list):
                    best_match = category[0]
                else:
                    best_match = category

        return best_match


if __name__ == '__main__':
    # Quick test
    print("Testing TransactionClassifier...")

    # Create dummy data
    np.random.seed(42)
    X_train = np.random.rand(100, 50)
    y_train = np.random.choice(['Food', 'Fuel', 'Shopping'], size=100)

    # Train classifier
    classifier = TransactionClassifier(seed=42)
    classifier.fit(X_train, y_train)

    # Test prediction
    X_test = np.random.rand(5, 50)
    predictions = classifier.predict_with_confidence(X_test, top_k=3)

    for i, pred in enumerate(predictions):
        print(f"\nSample {i + 1}:")
        print(f"  Category: {pred['predicted_category']}")
        print(f"  Confidence: {pred['confidence']:.2f}")
        print(f"  Alternatives: {pred['alternatives']}")

    print("\nClassifier test completed!")

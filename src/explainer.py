"""
Explainability Module
Provides feature attribution and interpretable explanations using LIME
"""

import numpy as np
from lime.lime_text import LimeTextExplainer
from typing import Dict, List, Tuple, Optional
import json


class TransactionExplainer:
    """Provide interpretable explanations for transaction predictions"""

    def __init__(self, classifier, preprocessor, class_names: List[str]):
        """
        Initialize explainer

        Args:
            classifier: Trained TransactionClassifier
            preprocessor: TransactionPreprocessor instance
            class_names: List of category names
        """
        self.classifier = classifier
        self.preprocessor = preprocessor
        self.class_names = class_names

        # Initialize LIME explainer
        self.lime_explainer = LimeTextExplainer(
            class_names=class_names,
            bow=True,
            random_state=42
        )

    def explain_prediction(self, transaction: str, num_features: int = 10) -> Dict:
        """
        Generate comprehensive explanation for a single transaction

        Args:
            transaction: Raw transaction string
            num_features: Number of top features to include

        Returns:
            Dictionary with detailed explanation
        """
        # Get prediction
        features = self.preprocessor.transform([transaction])
        predictions = self.classifier.predict_with_confidence(features, top_k=3)[0]

        # Get individual model votes
        individual_preds = self.classifier.get_individual_predictions(features)

        # Get LIME explanation
        lime_explanation = self._get_lime_explanation(transaction, num_features)

        # Get feature importance from Random Forest
        feature_importance = self._get_top_features(features, num_features)

        # Determine confidence flag
        confidence = predictions['confidence']
        if confidence >= 0.85:
            confidence_flag = "HIGH"
            requires_review = False
        elif confidence >= 0.70:
            confidence_flag = "MEDIUM"
            requires_review = False
        else:
            confidence_flag = "LOW"
            requires_review = True

        # Build explanation
        explanation = {
            'transaction': transaction,
            'predicted_category': predictions['predicted_category'],
            'confidence': predictions['confidence'],

            'explanation': {
                'top_features': lime_explanation,
                'feature_importance': feature_importance,
                'model_votes': self._format_model_votes(individual_preds)
            },

            'alternatives': predictions['alternatives'],
            'confidence_flag': confidence_flag,
            'requires_review': requires_review,

            'metadata': {
                'normalized_text': self.preprocessor.normalize_text(transaction),
                'text_features': self.preprocessor.extract_features(
                    self.preprocessor.normalize_text(transaction)
                )
            }
        }

        return explanation

    def _get_lime_explanation(self, transaction: str, num_features: int) -> List[Dict]:
        """
        Get LIME-based feature attribution

        Args:
            transaction: Transaction string
            num_features: Number of features

        Returns:
            List of feature attributions
        """
        def predict_fn(texts):
            """Prediction function for LIME"""
            features = self.preprocessor.transform(texts)
            return self.classifier.predict_proba(features)

        try:
            # Generate LIME explanation (with limited samples for speed)
            exp = self.lime_explainer.explain_instance(
                transaction,
                predict_fn,
                num_features=num_features,
                num_samples=100,  # Reduced from default 5000 for faster predictions
                top_labels=1
            )

            # Get predicted class
            predicted_class_idx = self.classifier.predict(
                self.preprocessor.transform([transaction])
            )
            predicted_class = np.where(
                self.classifier.classes_ == predicted_class_idx[0]
            )[0][0]

            # Extract feature weights
            feature_weights = exp.as_list(label=predicted_class)

            # Format as list of dicts
            features = []
            for feature, weight in feature_weights:
                impact = "strong_positive" if weight > 0.3 else \
                        "moderate_positive" if weight > 0.1 else \
                        "weak_positive" if weight > 0 else \
                        "strong_negative" if weight < -0.3 else \
                        "moderate_negative" if weight < -0.1 else \
                        "weak_negative"

                features.append({
                    'feature': feature,
                    'weight': round(weight, 4),
                    'impact': impact
                })

            return features

        except Exception as e:
            # Fallback if LIME fails
            return [{
                'feature': 'explanation_unavailable',
                'weight': 0.0,
                'impact': 'neutral',
                'error': str(e)
            }]

    def _get_top_features(self, features: np.ndarray, num_features: int) -> List[Dict]:
        """
        Get top features based on Random Forest feature importance

        Args:
            features: Feature vector
            num_features: Number of top features

        Returns:
            List of feature importance scores
        """
        try:
            # Get feature importance
            importance_scores = self.classifier.get_feature_importance()

            # Get top features
            top_indices = np.argsort(importance_scores)[::-1][:num_features]
            top_scores = importance_scores[top_indices]

            # Format output
            top_features = []
            for idx, score in zip(top_indices, top_scores):
                top_features.append({
                    'feature_index': int(idx),
                    'importance': round(float(score), 4)
                })

            return top_features

        except Exception as e:
            return [{
                'feature_index': -1,
                'importance': 0.0,
                'error': str(e)
            }]

    def _format_model_votes(self, individual_preds: Dict) -> Dict:
        """
        Format individual model predictions

        Args:
            individual_preds: Dictionary with predictions from each model

        Returns:
            Formatted model votes
        """
        formatted = {}

        for model_name, preds in individual_preds.items():
            formatted[model_name] = {
                'category': preds['predictions'][0],
                'confidence': round(float(preds['confidence'][0]), 4)
            }

        return formatted

    def explain_batch(self, transactions: List[str], num_features: int = 10) -> List[Dict]:
        """
        Generate explanations for multiple transactions

        Args:
            transactions: List of transaction strings
            num_features: Number of top features

        Returns:
            List of explanations
        """
        return [
            self.explain_prediction(transaction, num_features)
            for transaction in transactions
        ]

    def _convert_to_json_serializable(self, obj):
        """Convert numpy types to JSON-serializable Python types"""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        else:
            return obj

    def export_explanation(self, explanation: Dict, output_path: str, format: str = 'json'):
        """
        Export explanation to file

        Args:
            explanation: Explanation dictionary
            output_path: Path to save file
            format: Output format ('json' or 'txt')
        """
        if format == 'json':
            # Convert to JSON-serializable format
            explanation = self._convert_to_json_serializable(explanation)
            with open(output_path, 'w') as f:
                json.dump(explanation, f, indent=2)

        elif format == 'txt':
            with open(output_path, 'w') as f:
                f.write(f"Transaction: {explanation['transaction']}\n")
                f.write(f"Predicted Category: {explanation['predicted_category']}\n")
                f.write(f"Confidence: {explanation['confidence']:.2%}\n")
                f.write(f"Confidence Flag: {explanation['confidence_flag']}\n")
                f.write(f"Requires Review: {explanation['requires_review']}\n\n")

                f.write("Top Features:\n")
                for feat in explanation['explanation']['top_features']:
                    f.write(f"  - {feat['feature']}: {feat['weight']:.4f} ({feat['impact']})\n")

                f.write("\nModel Votes:\n")
                for model, vote in explanation['explanation']['model_votes'].items():
                    f.write(f"  - {model}: {vote['category']} ({vote['confidence']:.2%})\n")

                f.write("\nAlternatives:\n")
                for alt in explanation['alternatives']:
                    f.write(f"  - {alt['category']}: {alt['confidence']:.2%}\n")

    def generate_summary_report(self, explanations: List[Dict]) -> Dict:
        """
        Generate summary report from multiple explanations

        Args:
            explanations: List of explanation dictionaries

        Returns:
            Summary statistics
        """
        total = len(explanations)

        # Confidence distribution
        high_conf = sum(1 for e in explanations if e['confidence_flag'] == 'HIGH')
        medium_conf = sum(1 for e in explanations if e['confidence_flag'] == 'MEDIUM')
        low_conf = sum(1 for e in explanations if e['confidence_flag'] == 'LOW')

        # Review required
        requires_review = sum(1 for e in explanations if e['requires_review'])

        # Average confidence by category
        category_confidences = {}
        for exp in explanations:
            cat = exp['predicted_category']
            if cat not in category_confidences:
                category_confidences[cat] = []
            category_confidences[cat].append(exp['confidence'])

        avg_confidence_by_category = {
            cat: np.mean(confs)
            for cat, confs in category_confidences.items()
        }

        summary = {
            'total_predictions': total,
            'confidence_distribution': {
                'HIGH': high_conf,
                'MEDIUM': medium_conf,
                'LOW': low_conf
            },
            'confidence_distribution_pct': {
                'HIGH': round(high_conf / total * 100, 2),
                'MEDIUM': round(medium_conf / total * 100, 2),
                'LOW': round(low_conf / total * 100, 2)
            },
            'requires_review': requires_review,
            'requires_review_pct': round(requires_review / total * 100, 2),
            'avg_confidence_by_category': {
                cat: round(conf, 4)
                for cat, conf in avg_confidence_by_category.items()
            },
            'overall_avg_confidence': round(
                np.mean([e['confidence'] for e in explanations]), 4
            )
        }

        return summary


if __name__ == '__main__':
    print("Explainability module loaded successfully!")
    print("Use TransactionExplainer for generating predictions with explanations.")

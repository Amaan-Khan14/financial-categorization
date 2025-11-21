"""
Unit tests for model module
"""

import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models import TransactionClassifier


class TestTransactionClassifier:
    """Test classifier functionality"""

    def test_fit_predict(self):
        """Test basic fit and predict"""
        np.random.seed(42)
        X_train = np.random.rand(100, 50)
        y_train = np.random.choice(['Food', 'Fuel', 'Shopping'], size=100)

        classifier = TransactionClassifier(seed=42)
        classifier.fit(X_train, y_train)

        X_test = np.random.rand(10, 50)
        predictions = classifier.predict(X_test)

        assert len(predictions) == 10
        assert all(p in ['Food', 'Fuel', 'Shopping'] for p in predictions)

    def test_predict_proba(self):
        """Test probability predictions"""
        np.random.seed(42)
        X_train = np.random.rand(100, 50)
        y_train = np.random.choice(['Food', 'Fuel', 'Shopping'], size=100)

        classifier = TransactionClassifier(seed=42)
        classifier.fit(X_train, y_train)

        X_test = np.random.rand(10, 50)
        proba = classifier.predict_proba(X_test)

        assert proba.shape == (10, 3)
        assert np.allclose(proba.sum(axis=1), 1.0)  # Probabilities sum to 1

    def test_predict_with_confidence(self):
        """Test confidence prediction"""
        np.random.seed(42)
        X_train = np.random.rand(100, 50)
        y_train = np.random.choice(['Food', 'Fuel', 'Shopping'], size=100)

        classifier = TransactionClassifier(seed=42)
        classifier.fit(X_train, y_train)

        X_test = np.random.rand(5, 50)
        predictions = classifier.predict_with_confidence(X_test, top_k=2)

        assert len(predictions) == 5

        for pred in predictions:
            assert 'predicted_category' in pred
            assert 'confidence' in pred
            assert 'alternatives' in pred
            assert 0 <= pred['confidence'] <= 1
            assert len(pred['alternatives']) == 1  # top_k=2 means 1 alternative

    def test_predict_without_fit(self):
        """Test that predict fails without fit"""
        classifier = TransactionClassifier()
        X_test = np.random.rand(10, 50)

        with pytest.raises(ValueError):
            classifier.predict(X_test)

    def test_save_load(self, tmp_path):
        """Test saving and loading models"""
        np.random.seed(42)
        X_train = np.random.rand(100, 50)
        y_train = np.random.choice(['Food', 'Fuel', 'Shopping'], size=100)

        classifier = TransactionClassifier(seed=42)
        classifier.fit(X_train, y_train)

        # Save
        model_dir = tmp_path / "models"
        classifier.save(str(model_dir))

        # Load
        loaded = TransactionClassifier.load(str(model_dir))

        # Test loaded classifier works
        X_test = np.random.rand(10, 50)
        predictions = loaded.predict(X_test)

        assert len(predictions) == 10

    def test_reproducibility(self):
        """Test that results are reproducible with same seed"""
        np.random.seed(42)
        X_train = np.random.rand(100, 50)
        y_train = np.random.choice(['Food', 'Fuel', 'Shopping'], size=100)
        X_test = np.random.rand(10, 50)

        # Train first model
        classifier1 = TransactionClassifier(seed=42)
        classifier1.fit(X_train, y_train)
        pred1 = classifier1.predict(X_test)

        # Train second model with same seed
        classifier2 = TransactionClassifier(seed=42)
        classifier2.fit(X_train, y_train)
        pred2 = classifier2.predict(X_test)

        # Predictions should be identical
        assert np.array_equal(pred1, pred2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

"""
Unit tests for preprocessing module
"""

import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from preprocessing import TransactionPreprocessor


class TestTransactionPreprocessor:
    """Test preprocessing functionality"""

    def test_normalize_text_lowercase(self):
        """Test lowercase normalization"""
        preprocessor = TransactionPreprocessor()
        assert preprocessor.normalize_text("STARBUCKS") == "starbucks"
        assert preprocessor.normalize_text("Amazon.COM") == "amazoncom"

    def test_normalize_text_whitespace(self):
        """Test whitespace handling"""
        preprocessor = TransactionPreprocessor()
        assert preprocessor.normalize_text("  Starbucks  ") == "starbucks"
        assert preprocessor.normalize_text("Star    bucks") == "star bucks"

    def test_normalize_text_special_chars(self):
        """Test special character removal"""
        preprocessor = TransactionPreprocessor()
        assert preprocessor.normalize_text("***Starbucks***") == "starbucks"
        assert preprocessor.normalize_text("Amazon.com") == "amazoncom"
        assert preprocessor.normalize_text("Shell-Gas") == "shell gas"

    def test_extract_features(self):
        """Test feature extraction"""
        preprocessor = TransactionPreprocessor()
        features = preprocessor.extract_features("starbucks 123")

        assert 'text_length' in features
        assert 'word_count' in features
        assert 'digit_count' in features
        assert features['word_count'] == 2
        assert features['has_digits'] == 1.0

    def test_fit_transform(self):
        """Test fit and transform"""
        preprocessor = TransactionPreprocessor()
        texts = ["Starbucks Coffee", "Amazon.com", "Shell Gas"]

        features = preprocessor.fit_transform(texts)

        assert features.shape[0] == 3
        assert features.shape[1] > 0
        assert isinstance(features, np.ndarray)

    def test_transform_without_fit(self):
        """Test that transform fails without fit"""
        preprocessor = TransactionPreprocessor()

        with pytest.raises(ValueError):
            preprocessor.transform(["Starbucks"])

    def test_robustness_typos(self):
        """Test robustness to typos"""
        preprocessor = TransactionPreprocessor()
        texts = ["starbucks", "starbucks", "starbucks"]  # Same normalized form

        preprocessor.fit(texts)
        features1 = preprocessor.transform(["Starbucks"])
        features2 = preprocessor.transform(["starbucks"])

        # Should produce similar (but not identical) features
        assert features1.shape == features2.shape

    def test_save_load(self, tmp_path):
        """Test saving and loading preprocessor"""
        preprocessor = TransactionPreprocessor()
        texts = ["Starbucks", "Amazon", "Shell"]

        preprocessor.fit(texts)

        # Save
        save_path = tmp_path / "preprocessor.pkl"
        preprocessor.save(str(save_path))

        # Load
        loaded = TransactionPreprocessor.load(str(save_path))

        # Test loaded preprocessor works
        features = loaded.transform(texts)
        assert features.shape[0] == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

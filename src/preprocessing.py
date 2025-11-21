"""
Transaction Preprocessing Pipeline
Handles normalization, tokenization, and feature extraction
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Tuple
import joblib


class TransactionPreprocessor:
    """Robust preprocessing for transaction strings"""

    def __init__(self, max_features=5000, ngram_range=(1, 3), seed=42):
        """
        Initialize preprocessor

        Args:
            max_features: Maximum number of TF-IDF features
            ngram_range: Range for word n-grams
            seed: Random seed for reproducibility
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.seed = seed

        # TF-IDF vectorizer for word-level features
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            lowercase=True,
            analyzer='word',
            token_pattern=r'\b\w+\b',
            min_df=2,
            max_df=0.95
        )

        # Character n-gram vectorizer
        self.char_vectorizer = TfidfVectorizer(
            max_features=max_features // 2,
            ngram_range=(2, 4),
            lowercase=True,
            analyzer='char',
            min_df=2
        )

        self.is_fitted = False

    def normalize_text(self, text: str) -> str:
        """
        Normalize transaction text

        Args:
            text: Raw transaction string

        Returns:
            Normalized text
        """
        if not isinstance(text, str):
            text = str(text)

        # Lowercase
        text = text.lower()

        # Remove leading/trailing whitespace
        text = text.strip()

        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)

        # Remove leading/trailing special characters
        text = text.strip('*@#$%^&()[]{}.,;:!?"\'-_+=/')

        # Replace common separators with spaces
        text = re.sub(r'[/\-_]+', ' ', text)

        # Remove remaining special characters (keep alphanumeric and spaces)
        text = re.sub(r'[^a-z0-9\s]', '', text)

        # Final trim
        text = text.strip()

        return text

    def extract_features(self, text: str) -> Dict[str, float]:
        """
        Extract hand-crafted features from transaction text

        Args:
            text: Normalized transaction string

        Returns:
            Dictionary of features
        """
        features = {}

        # Length-based features
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0

        # Character type features
        features['digit_count'] = sum(c.isdigit() for c in text)
        features['alpha_count'] = sum(c.isalpha() for c in text)
        features['has_digits'] = float(any(c.isdigit() for c in text))

        # Digit ratio
        total_chars = len(text.replace(' ', ''))
        features['digit_ratio'] = features['digit_count'] / total_chars if total_chars > 0 else 0

        # Common merchant indicators
        indicators = ['store', 'shop', 'market', 'station', 'company', 'inc', 'llc']
        features['has_merchant_indicator'] = float(any(ind in text for ind in indicators))

        return features

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize transaction text

        Args:
            text: Normalized text

        Returns:
            List of tokens
        """
        return text.split()

    def fit(self, texts: List[str]) -> 'TransactionPreprocessor':
        """
        Fit the preprocessor on training data

        Args:
            texts: List of raw transaction strings

        Returns:
            Self
        """
        # Normalize texts
        normalized = [self.normalize_text(text) for text in texts]

        # Fit TF-IDF vectorizers
        self.tfidf_vectorizer.fit(normalized)
        self.char_vectorizer.fit(normalized)

        self.is_fitted = True
        return self

    def transform(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform texts to feature vectors

        Args:
            texts: List of raw transaction strings

        Returns:
            Tuple of (word_features, char_features, additional_features)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        # Normalize texts
        normalized = [self.normalize_text(text) for text in texts]

        # TF-IDF features
        word_features = self.tfidf_vectorizer.transform(normalized).toarray()
        char_features = self.char_vectorizer.transform(normalized).toarray()

        # Additional hand-crafted features
        additional_features = np.array([
            list(self.extract_features(text).values())
            for text in normalized
        ])

        # Concatenate all features
        combined_features = np.hstack([
            word_features,
            char_features,
            additional_features
        ])

        return combined_features

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit and transform in one step

        Args:
            texts: List of raw transaction strings

        Returns:
            Feature matrix
        """
        self.fit(texts)
        return self.transform(texts)

    def save(self, path: str):
        """Save preprocessor to disk"""
        joblib.dump({
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'char_vectorizer': self.char_vectorizer,
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'seed': self.seed,
            'is_fitted': self.is_fitted
        }, path)

    @classmethod
    def load(cls, path: str) -> 'TransactionPreprocessor':
        """Load preprocessor from disk"""
        data = joblib.load(path)
        preprocessor = cls(
            max_features=data['max_features'],
            ngram_range=data['ngram_range'],
            seed=data['seed']
        )
        preprocessor.tfidf_vectorizer = data['tfidf_vectorizer']
        preprocessor.char_vectorizer = data['char_vectorizer']
        preprocessor.is_fitted = data['is_fitted']
        return preprocessor


class RobustnessEvaluator:
    """Evaluate preprocessor robustness on various perturbations"""

    def __init__(self, preprocessor: TransactionPreprocessor):
        self.preprocessor = preprocessor

    def test_typos(self, original: str, n_tests=5) -> List[Tuple[str, np.ndarray]]:
        """Test with typo variations"""
        import random

        results = []
        for _ in range(n_tests):
            text = list(original)
            if len(text) > 3:
                idx = random.randint(1, len(text) - 2)
                text[idx] = random.choice('abcdefghijklmnopqrstuvwxyz')
            perturbed = ''.join(text)
            features = self.preprocessor.transform([perturbed])
            results.append((perturbed, features))

        return results

    def test_case_variations(self, text: str) -> List[Tuple[str, np.ndarray]]:
        """Test with case variations"""
        variations = [
            text.lower(),
            text.upper(),
            text.title(),
            ''.join([c.upper() if i % 2 else c.lower() for i, c in enumerate(text)])
        ]

        results = []
        for var in variations:
            features = self.preprocessor.transform([var])
            results.append((var, features))

        return results

    def test_special_chars(self, text: str) -> List[Tuple[str, np.ndarray]]:
        """Test with special character noise"""
        variations = [
            f"***{text}***",
            f"@@@{text}@@@",
            f"  {text}  ",
            f"{text}.",
            f"{text}-",
        ]

        results = []
        for var in variations:
            features = self.preprocessor.transform([var])
            results.append((var, features))

        return results


if __name__ == '__main__':
    # Quick test
    preprocessor = TransactionPreprocessor()

    # Sample transactions
    samples = [
        "Starbucks Coffee #1234",
        "AMAZON.COM",
        "Shell Gas Station",
        "walmart store 456",
        "***CVS PHARMACY***"
    ]

    print("Testing preprocessing pipeline...")
    for text in samples:
        normalized = preprocessor.normalize_text(text)
        features = preprocessor.extract_features(normalized)
        print(f"\nOriginal: {text}")
        print(f"Normalized: {normalized}")
        print(f"Features: {features}")

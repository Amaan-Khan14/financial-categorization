"""
Unit tests for taxonomy loader
"""

import sys
from pathlib import Path
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from taxonomy_loader import TaxonomyLoader


class TestTaxonomyLoader:
    """Test taxonomy loading and management"""

    @pytest.fixture
    def sample_taxonomy_file(self, tmp_path):
        """Create a sample taxonomy file"""
        taxonomy = {
            'version': '1.0',
            'categories': {
                'Food': {
                    'keywords': ['starbucks', 'mcdonalds'],
                    'confidence_boost': 0.05
                },
                'Fuel': {
                    'keywords': ['shell', 'exxon'],
                    'confidence_boost': 0.03
                }
            },
            'confidence_thresholds': {
                'high': 0.85,
                'medium': 0.70,
                'low': 0.60
            }
        }

        file_path = tmp_path / 'taxonomy.yaml'
        with open(file_path, 'w') as f:
            yaml.dump(taxonomy, f)

        return file_path

    def test_load_taxonomy(self, sample_taxonomy_file):
        """Test loading taxonomy from file"""
        loader = TaxonomyLoader(str(sample_taxonomy_file))

        assert loader.get_category_count() == 2
        assert 'Food' in loader.get_categories()
        assert 'Fuel' in loader.get_categories()

    def test_get_keywords(self, sample_taxonomy_file):
        """Test getting keywords for category"""
        loader = TaxonomyLoader(str(sample_taxonomy_file))

        food_keywords = loader.get_keywords('Food')
        assert 'starbucks' in food_keywords
        assert 'mcdonalds' in food_keywords

    def test_get_confidence_boost(self, sample_taxonomy_file):
        """Test getting confidence boost"""
        loader = TaxonomyLoader(str(sample_taxonomy_file))

        boost = loader.get_confidence_boost('Food')
        assert boost == 0.05

    def test_update_category(self, sample_taxonomy_file):
        """Test updating category"""
        loader = TaxonomyLoader(str(sample_taxonomy_file))

        loader.update_category('Shopping', ['amazon', 'walmart'])
        assert 'Shopping' in loader.get_categories()
        assert loader.get_category_count() == 3

    def test_remove_category(self, sample_taxonomy_file):
        """Test removing category"""
        loader = TaxonomyLoader(str(sample_taxonomy_file))

        loader.remove_category('Fuel')
        assert 'Fuel' not in loader.get_categories()
        assert loader.get_category_count() == 1

    def test_keyword_to_category_map(self, sample_taxonomy_file):
        """Test reverse keyword mapping"""
        loader = TaxonomyLoader(str(sample_taxonomy_file))

        keyword_map = loader.keyword_to_category_map()
        assert keyword_map['starbucks'] == 'Food'
        assert keyword_map['shell'] == 'Fuel'

    def test_statistics(self, sample_taxonomy_file):
        """Test statistics generation"""
        loader = TaxonomyLoader(str(sample_taxonomy_file))

        stats = loader.get_statistics()
        assert stats['total_categories'] == 2
        assert stats['version'] == '1.0'
        assert 'Food' in stats['categories']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

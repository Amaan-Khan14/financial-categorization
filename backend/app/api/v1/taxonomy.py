"""
Taxonomy management endpoints
"""
from fastapi import APIRouter, Depends, HTTPException
from app.core.model_loader import ModelLoader
import yaml
from pathlib import Path
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
TAXONOMY_FILE = PROJECT_ROOT / "config" / "taxonomy.yaml"


def get_model_loader():
    """Dependency to get model loader"""
    from app.main import model_loader
    return model_loader


@router.get("/")
async def get_taxonomy(model_loader: ModelLoader = Depends(get_model_loader)):
    """
    Get current taxonomy configuration

    Returns all categories with their keywords
    """
    try:
        # Read taxonomy file
        with open(TAXONOMY_FILE, 'r') as f:
            taxonomy_data = yaml.safe_load(f)

        # Get categories from model loader
        categories = model_loader.get_categories()

        # Format response
        category_info = {}
        for category in categories:
            keywords = taxonomy_data['categories'][category].get('keywords', [])
            category_info[category] = {
                "keywords": keywords,
                "count": len(keywords)
            }

        return {
            "version": taxonomy_data.get('version', '1.0'),
            "categories": category_info,
            "total_categories": len(categories)
        }

    except Exception as e:
        logger.error(f"Failed to get taxonomy: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve taxonomy: {str(e)}")


@router.get("/categories")
async def list_categories(model_loader: ModelLoader = Depends(get_model_loader)):
    """Get list of all available categories"""
    try:
        categories = model_loader.get_categories()
        return {
            "categories": categories,
            "total": len(categories)
        }
    except Exception as e:
        logger.error(f"Failed to list categories: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list categories: {str(e)}")

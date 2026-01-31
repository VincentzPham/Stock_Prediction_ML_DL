"""
API Routes Package.

This module exports all route handlers (APIRouter instances).
"""

from backend.api.routes.root import router as root_router
from backend.api.routes.tickers import router as tickers_router
from backend.api.routes.models import router as models_router
from backend.api.routes.predictions import router as predictions_router
from backend.api.routes.comparison import router as comparison_router
from backend.api.routes.sentiment import sentiment_router

__all__ = [
    "root_router",
    "tickers_router",
    "models_router",
    "predictions_router",
    "comparison_router",
    "sentiment_router",
]

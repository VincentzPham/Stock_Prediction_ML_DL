"""
API Schemas Package.

This module exports all Pydantic models used for request/response validation.
"""

from backend.api.schemas.requests import PredictRequest
from backend.api.schemas.responses import (
    PredictResponse,
    PredictionDay,
    HistoricalResponse,
    HistoricalDataPoint,
    MetricsResponse,
    LatestPriceResponse,
    TickersResponse,
    ModelsResponse,
    MessageResponse,
)

__all__ = [
    "PredictRequest",
    "PredictResponse",
    "PredictionDay",
    "HistoricalResponse",
    "HistoricalDataPoint",
    "MetricsResponse",
    "LatestPriceResponse",
    "TickersResponse",
    "ModelsResponse",
    "MessageResponse",
]

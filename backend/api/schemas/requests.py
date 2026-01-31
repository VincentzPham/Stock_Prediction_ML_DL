"""
Request Schemas.

Pydantic models for API request validation.
"""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """
    Request schema for stock price prediction.
    
    Attributes:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT').
        model: Name of the prediction model to use.
        horizon: Number of trading days to predict ahead (default: 7).
    """
    
    ticker: str = Field(..., description="Stock ticker symbol", example="AAPL")
    model: str = Field(..., description="Model name", example="LSTM")
    horizon: int = Field(default=7, ge=1, le=60, description="Prediction horizon in trading days")

"""
Response Schemas.

Pydantic models for API response serialization.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class MessageResponse(BaseModel):
    """Generic message response."""
    
    message: str


class TickersResponse(BaseModel):
    """Response containing available tickers."""
    
    tickers: List[str]


class ModelsResponse(BaseModel):
    """Response containing available models."""
    
    models: List[str]


class PredictionDay(BaseModel):
    """
    Single day prediction result.
    
    Attributes:
        day: Day number in the prediction sequence (1-indexed).
        date: Predicted date in ISO format (YYYY-MM-DD).
        predicted_price: Predicted stock price in USD.
    """
    
    day: int = Field(..., ge=1, description="Day number in prediction sequence")
    date: str = Field(..., description="Prediction date (YYYY-MM-DD)")
    predicted_price: float = Field(..., description="Predicted price in USD")


class PredictResponse(BaseModel):
    """
    Response schema for stock price prediction.
    
    Attributes:
        ticker: Stock ticker symbol.
        model: Model name used for prediction.
        last_actual_date: Date of the last known actual price.
        last_actual_price: Last known actual closing price.
        horizon: Number of days predicted.
        predictions: List of daily predictions.
        currency: Currency of prices (default: USD).
        model_path: Path to the model file used.
    """
    
    ticker: str
    model: str
    last_actual_date: str
    last_actual_price: float
    horizon: int
    predictions: List[PredictionDay]
    currency: str = "USD"
    model_path: str


class HistoricalDataPoint(BaseModel):
    """Single historical data point."""
    
    date: str = Field(..., description="Date (YYYY-MM-DD)")
    actual: float = Field(..., description="Actual closing price")


class HistoricalResponse(BaseModel):
    """
    Response schema for historical price data.
    
    Attributes:
        ticker: Stock ticker symbol.
        data: List of historical data points.
    """
    
    ticker: str
    data: List[HistoricalDataPoint]


class MetricsResponse(BaseModel):
    """
    Response schema for model evaluation metrics.
    
    Attributes:
        ticker: Stock ticker symbol.
        model: Model name.
        mse: Mean Squared Error.
        rmse: Root Mean Squared Error.
        mae: Mean Absolute Error.
        mape: Mean Absolute Percentage Error.
        r2: R-squared score.
    """
    
    ticker: str
    model: str
    mse: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    mape: Optional[float] = None
    r2: Optional[float] = None


class LatestPriceResponse(BaseModel):
    """
    Response schema for latest stock price.
    
    Attributes:
        ticker: Stock ticker symbol.
        date: Date of the price data.
        close: Closing price.
        open: Opening price.
        high: Highest price of the day.
        low: Lowest price of the day.
        volume: Trading volume.
    """
    
    ticker: str
    date: str
    close: float
    open: float
    high: float
    low: float
    volume: int

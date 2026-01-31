"""
Ticker Routes.

Endpoints for ticker-related operations (list tickers, historical data, latest price).
"""

from fastapi import APIRouter, HTTPException

from backend.config import TICKERS, TARGET_COLUMN
from backend.data.preprocessor import DataPreprocessor
from backend.api.schemas import (
    TickersResponse,
    HistoricalResponse,
    HistoricalDataPoint,
    LatestPriceResponse,
)


router = APIRouter(prefix="/tickers", tags=["Tickers"])


@router.get("", response_model=TickersResponse)
def get_tickers() -> TickersResponse:
    """
    Get list of available stock tickers.
    
    Returns:
        TickersResponse containing list of supported ticker symbols.
    """
    return TickersResponse(tickers=TICKERS)


@router.get("/{ticker}/historical", response_model=HistoricalResponse)
def get_historical(ticker: str, days: int = 60) -> HistoricalResponse:
    """
    Get historical price data for a ticker.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL').
        days: Number of historical days to return (default: 60).
    
    Returns:
        HistoricalResponse containing list of date/price pairs.
        
    Raises:
        HTTPException: 400 if ticker is invalid, 500 on server error.
    """
    if ticker not in TICKERS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid ticker. Available: {TICKERS}"
        )
    
    try:
        preprocessor = DataPreprocessor(ticker)
        preprocessor.load_data()
        
        # Get last N days
        df = preprocessor.df.tail(days)
        
        data = []
        for idx, row in df.iterrows():
            data.append(
                HistoricalDataPoint(
                    date=str(idx.date()),
                    actual=round(float(row[TARGET_COLUMN]), 2)
                )
            )
        
        return HistoricalResponse(ticker=ticker, data=data)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{ticker}/latest", response_model=LatestPriceResponse)
def get_latest_price(ticker: str) -> LatestPriceResponse:
    """
    Get the latest price data for a ticker.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL').
    
    Returns:
        LatestPriceResponse containing OHLCV data for the latest date.
        
    Raises:
        HTTPException: 400 if ticker is invalid, 500 on server error.
    """
    if ticker not in TICKERS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid ticker. Available: {TICKERS}"
        )
    
    try:
        preprocessor = DataPreprocessor(ticker)
        preprocessor.load_data()
        
        last_row = preprocessor.df.iloc[-1]
        last_date = preprocessor.df.index[-1]
        
        return LatestPriceResponse(
            ticker=ticker,
            date=str(last_date.date()),
            close=round(float(last_row[TARGET_COLUMN]), 2),
            open=round(float(last_row["Open"]), 2),
            high=round(float(last_row["High"]), 2),
            low=round(float(last_row["Low"]), 2),
            volume=int(last_row["Volume"]),
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

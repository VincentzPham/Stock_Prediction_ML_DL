"""
Sentiment API Routes.

Provides endpoints for accessing sentiment analysis data.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any, Optional
import pandas as pd

from backend.config import DATA_DIR, TICKERS


sentiment_router = APIRouter(prefix="/sentiment", tags=["sentiment"])

# Sentiment data directory
SENTIMENT_DIR = DATA_DIR / "sentiment"


def _load_sentiment_data(ticker: str) -> Optional[pd.DataFrame]:
    """
    Load daily sentiment data for a ticker.
    
    Args:
        ticker: Stock ticker symbol.
        
    Returns:
        DataFrame with sentiment data, or None if not available.
    """
    # Try v2 format first (more complete)
    v2_path = SENTIMENT_DIR / f"{ticker}_daily_v2.csv"
    v1_path = SENTIMENT_DIR / f"{ticker}_daily.csv"
    
    if v2_path.exists():
        try:
            df = pd.read_csv(v2_path)
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date", ascending=False)
            return df
        except Exception:
            pass
    
    if v1_path.exists():
        try:
            df = pd.read_csv(v1_path)
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date", ascending=False)
            return df
        except Exception:
            pass
    
    return None


@sentiment_router.get("/available")
async def get_available_sentiment() -> Dict[str, Any]:
    """
    Get list of tickers with available sentiment data.
    
    Returns:
        Dictionary with list of tickers that have sentiment data.
    """
    available = []
    
    for ticker in TICKERS:
        v2_path = SENTIMENT_DIR / f"{ticker}_daily_v2.csv"
        v1_path = SENTIMENT_DIR / f"{ticker}_daily.csv"
        
        if v2_path.exists() or v1_path.exists():
            available.append(ticker)
    
    return {
        "available_tickers": available,
        "total_tickers": len(TICKERS),
        "coverage": len(available) / len(TICKERS) * 100,
    }


@sentiment_router.get("/{ticker}/daily")
async def get_daily_sentiment(
    ticker: str,
    days: int = 30
) -> Dict[str, Any]:
    """
    Get daily sentiment data for a ticker.
    
    Args:
        ticker: Stock ticker symbol.
        days: Number of days to return (default 30).
        
    Returns:
        Dictionary with daily sentiment data.
    """
    if ticker not in TICKERS:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found")
    
    df = _load_sentiment_data(ticker)
    
    if df is None or df.empty:
        return {
            "ticker": ticker,
            "available": False,
            "message": f"No sentiment data available for {ticker}",
            "data": [],
        }
    
    # Limit to requested days
    df = df.head(days)
    
    # Convert to records
    records = []
    for _, row in df.iterrows():
        record = {
            "date": row["Date"].strftime("%Y-%m-%d"),
            "sentiment_score": round(row.get("Sentiment_Score", 0), 4),
        }
        
        # Add optional fields if present
        if "News_Count" in df.columns:
            record["news_count"] = int(row.get("News_Count", 0))
        if "Sources" in df.columns:
            record["sources"] = row.get("Sources", "")
        if "Sentiment_Score_Raw" in df.columns:
            record["raw_score"] = round(row.get("Sentiment_Score_Raw", 0), 4)
        
        records.append(record)
    
    return {
        "ticker": ticker,
        "available": True,
        "days": len(records),
        "data": records,
    }


@sentiment_router.get("/{ticker}/latest")
async def get_latest_sentiment(ticker: str) -> Dict[str, Any]:
    """
    Get latest sentiment data for a ticker.
    
    Args:
        ticker: Stock ticker symbol.
        
    Returns:
        Dictionary with latest sentiment data.
    """
    if ticker not in TICKERS:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found")
    
    df = _load_sentiment_data(ticker)
    
    if df is None or df.empty:
        return {
            "ticker": ticker,
            "available": False,
            "message": f"No sentiment data available for {ticker}",
        }
    
    # Get the most recent entry
    latest = df.iloc[0]
    
    result = {
        "ticker": ticker,
        "available": True,
        "date": latest["Date"].strftime("%Y-%m-%d"),
        "sentiment_score": round(latest.get("Sentiment_Score", 0), 4),
        "sentiment_label": _get_sentiment_label(latest.get("Sentiment_Score", 0)),
    }
    
    # Add optional fields
    if "News_Count" in df.columns:
        result["news_count"] = int(latest.get("News_Count", 0))
    if "Sources" in df.columns:
        result["sources"] = latest.get("Sources", "")
    
    return result


@sentiment_router.get("/overview")
async def get_sentiment_overview() -> List[Dict[str, Any]]:
    """
    Get sentiment overview for all tickers with available data.
    
    Returns:
        List of dictionaries with latest sentiment for each ticker.
    """
    overview = []
    
    for ticker in TICKERS:
        df = _load_sentiment_data(ticker)
        
        data = {
            "ticker": ticker,
            "available": False,
        }
        
        if df is not None and not df.empty:
            latest = df.iloc[0]
            score = latest.get("Sentiment_Score", 0)
            
            data["available"] = True
            data["date"] = latest["Date"].strftime("%Y-%m-%d")
            data["sentiment_score"] = round(score, 4)
            data["sentiment_label"] = _get_sentiment_label(score)
            
            if "News_Count" in df.columns:
                data["news_count"] = int(latest.get("News_Count", 0))
        
        overview.append(data)
    
    return overview


def _get_sentiment_label(score: float) -> str:
    """
    Convert sentiment score to human-readable label.
    
    Args:
        score: Sentiment score (-1 to 1).
        
    Returns:
        Sentiment label string.
    """
    if score > 0.2:
        return "Bullish"
    elif score > 0.05:
        return "Slightly Bullish"
    elif score > -0.05:
        return "Neutral"
    elif score > -0.2:
        return "Slightly Bearish"
    else:
        return "Bearish"

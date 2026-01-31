"""
Caching utilities for frontend API calls.

Uses Streamlit's cache_data decorator to cache expensive API calls.
Default TTL is 5 minutes (300 seconds).
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Any

from frontend.api_client import (
    api_client,
    get_sentiment_overview as _get_sentiment_overview,
    get_ticker_sentiment as _get_ticker_sentiment,
)


# Cache TTL in seconds (5 minutes)
CACHE_TTL = 300


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_cached_leaderboard() -> Dict[str, Any]:
    """
    Get cached global leaderboard data.
    
    Returns:
        Dictionary with leaderboard and summary statistics.
    """
    return api_client.get_leaderboard()


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_cached_market_overview() -> List[Dict[str, Any]]:
    """
    Get cached market overview with latest prices.
    
    Returns:
        List of ticker data with latest prices.
    """
    return api_client.get_market_overview()


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_cached_historical_data(ticker: str, days: int = 60) -> pd.DataFrame:
    """
    Get cached historical price data for a ticker.
    
    Args:
        ticker: Stock ticker symbol.
        days: Number of historical days to fetch.
        
    Returns:
        DataFrame with columns ['date', 'actual'].
    """
    return api_client.get_historical_data(ticker, days)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_cached_metrics(ticker: str, model: str) -> Dict[str, Any]:
    """
    Get cached evaluation metrics for a trained model.
    
    Args:
        ticker: Stock ticker symbol.
        model: Model name.
        
    Returns:
        Dictionary with metrics (mse, rmse, mae, mape, r2).
    """
    return api_client.get_metrics(ticker, model)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_cached_ticker_comparison(ticker: str) -> Dict[str, Any]:
    """
    Get cached comparison of all models for a specific ticker.
    
    Args:
        ticker: Stock ticker symbol.
        
    Returns:
        Dictionary with comparison data for all models.
    """
    return api_client.get_ticker_comparison(ticker)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_cached_sentiment_overview() -> List[Dict[str, Any]]:
    """
    Get cached sentiment overview for all tickers.
    
    Returns:
        List of dictionaries with latest sentiment for each ticker.
    """
    return _get_sentiment_overview()


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_cached_ticker_sentiment(ticker: str, days: int = 30) -> Dict[str, Any]:
    """
    Get cached daily sentiment data for a ticker.
    
    Args:
        ticker: Stock ticker symbol.
        days: Number of days to return.
        
    Returns:
        Dictionary with sentiment data.
    """
    return _get_ticker_sentiment(ticker, days)


def clear_all_cache() -> None:
    """Clear all cached data."""
    get_cached_leaderboard.clear()
    get_cached_market_overview.clear()
    get_cached_historical_data.clear()
    get_cached_metrics.clear()
    get_cached_ticker_comparison.clear()
    get_cached_sentiment_overview.clear()
    get_cached_ticker_sentiment.clear()

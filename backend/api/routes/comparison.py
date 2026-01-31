"""
Comparison Routes.

Endpoints for model comparison and leaderboard functionality.
"""

from fastapi import APIRouter

from backend.config import TICKERS
from backend.api.services.comparison_service import ComparisonService


router = APIRouter(prefix="/compare", tags=["Comparison"])


@router.get("/{ticker}")
def get_ticker_comparison(ticker: str):
    """
    Get comparison of all models for a specific ticker.
    
    Args:
        ticker: Stock ticker symbol.
        
    Returns:
        Comparison data for all models for the specified ticker.
    """
    if ticker not in TICKERS:
        return {"error": f"Invalid ticker. Available: {TICKERS}"}
    
    return ComparisonService.get_ticker_comparison(ticker)


@router.get("/leaderboard/all")
def get_leaderboard():
    """
    Get global leaderboard across all tickers and models.
    
    Returns top performing models sorted by MAPE (lowest = best).
    Includes summary statistics and coverage information.
    
    Returns:
        Leaderboard data with rankings and statistics.
    """
    return ComparisonService.get_leaderboard()


@router.get("/market/overview")
def get_market_overview():
    """
    Get market overview with latest prices for all tickers.
    
    Returns:
        List of tickers with latest price data and daily changes.
    """
    return ComparisonService.get_market_overview()

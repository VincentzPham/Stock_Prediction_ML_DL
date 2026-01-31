"""
API Client Module.

Handles all communication with the backend API.
"""

import requests
import pandas as pd
from typing import Dict, List, Any

from frontend.config import API_URL, REQUEST_TIMEOUT


class APIClient:
    """
    Client for communicating with the Stock Prediction API.
    
    Provides methods for fetching data and making predictions
    using the backend API endpoints.
    """
    
    def __init__(self, base_url: str = API_URL):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL of the API server.
        """
        self.base_url = base_url.rstrip("/")
    
    def get_tickers(self) -> List[str]:
        """
        Fetch list of available tickers.
        
        Returns:
            List of ticker symbols.
        """
        try:
            response = requests.get(f"{self.base_url}/tickers", timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                return response.json().get("tickers", [])
        except Exception:
            pass
        return []
    
    def get_models(self) -> List[str]:
        """
        Fetch list of available models.
        
        Returns:
            List of model names.
        """
        try:
            response = requests.get(f"{self.base_url}/models", timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                return response.json().get("models", [])
        except Exception:
            pass
        return []
    
    def get_historical_data(self, ticker: str, days: int = 60) -> pd.DataFrame:
        """
        Fetch historical price data for a ticker.
        
        Args:
            ticker: Stock ticker symbol.
            days: Number of historical days to fetch.
            
        Returns:
            DataFrame with columns ['date', 'actual'].
        """
        try:
            response = requests.get(
                f"{self.base_url}/tickers/{ticker}/historical",
                params={"days": days},
                timeout=REQUEST_TIMEOUT
            )
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data["data"])
                df["date"] = pd.to_datetime(df["date"])
                return df
        except Exception:
            pass
        return pd.DataFrame()
    
    def get_latest_price(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch latest price data for a ticker.
        
        Args:
            ticker: Stock ticker symbol.
            
        Returns:
            Dictionary with price data (close, open, high, low, volume, date).
        """
        try:
            response = requests.get(
                f"{self.base_url}/tickers/{ticker}/latest",
                timeout=REQUEST_TIMEOUT
            )
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return {}
    
    def get_metrics(self, ticker: str, model: str) -> Dict[str, Any]:
        """
        Fetch evaluation metrics for a trained model.
        
        Args:
            ticker: Stock ticker symbol.
            model: Model name.
            
        Returns:
            Dictionary with metrics (mse, rmse, mae, mape, r2).
        """
        try:
            response = requests.get(
                f"{self.base_url}/models/{ticker}/{model}/metrics",
                timeout=REQUEST_TIMEOUT
            )
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return {}
    
    def predict(
        self,
        ticker: str,
        model: str,
        horizon: int
    ) -> Dict[str, Any]:
        """
        Generate price predictions.
        
        Args:
            ticker: Stock ticker symbol.
            model: Model name to use for prediction.
            horizon: Number of days to predict.
            
        Returns:
            Dictionary with prediction results.
            
        Raises:
            requests.exceptions.ConnectionError: If cannot connect to API.
            requests.exceptions.Timeout: If request times out.
            requests.exceptions.HTTPError: If API returns error status.
        """
        payload = {
            "ticker": ticker,
            "model": model,
            "horizon": horizon,
        }
        
        response = requests.post(
            f"{self.base_url}/predictions",
            json=payload,
            timeout=REQUEST_TIMEOUT
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            # Raise exception with error detail
            error_detail = response.json().get("detail", response.text)
            raise requests.exceptions.HTTPError(error_detail)
    
    def get_ticker_comparison(self, ticker: str) -> Dict[str, Any]:
        """
        Get comparison of all models for a specific ticker.
        
        Args:
            ticker: Stock ticker symbol.
            
        Returns:
            Dictionary with comparison data for all models.
        """
        try:
            response = requests.get(
                f"{self.base_url}/compare/{ticker}",
                timeout=REQUEST_TIMEOUT
            )
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return {}
    
    def get_leaderboard(self) -> Dict[str, Any]:
        """
        Get global leaderboard data.
        
        Returns:
            Dictionary with leaderboard and summary statistics.
        """
        try:
            response = requests.get(
                f"{self.base_url}/compare/leaderboard/all",
                timeout=REQUEST_TIMEOUT
            )
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return {}
    
    def get_market_overview(self) -> List[Dict[str, Any]]:
        """
        Get market overview with latest prices.
        
        Returns:
            List of ticker data with latest prices.
        """
        try:
            response = requests.get(
                f"{self.base_url}/compare/market/overview",
                timeout=REQUEST_TIMEOUT
            )
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return []


# Default client instance
api_client = APIClient()


# Sentiment API methods (added as module-level functions for easier caching)
def get_sentiment_overview() -> List[Dict[str, Any]]:
    """
    Get sentiment overview for all tickers.
    
    Returns:
        List of dictionaries with latest sentiment for each ticker.
    """
    try:
        response = requests.get(
            f"{api_client.base_url}/sentiment/overview",
            timeout=REQUEST_TIMEOUT
        )
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return []


def get_ticker_sentiment(ticker: str, days: int = 30) -> Dict[str, Any]:
    """
    Get daily sentiment data for a ticker.
    
    Args:
        ticker: Stock ticker symbol.
        days: Number of days to return.
        
    Returns:
        Dictionary with sentiment data.
    """
    try:
        response = requests.get(
            f"{api_client.base_url}/sentiment/{ticker}/daily",
            params={"days": days},
            timeout=REQUEST_TIMEOUT
        )
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return {"available": False, "data": []}

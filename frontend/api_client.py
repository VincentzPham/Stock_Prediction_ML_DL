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


# Default client instance
api_client = APIClient()

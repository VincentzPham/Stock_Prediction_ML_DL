"""
Exponential Smoothing Model
Holt-Winters Exponential Smoothing for stock prediction.
"""

import warnings

import numpy as np
import pandas as pd
from typing import Dict, Any

from backend.config import TS_CONFIG
from backend.models.base import BaseModel

warnings.filterwarnings("ignore")


class ExponentialSmoothingModel(BaseModel):
    """
    Exponential Smoothing (Holt-Winters) model cho stock price prediction.
    """

    MODEL_NAME = "Exponential Smoothing"
    MODEL_TYPE = "time_series"

    def __init__(self, ticker: str):
        super().__init__(ticker)
        self.config = TS_CONFIG.get("Exponential Smoothing", {})

    def build(
        self,
        trend: str = None,
        seasonal: str = None,
        seasonal_periods: int = None,
        **kwargs,
    ) -> None:
        """
        Cấu hình Exponential Smoothing model.

        Args:
            trend: 'add', 'mul', hoặc None
            seasonal: 'add', 'mul', hoặc None
            seasonal_periods: Số periods trong 1 season
        """
        self.trend = trend or self.config.get("trend", "add")
        self.seasonal = seasonal or self.config.get("seasonal", None)
        self.seasonal_periods = seasonal_periods or self.config.get(
            "seasonal_periods", None
        )

        print(
            f"Exponential Smoothing configured: trend={self.trend}, seasonal={self.seasonal}"
        )

    def train(
        self,
        train_series: pd.Series,
        y_train=None,  # unused
        X_val=None,  # unused
        y_val=None,  # unused
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train Exponential Smoothing model.
        """
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        if not hasattr(self, "trend"):
            self.build()

        print(f"\nTraining {self.MODEL_NAME} for {self.ticker}...")
        print(f"  Training samples: {len(train_series)}")

        # Xử lý seasonal
        seasonal = self.seasonal
        seasonal_periods = self.seasonal_periods

        # Nếu data không đủ cho seasonality, disable nó
        if seasonal_periods and len(train_series) < 2 * seasonal_periods:
            print("  Warning: Data too short for seasonality, disabling...")
            seasonal = None
            seasonal_periods = None

        model = ExponentialSmoothing(
            train_series,
            trend=self.trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
        )
        self.model = model.fit()

        self.is_trained = True
        self.train_series = train_series

        return {"aic": self.model.aic, "bic": self.model.bic}

    def predict(self, steps: int = None, test_series: pd.Series = None) -> np.ndarray:
        """
        Dự đoán giá stock.
        """
        if self.model is None:
            raise ValueError("Model chưa được train.")

        if steps is not None:
            forecast = self.model.forecast(steps=steps)
            return np.array(forecast)
        elif test_series is not None:
            # Forecast cho số steps = len(test_series)
            forecast = self.model.forecast(steps=len(test_series))
            return np.array(forecast)
        else:
            raise ValueError("Phải cung cấp steps hoặc test_series.")

    def predict_next(self, preprocessor, horizon: int = 1) -> float:
        """
        Dự đoán giá trị tương lai.
        """
        if self.model is None:
            raise ValueError("Model chưa được load/train.")

        # Forecast horizon steps ahead
        forecast = self.model.forecast(steps=horizon)
        # Return the last prediction (for the horizon-th day)
        prediction = forecast.iloc[-1] if hasattr(forecast, "iloc") else forecast[-1]

        return float(prediction)

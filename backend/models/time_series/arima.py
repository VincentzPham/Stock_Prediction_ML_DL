"""
ARIMA Model
AutoRegressive Integrated Moving Average for stock prediction.
"""

import warnings

import numpy as np
import pandas as pd
from typing import Dict, Any

from backend.config import TS_CONFIG
from backend.models.base import BaseModel

warnings.filterwarnings("ignore")


class ARIMAModel(BaseModel):
    """
    ARIMA model cho stock price prediction.
    """

    MODEL_NAME = "ARIMA"
    MODEL_TYPE = "time_series"

    def __init__(self, ticker: str):
        super().__init__(ticker)
        self.config = TS_CONFIG.get("ARIMA", {})
        self.order = self.config.get("order", (4, 1, 0))

    def build(self, order: tuple = None, **kwargs) -> None:
        """
        Cấu hình ARIMA order.

        Args:
            order: Tuple (p, d, q)
        """
        if order is not None:
            self.order = order
        print(f"ARIMA configured with order={self.order}")

    def train(
        self,
        train_series: pd.Series,
        y_train=None,  # unused, for compatibility
        X_val=None,  # unused
        y_val=None,  # unused
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train ARIMA model.

        Args:
            train_series: Time series data for training.
        """
        from statsmodels.tsa.arima.model import ARIMA

        print(f"\nTraining {self.MODEL_NAME} for {self.ticker}...")
        print(f"  Order: {self.order}")
        print(f"  Training samples: {len(train_series)}")

        model = ARIMA(train_series, order=self.order)
        self.model = model.fit()

        self.is_trained = True

        # Return summary info
        return {"aic": self.model.aic, "bic": self.model.bic, "order": self.order}

    def predict(self, steps: int = None, test_series: pd.Series = None) -> np.ndarray:
        """
        Dự đoán giá stock.

        Có 2 modes:
        1. Forecast n steps ahead: predict(steps=n)
        2. Rolling forecast với actual values: predict(test_series=series)
        """
        if self.model is None:
            raise ValueError("Model chưa được train.")

        if test_series is not None:
            # Rolling forecast
            return self._rolling_forecast(test_series)
        elif steps is not None:
            # Simple forecast
            forecast = self.model.forecast(steps=steps)
            return np.array(forecast)
        else:
            raise ValueError("Phải cung cấp steps hoặc test_series.")

    def predict_next(self, preprocessor, horizon: int = 1) -> float:
        """
        Dự đoán giá trị tương lai.
        """
        if self.model is None:
            raise ValueError("Model chưa được train.")

        # Forecast n steps
        forecast = self.model.forecast(steps=horizon)

        # Return last value
        val = forecast.iloc[-1] if hasattr(forecast, "iloc") else forecast[-1]
        return float(val)

    def _rolling_forecast(self, test_series: pd.Series) -> np.ndarray:
        """
        Rolling forecast: dự đoán từng bước và cập nhật model.
        """
        from statsmodels.tsa.arima.model import ARIMA

        predictions = []

        # Get original training data
        train_data = list(self.model.model.endog)

        print(f"  Rolling forecast for {len(test_series)} steps...")

        for i, actual_value in enumerate(test_series):
            # Fit model on history
            model = ARIMA(train_data, order=self.order)
            model_fit = model.fit()

            # Forecast
            forecast = model_fit.forecast(steps=1)
            predictions.append(
                forecast.iloc[0] if hasattr(forecast, "iloc") else forecast[0]
            )

            # Append actual value to history
            train_data.append(actual_value)

            if (i + 1) % 100 == 0:
                print(f"    Completed {i + 1}/{len(test_series)} steps")

        return np.array(predictions)

    def get_summary(self) -> str:
        """Trả về summary của model."""
        if self.model is None:
            return "Model not trained"
        return str(self.model.summary())

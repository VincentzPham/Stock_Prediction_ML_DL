"""
SARIMA Model
Seasonal ARIMA for stock prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import sys
import warnings

warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent.parent))
from config import TS_CONFIG

from .base import BaseModel


class SARIMAModel(BaseModel):
    """
    SARIMA (Seasonal ARIMA) model cho stock price prediction.
    """
    
    MODEL_NAME = "SARIMA"
    MODEL_TYPE = "time_series"
    
    def __init__(self, ticker: str):
        super().__init__(ticker)
        self.config = TS_CONFIG.get('SARIMA', {})
        self.order = self.config.get('order', (4, 1, 0))
        self.seasonal_order = self.config.get('seasonal_order', (1, 1, 1, 7))
        
    def build(
        self, 
        order: tuple = None,
        seasonal_order: tuple = None,
        **kwargs
    ) -> None:
        """
        Cấu hình SARIMA order.
        
        Args:
            order: Tuple (p, d, q)
            seasonal_order: Tuple (P, D, Q, m)
        """
        if order is not None:
            self.order = order
        if seasonal_order is not None:
            self.seasonal_order = seasonal_order
            
        print(f"SARIMA configured with order={self.order}, seasonal_order={self.seasonal_order}")
        
    def train(
        self, 
        train_series: pd.Series, 
        y_train=None,  # unused
        X_val=None,    # unused
        y_val=None,    # unused
        use_log: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train SARIMA model.
        
        Args:
            train_series: Time series data for training.
            use_log: Có log transform data không (recommended).
        """
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        
        print(f"\nTraining {self.MODEL_NAME} for {self.ticker}...")
        print(f"  Order: {self.order}")
        print(f"  Seasonal Order: {self.seasonal_order}")
        print(f"  Training samples: {len(train_series)}")
        print(f"  Log transform: {use_log}")
        
        self.use_log = use_log
        
        # Log transform
        if use_log:
            self.train_data = np.log(train_series)
        else:
            self.train_data = train_series
        
        model = SARIMAX(
            self.train_data, 
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self.model = model.fit(disp=False)
        
        self.is_trained = True
        
        return {
            'aic': self.model.aic,
            'bic': self.model.bic,
            'order': self.order,
            'seasonal_order': self.seasonal_order
        }
    
    def predict(self, steps: int = None, test_series: pd.Series = None) -> np.ndarray:
        """
        Dự đoán giá stock.
        """
        if self.model is None:
            raise ValueError("Model chưa được train.")
        
        if test_series is not None:
            return self._rolling_forecast(test_series)
        elif steps is not None:
            forecast = self.model.forecast(steps=steps)
            if self.use_log:
                forecast = np.exp(forecast)
            return np.array(forecast)
        else:
            raise ValueError("Phải cung cấp steps hoặc test_series.")
    
    def _rolling_forecast(self, test_series: pd.Series) -> np.ndarray:
        """
        Rolling forecast với SARIMA.
        """
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        
        predictions = []
        
        if self.use_log:
            train_data = list(np.log(test_series.iloc[:0]))  # empty start
            history = list(self.train_data)
        else:
            history = list(self.train_data)
        
        print(f"  Rolling forecast for {len(test_series)} steps...")
        
        for i, actual_value in enumerate(test_series):
            # Fit model
            model = SARIMAX(
                history, 
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            model_fit = model.fit(disp=False)
            
            # Forecast
            forecast = model_fit.forecast(steps=1)
            yhat = forecast.iloc[0] if hasattr(forecast, 'iloc') else forecast[0]
            
            # Inverse log if needed
            if self.use_log:
                predictions.append(np.exp(yhat))
                history.append(np.log(actual_value))
            else:
                predictions.append(yhat)
                history.append(actual_value)
            
            if (i + 1) % 100 == 0:
                print(f"    Completed {i + 1}/{len(test_series)} steps")
        
        return np.array(predictions)

    def predict_next(self, preprocessor, horizon: int = 1) -> float:
        """
        Dự đoán giá trị tương lai.
        """
        if self.model is None:
            raise ValueError("Model chưa được load/train.")
        
        # Forecast horizon steps ahead
        forecast = self.model.forecast(steps=horizon)
        # Return the last prediction (for the horizon-th day)
        yhat = forecast.iloc[-1] if hasattr(forecast, 'iloc') else forecast[-1]
        
        # Inverse log if model was trained with log transform
        if hasattr(self, 'use_log') and self.use_log:
            prediction = np.exp(yhat)
        else:
            prediction = yhat
        
        return float(prediction)

"""
Prophet Model
Facebook Prophet for stock prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

from backend.config import TS_CONFIG
from backend.models.base import BaseModel


class ProphetModel(BaseModel):
    """
    Prophet model cho stock price prediction.
    """

    MODEL_NAME = "Prophet"
    MODEL_TYPE = "time_series"

    def __init__(self, ticker: str):
        super().__init__(ticker)
        self.config = TS_CONFIG.get("Prophet", {})

    def build(
        self,
        yearly_seasonality: bool = None,
        weekly_seasonality: bool = None,
        daily_seasonality: bool = None,
        changepoint_prior_scale: float = None,
        **kwargs,
    ) -> None:
        """
        Cấu hình Prophet model.
        """
        from prophet import Prophet

        self.model = Prophet(
            yearly_seasonality=yearly_seasonality
            or self.config.get("yearly_seasonality", True),
            weekly_seasonality=weekly_seasonality
            or self.config.get("weekly_seasonality", True),
            daily_seasonality=daily_seasonality
            or self.config.get("daily_seasonality", False),
            changepoint_prior_scale=changepoint_prior_scale
            or self.config.get("changepoint_prior_scale", 0.05),
        )

        print("Prophet model configured")

    def train(
        self,
        train_df: pd.DataFrame,  # DataFrame với columns ['ds', 'y']
        y_train=None,  # unused
        X_val=None,  # unused
        y_val=None,  # unused
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train Prophet model.

        Args:
            train_df: DataFrame với columns 'ds' (date) và 'y' (value)
        """
        if self.model is None:
            self.build()

        print(f"\nTraining {self.MODEL_NAME} for {self.ticker}...")
        print(f"  Training samples: {len(train_df)}")

        self.model.fit(train_df)

        self.is_trained = True

        return {"status": "trained"}

    def predict(self, periods: int = None, test_df: pd.DataFrame = None) -> np.ndarray:
        """
        Dự đoán giá stock.

        Args:
            periods: Số ngày cần dự đoán
            test_df: DataFrame với dates cần dự đoán
        """
        if self.model is None:
            raise ValueError("Model chưa được train.")

        if test_df is not None:
            # Dự đoán trên dates cụ thể
            future = test_df[["ds"]].copy()
        elif periods is not None:
            # Tạo future dates
            future = self.model.make_future_dataframe(periods=periods)
            future = future.tail(periods)
        else:
            raise ValueError("Phải cung cấp periods hoặc test_df.")

        forecast = self.model.predict(future)

        return forecast["yhat"].values

    def predict_next(self, preprocessor, horizon: int = 1) -> float:
        """
        Dự đoán giá trị tương lai.
        """
        if self.model is None:
            raise ValueError("Model chưa được train.")

        # Tạo future dates
        future = self.model.make_future_dataframe(periods=horizon)

        # Predict
        forecast = self.model.predict(future)

        return float(forecast["yhat"].iloc[-1])

    def plot_components(self, save: bool = True) -> Optional[Path]:
        """Vẽ các components của Prophet."""
        if self.model is None:
            raise ValueError("Model chưa được train.")

        import matplotlib.pyplot as plt
        from datetime import datetime

        future = self.model.make_future_dataframe(periods=0)
        forecast = self.model.predict(future)

        fig = self.model.plot_components(forecast)

        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = (
                self.result_dir
                / f"{self.ticker}_{self.MODEL_NAME}_components_{timestamp}.png"
            )
            fig.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close()
            return filepath
        else:
            plt.show()
            return None

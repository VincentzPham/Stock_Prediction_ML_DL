"""
Decision Tree Model
Decision Tree Regressor for stock prediction.
"""

import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path

from backend.config import ML_CONFIG
from backend.models.base import BaseModel


class DecisionTreeModel(BaseModel):
    """
    Decision Tree model cho stock price prediction.
    """

    MODEL_NAME = "Decision Tree"
    MODEL_TYPE = "ml"

    def __init__(self, ticker: str):
        super().__init__(ticker)
        self.config = ML_CONFIG.get("Decision Tree", {})

    def build(
        self,
        max_depth: int = None,
        min_samples_split: int = None,
        min_samples_leaf: int = None,
        random_state: int = None,
        **kwargs,
    ) -> None:
        """
        Xây dựng Decision Tree model.
        """
        from sklearn.tree import DecisionTreeRegressor

        self.model = DecisionTreeRegressor(
            max_depth=max_depth or self.config.get("max_depth", None),
            min_samples_split=min_samples_split
            or self.config.get("min_samples_split", 2),
            min_samples_leaf=min_samples_leaf or self.config.get("min_samples_leaf", 1),
            random_state=random_state or self.config.get("random_state", 42),
        )

        print("Decision Tree model built")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train Decision Tree model.
        """
        if self.model is None:
            self.build()

        print(f"\nTraining {self.MODEL_NAME} for {self.ticker}...")
        print(f"  Training samples: {len(X_train)}")

        self.model.fit(X_train, y_train)

        self.is_trained = True

        return {
            "max_depth": self.model.get_depth(),
            "n_leaves": self.model.get_n_leaves(),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Dự đoán giá stock.
        """
        if self.model is None:
            raise ValueError("Model chưa được build/train.")

        predictions = self.model.predict(X)
        return predictions

    def predict_next(self, preprocessor, horizon: int = 1) -> float:
        """
        Dự đoán giá trị tương lai.
        """
        if self.model is None:
            raise ValueError("Model chưa được load/train.")

        # Get features used for ML
        feature_cols = [
            "Open",
            "High",
            "Low",
            "Volume",
            "Price_Diff",
            "Avg_Price",
            "Volume_Ratio",
        ]
        # Ensure columns exist in the dataframe
        feature_cols = [c for c in feature_cols if c in preprocessor.df.columns]

        # Get the last row of data (most recent day)
        X_input = preprocessor.df.iloc[[-1]][feature_cols].values

        # Predict
        prediction = self.model.predict(X_input)[0]

        return float(prediction)

    def plot_tree(
        self, feature_names: list = None, save: bool = True
    ) -> Optional[Path]:
        """Vẽ cây quyết định."""
        import matplotlib.pyplot as plt
        from sklearn.tree import plot_tree
        from datetime import datetime

        if self.model is None:
            raise ValueError("Model chưa được train.")

        plt.figure(figsize=(20, 10))
        plot_tree(
            self.model,
            feature_names=feature_names,
            filled=True,
            rounded=True,
            max_depth=3,  # Limit depth for visualization
        )
        plt.title(f"{self.ticker} - {self.MODEL_NAME}")

        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = (
                self.result_dir
                / f"{self.ticker}_{self.MODEL_NAME}_tree_{timestamp}.png"
            )
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
            plt.close()
            return filepath
        else:
            plt.show()
            return None

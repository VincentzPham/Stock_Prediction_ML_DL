"""
Random Forest Model
Random Forest Regressor for stock prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import ML_CONFIG

from .base import BaseModel


class RandomForestModel(BaseModel):
    """
    Random Forest model cho stock price prediction.
    """
    
    MODEL_NAME = "Random Forest"
    MODEL_TYPE = "ml"
    
    def __init__(self, ticker: str):
        super().__init__(ticker)
        self.config = ML_CONFIG.get('Random Forest', {})
        
    def build(
        self,
        n_estimators: int = None,
        max_depth: int = None,
        min_samples_split: int = None,
        min_samples_leaf: int = None,
        random_state: int = None,
        **kwargs
    ) -> None:
        """
        Xây dựng Random Forest model.
        """
        from sklearn.ensemble import RandomForestRegressor
        
        self.model = RandomForestRegressor(
            n_estimators=n_estimators or self.config.get('n_estimators', 100),
            max_depth=max_depth or self.config.get('max_depth', None),
            min_samples_split=min_samples_split or self.config.get('min_samples_split', 2),
            min_samples_leaf=min_samples_leaf or self.config.get('min_samples_leaf', 1),
            random_state=random_state or self.config.get('random_state', 42),
            n_jobs=-1  # Use all CPU cores
        )
        
        print(f"Random Forest model built with n_estimators={self.model.n_estimators}")
        
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        X_val: np.ndarray = None, 
        y_val: np.ndarray = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train Random Forest model.
        """
        if self.model is None:
            self.build()
        
        print(f"\nTraining {self.MODEL_NAME} for {self.ticker}...")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Features: {X_train.shape[1] if len(X_train.shape) > 1 else 1}")
        
        self.model.fit(X_train, y_train)
        
        self.is_trained = True
        
        # Feature importance
        if hasattr(X_train, 'shape') and len(X_train.shape) > 1:
            self.feature_importance = self.model.feature_importances_
        
        return {
            'n_estimators': self.model.n_estimators,
            'feature_importance': list(self.model.feature_importances_) if hasattr(self.model, 'feature_importances_') else None
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
        feature_cols = ['Open', 'High', 'Low', 'Volume', 'Price_Diff', 'Avg_Price', 'Volume_Ratio']
        # Ensure columns exist in the dataframe
        feature_cols = [c for c in feature_cols if c in preprocessor.df.columns]
        
        # Get the last row of data (most recent day)
        X_input = preprocessor.df.iloc[[-1]][feature_cols].values
        
        # Predict
        prediction = self.model.predict(X_input)[0]
        
        return float(prediction)

    def plot_feature_importance(
        self, 
        feature_names: list = None,
        save: bool = True
    ) -> Optional[Path]:
        """Vẽ biểu đồ feature importance."""
        import matplotlib.pyplot as plt
        from datetime import datetime
        
        if not hasattr(self, 'feature_importance'):
            raise ValueError("Model chưa được train.")
        
        importance = self.feature_importance
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importance))]
        
        # Sort by importance
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title(f'{self.ticker} - {self.MODEL_NAME} Feature Importance')
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.tight_layout()
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.result_dir / f"{self.ticker}_{self.MODEL_NAME}_importance_{timestamp}.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            return filepath
        else:
            plt.show()
            return None

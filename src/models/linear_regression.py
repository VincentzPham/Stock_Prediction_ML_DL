"""
Linear Regression Model
Multiple Linear Regression for stock prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import ML_CONFIG

from .base import BaseModel


class LinearRegressionModel(BaseModel):
    """
    Multiple Linear Regression model cho stock price prediction.
    """
    
    MODEL_NAME = "Multiple Linear Regression"
    MODEL_TYPE = "ml"
    
    def __init__(self, ticker: str):
        super().__init__(ticker)
        self.config = ML_CONFIG.get('Multiple Linear Regression', {})
        
    def build(
        self,
        fit_intercept: bool = None,
        **kwargs
    ) -> None:
        """
        Xây dựng Linear Regression model.
        """
        from sklearn.linear_model import LinearRegression
        
        self.model = LinearRegression(
            fit_intercept=fit_intercept if fit_intercept is not None else self.config.get('fit_intercept', True)
        )
        
        print(f"Linear Regression model built")
        
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        X_val: np.ndarray = None, 
        y_val: np.ndarray = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train Linear Regression model.
        """
        if self.model is None:
            self.build()
        
        print(f"\nTraining {self.MODEL_NAME} for {self.ticker}...")
        print(f"  Training samples: {len(X_train)}")
        
        self.model.fit(X_train, y_train)
        
        self.is_trained = True
        
        # Store coefficients
        self.coefficients = self.model.coef_
        self.intercept = self.model.intercept_
        
        return {
            'coefficients': list(self.coefficients),
            'intercept': float(self.intercept),
            'r2_train': self.model.score(X_train, y_train)
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Dự đoán giá stock.
        """
        if self.model is None:
            raise ValueError("Model chưa được build/train.")
        
        predictions = self.model.predict(X)
        return predictions
    
    def get_equation(self, feature_names: list = None) -> str:
        """Trả về phương trình regression."""
        if not hasattr(self, 'coefficients'):
            raise ValueError("Model chưa được train.")
        
        if feature_names is None:
            feature_names = [f'x{i}' for i in range(len(self.coefficients))]
        
        terms = []
        for name, coef in zip(feature_names, self.coefficients):
            if coef >= 0:
                terms.append(f"+ {coef:.4f}*{name}")
            else:
                terms.append(f"- {abs(coef):.4f}*{name}")
        
        equation = f"y = {self.intercept:.4f} " + " ".join(terms)
        return equation

"""
Base Model Class
Abstract base class cho tất cả models.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
import json
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import MODELS_DIR, RESULTS_DIR, SAVE_PLOTS, SAVE_METRICS


class BaseModel(ABC):
    """
    Abstract base class cho tất cả stock prediction models.
    """
    
    MODEL_NAME: str = "BaseModel"
    MODEL_TYPE: str = "base"  # 'deep_learning', 'time_series', 'ml'
    
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.model = None
        self.is_trained = False
        self.history = None
        self.metrics: Dict[str, float] = {}
        self.predictions: Optional[np.ndarray] = None
        self.actuals: Optional[np.ndarray] = None
        
        # Paths
        self.model_dir = MODELS_DIR / ticker / self.MODEL_NAME
        self.result_dir = RESULTS_DIR / ticker / self.MODEL_NAME
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        
    @abstractmethod
    def build(self, **kwargs) -> None:
        """Xây dựng model architecture."""
        pass
    
    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs) -> Dict[str, Any]:
        """
        Train model.
        
        Returns:
            Dict chứa training history/info.
        """
        pass
    
    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """
        Dự đoán.
        
        Returns:
            Array predictions.
        """
        pass
    
    def predict_next(self, preprocessor, horizon: int = 1) -> float:
        """
        Dự đoán giá trị tương lai (horizon days ahead).
        Default implementation raises NotImplementedError.
        Các class con cần override method này.
        
        Args:
            preprocessor: DataPreprocessor object đã load data.
            horizon: Số ngày dự đoán.
            
        Returns:
            Giá trị dự đoán (float).
        """
        raise NotImplementedError(f"Model {self.MODEL_NAME} chưa hỗ trợ predict_next.")

    def evaluate(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Tính các metrics đánh giá.
        
        Args:
            y_true: Giá trị thực
            y_pred: Giá trị dự đoán
            
        Returns:
            Dict chứa các metrics.
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # Mean Absolute Error
        mae = np.mean(np.abs(y_true - y_pred))
        
        # Mean Squared Error
        mse = np.mean((y_true - y_pred) ** 2)
        
        # Root Mean Squared Error
        rmse = np.sqrt(mse)
        
        # Mean Absolute Percentage Error
        # Tránh chia cho 0
        non_zero_mask = y_true != 0
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        self.metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2
        }
        
        return self.metrics
    
    def save_model(self, filename: Optional[str] = None) -> Path:
        """
        Lưu model.
        
        Args:
            filename: Tên file. Nếu None, tự động tạo.
            
        Returns:
            Path đến file đã lưu.
        """
        if self.model is None:
            raise ValueError("Model chưa được train.")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if filename is None:
            filename = f"{self.ticker}_{self.MODEL_NAME}_{timestamp}"
        
        # Xác định extension dựa trên loại model
        if self.MODEL_TYPE == 'deep_learning':
            filepath = self.model_dir / f"{filename}.keras"
            self.model.save(filepath)
        else:
            filepath = self.model_dir / f"{filename}.pkl"
            joblib.dump(self.model, filepath)
        
        print(f"Model saved to {filepath}")
        return filepath
    
    def load_model(self, filepath: Path) -> None:
        """Load model từ file."""
        if filepath.suffix == '.keras':
            from tensorflow.keras.models import load_model
            self.model = load_model(filepath)
        else:
            self.model = joblib.load(filepath)
        
        self.is_trained = True
        print(f"Model loaded from {filepath}")
    
    def save_results(
        self, 
        dates: pd.DatetimeIndex,
        actuals: np.ndarray,
        predictions: np.ndarray,
        filename: Optional[str] = None
    ) -> Path:
        """
        Lưu kết quả dự đoán và metrics.
        
        Returns:
            Path đến file CSV.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if filename is None:
            filename = f"{self.ticker}_{self.MODEL_NAME}_results_{timestamp}"
        
        # Tạo DataFrame kết quả
        results_df = pd.DataFrame({
            'Date': dates[-len(predictions):],
            'Actual': np.array(actuals).flatten()[-len(predictions):],
            'Predicted': np.array(predictions).flatten()
        })
        
        # Thêm metrics
        for key, value in self.metrics.items():
            results_df[key] = value
        
        filepath = self.result_dir / f"{filename}.csv"
        results_df.to_csv(filepath, index=False)
        
        print(f"Results saved to {filepath}")
        return filepath
    
    def save_metrics(self, filename: Optional[str] = None) -> Path:
        """Lưu metrics riêng."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if filename is None:
            filename = f"{self.ticker}_{self.MODEL_NAME}_metrics_{timestamp}"
        
        filepath = self.result_dir / f"{filename}.json"
        
        with open(filepath, 'w') as f:
            json.dump({
                'ticker': self.ticker,
                'model': self.MODEL_NAME,
                'timestamp': timestamp,
                'metrics': self.metrics
            }, f, indent=2)
        
        return filepath
    
    def plot_predictions(
        self,
        dates: pd.DatetimeIndex,
        actuals: np.ndarray,
        predictions: np.ndarray,
        title: Optional[str] = None,
        save: bool = SAVE_PLOTS
    ) -> Optional[Path]:
        """
        Vẽ biểu đồ so sánh actual vs predicted.
        """
        plt.figure(figsize=(14, 6))
        
        dates = dates[-len(predictions):]
        actuals = np.array(actuals).flatten()[-len(predictions):]
        predictions = np.array(predictions).flatten()
        
        plt.plot(dates, actuals, label='Actual', color='blue', alpha=0.7)
        plt.plot(dates, predictions, label='Predicted', color='red', alpha=0.7)
        
        if title is None:
            title = f'{self.ticker} - {self.MODEL_NAME} Prediction'
        
        plt.title(title, fontsize=14)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.result_dir / f"{self.ticker}_{self.MODEL_NAME}_prediction_{timestamp}.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            return filepath
        else:
            plt.show()
            return None
    
    def plot_residuals(
        self,
        actuals: np.ndarray,
        predictions: np.ndarray,
        save: bool = SAVE_PLOTS
    ) -> Optional[Path]:
        """Vẽ biểu đồ residuals."""
        residuals = np.array(actuals).flatten() - np.array(predictions).flatten()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Residual plot
        axes[0].scatter(predictions, residuals, alpha=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs Predicted')
        
        # Histogram
        axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Residual Value')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Residual Distribution')
        
        plt.suptitle(f'{self.ticker} - {self.MODEL_NAME} Residual Analysis', fontsize=14)
        plt.tight_layout()
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.result_dir / f"{self.ticker}_{self.MODEL_NAME}_residuals_{timestamp}.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            return filepath
        else:
            plt.show()
            return None
    
    def summary(self) -> Dict[str, Any]:
        """Trả về summary của model."""
        return {
            'ticker': self.ticker,
            'model_name': self.MODEL_NAME,
            'model_type': self.MODEL_TYPE,
            'is_trained': self.is_trained,
            'metrics': self.metrics
        }
    
    def __repr__(self) -> str:
        return f"{self.MODEL_NAME}(ticker='{self.ticker}', trained={self.is_trained})"

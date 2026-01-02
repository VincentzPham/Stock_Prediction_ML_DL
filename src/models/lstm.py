"""
LSTM Model
Long Short-Term Memory neural network for stock prediction.
"""

import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import DL_CONFIG, TIME_STEP, TARGET_COLUMN

from .base import BaseModel


class LSTMModel(BaseModel):
    """
    LSTM model cho stock price prediction.
    """
    
    MODEL_NAME = "LSTM"
    MODEL_TYPE = "deep_learning"
    
    def __init__(self, ticker: str):
        super().__init__(ticker)
        self.config = DL_CONFIG.get('LSTM', {})
        self.time_step = TIME_STEP
        
    def build(
        self, 
        input_shape: tuple = None,
        units: list = None,
        dropout: float = None,
        **kwargs
    ) -> None:
        """
        Xây dựng LSTM model architecture.
        
        Args:
            input_shape: Shape của input (time_steps, features)
            units: List số units cho mỗi LSTM layer
            dropout: Dropout rate
        """
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        
        if input_shape is None:
            input_shape = (self.time_step, 1)
        
        units = units or self.config.get('units', [50, 50])
        dropout = dropout or self.config.get('dropout', 0.2)
        
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            units=units[0], 
            return_sequences=True if len(units) > 1 else False,
            input_shape=input_shape
        ))
        model.add(Dropout(dropout))
        
        # Additional LSTM layers
        for i, unit in enumerate(units[1:], 1):
            return_seq = i < len(units) - 1
            model.add(LSTM(units=unit, return_sequences=return_seq))
            model.add(Dropout(dropout))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile
        model.compile(
            optimizer=self.config.get('optimizer', 'adam'),
            loss=self.config.get('loss', 'mean_squared_error')
        )
        
        self.model = model
        print(f"Built {self.MODEL_NAME} model:")
        model.summary()
        
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        X_val: np.ndarray = None, 
        y_val: np.ndarray = None,
        epochs: int = None,
        batch_size: int = None,
        patience: int = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train LSTM model.
        """
        from tensorflow.keras.callbacks import EarlyStopping
        
        if self.model is None:
            self.build(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        epochs = epochs or self.config.get('epochs', 50)
        batch_size = batch_size or self.config.get('batch_size', 32)
        patience = patience or self.config.get('patience', 10)
        
        callbacks = []
        if patience > 0:
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stopping)
        
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        print(f"\nTraining {self.MODEL_NAME} for {self.ticker}...")
        print(f"  Epochs: {epochs}, Batch Size: {batch_size}")
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        self.history = history.history
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Dự đoán giá stock.
        """
        if self.model is None:
            raise ValueError("Model chưa được build/train.")
        
        predictions = self.model.predict(X, verbose=0)
        return predictions
    
    def predict_next(self, preprocessor, horizon: int = 1) -> float:
        """
        Dự đoán giá trị tương lai.
        """
        if self.model is None:
            raise ValueError("Model chưa được load/train.")
            
        # Prepare scaler and data
        # We call prepare_lstm_data to ensure the scaler is fitted on current data
        _, _, _, _, scaler = preprocessor.prepare_lstm_data(time_step=self.time_step, horizon=horizon)
        
        # Get the last sequence of data
        # We need the last TIME_STEP values from the SCALED data
        data = preprocessor.df[TARGET_COLUMN].values.reshape(-1, 1)
        scaled_data = scaler.transform(data)
        
        # Input shape: (1, time_step, 1)
        last_sequence = scaled_data[-self.time_step:]
        X_input = last_sequence.reshape(1, self.time_step, 1)
        
        # Predict
        pred_scaled = self.model.predict(X_input, verbose=0)
        prediction = scaler.inverse_transform(pred_scaled)[0][0]
        
        return float(prediction)

    def plot_training_history(self, save: bool = True) -> Optional[Path]:
        """Vẽ training history."""
        import matplotlib.pyplot as plt
        from datetime import datetime
        
        if self.history is None:
            raise ValueError("Model chưa được train.")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        axes[0].plot(self.history['loss'], label='Train Loss')
        if 'val_loss' in self.history:
            axes[0].plot(self.history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss zoomed (last 50%)
        n = len(self.history['loss'])
        start = n // 2
        axes[1].plot(range(start, n), self.history['loss'][start:], label='Train Loss')
        if 'val_loss' in self.history:
            axes[1].plot(range(start, n), self.history['val_loss'][start:], label='Validation Loss')
        axes[1].set_title('Model Loss (Last 50%)')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(f'{self.ticker} - {self.MODEL_NAME} Training History', fontsize=14)
        plt.tight_layout()
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.result_dir / f"{self.ticker}_{self.MODEL_NAME}_history_{timestamp}.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            return filepath
        else:
            plt.show()
            return None

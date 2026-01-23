"""
BiLSTM Model
Bidirectional LSTM neural network for stock prediction.
"""

import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import DL_CONFIG, TIME_STEP

from .base import BaseModel


class BiLSTMModel(BaseModel):
    """
    Bidirectional LSTM model cho stock price prediction.
    """
    
    MODEL_NAME = "BiLSTM"
    MODEL_TYPE = "deep_learning"
    
    def __init__(self, ticker: str):
        super().__init__(ticker)
        self.config = DL_CONFIG.get('BiLSTM', {})
        self.time_step = TIME_STEP
        
    def build(
        self, 
        input_shape: tuple = None,
        units: list = None,
        dropout: float = None,
        **kwargs
    ) -> None:
        """
        Xây dựng BiLSTM model architecture.
        """
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
        
        if input_shape is None:
            input_shape = (self.time_step, 1)
        
        units = units or self.config.get('units', [50, 50])
        dropout = dropout or self.config.get('dropout', 0.2)
        
        model = Sequential()
        
        # First Bidirectional LSTM layer
        model.add(Bidirectional(
            LSTM(units=units[0], return_sequences=True if len(units) > 1 else False),
            input_shape=input_shape
        ))
        model.add(Dropout(dropout))
        
        # Additional Bidirectional LSTM layers
        for i, unit in enumerate(units[1:], 1):
            return_seq = i < len(units) - 1
            model.add(Bidirectional(LSTM(units=unit, return_sequences=return_seq)))
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
        Train BiLSTM model.
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
        from config import TARGET_COLUMN
        
        if self.model is None:
            raise ValueError("Model chưa được load/train.")
            
        # Prepare scaler and data
        _, _, _, _, scaler = preprocessor.prepare_lstm_data(time_step=self.time_step, horizon=horizon)
        
        # Get the last sequence of data
        data = preprocessor.df[TARGET_COLUMN].values.reshape(-1, 1)
        scaled_data = scaler.transform(data)
        
        # Input shape: (1, time_step, 1)
        last_sequence = scaled_data[-self.time_step:]
        X_input = last_sequence.reshape(1, self.time_step, 1)
        
        # Predict
        pred_scaled = self.model.predict(X_input, verbose=0)
        prediction = scaler.inverse_transform(pred_scaled)[0][0]
        
        return float(prediction)

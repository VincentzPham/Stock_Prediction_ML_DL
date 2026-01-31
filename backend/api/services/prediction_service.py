"""
Prediction Service.

Handles stock price prediction logic for different model types.
"""

from typing import List
import numpy as np

from backend.config import TARGET_COLUMN


class PredictionService:
    """
    Service class for stock price predictions.
    
    Provides prediction methods for different model types:
    - Deep Learning (LSTM, RNN, etc.)
    - Machine Learning (Random Forest, Decision Tree, etc.)
    - Time Series (ARIMA, Prophet, etc.)
    """
    
    @classmethod
    def predict_multi_step_ml(
        cls,
        model_instance,
        preprocessor,
        num_steps: int
    ) -> List[float]:
        """
        Multi-step prediction for ML models with rolling feature updates.
        
        Uses iterative prediction where each step's prediction is used to
        update features for the next step, simulating realistic multi-day
        forecasting.

        Args:
            model_instance: Trained ML model instance with .model attribute.
            preprocessor: DataPreprocessor with loaded data.
            num_steps: Number of prediction steps.

        Returns:
            List of predicted prices for each step.
        """
        predictions = []
        
        # Feature columns used by ML models
        feature_cols = [
            "Open", "High", "Low", "Volume",
            "Price_Diff", "Avg_Price", "Volume_Ratio",
        ]
        feature_cols = [c for c in feature_cols if c in preprocessor.df.columns]
        
        # Get last row features and price
        current_features = preprocessor.df.iloc[-1][feature_cols].copy()
        last_close = float(preprocessor.df[TARGET_COLUMN].iloc[-1])
        
        # Calculate typical daily volatility for realistic High/Low estimates
        daily_range_pct = (
            (preprocessor.df["High"] - preprocessor.df["Low"])
            .tail(20)
            .mean() / last_close
        )
        
        for step in range(num_steps):
            # Predict using current features
            X_input = current_features.values.reshape(1, -1)
            pred = float(model_instance.model.predict(X_input)[0])
            
            # Constrain prediction to reasonable range (+/-15% from last actual)
            pred = np.clip(pred, last_close * 0.85, last_close * 1.15)
            predictions.append(pred)
            
            # Update features for next step based on prediction
            est_open = pred * (1 + np.random.uniform(-0.005, 0.005))
            est_high = pred * (1 + daily_range_pct * 0.5)
            est_low = pred * (1 - daily_range_pct * 0.5)
            
            if "Open" in feature_cols:
                current_features["Open"] = est_open
            if "High" in feature_cols:
                current_features["High"] = est_high
            if "Low" in feature_cols:
                current_features["Low"] = est_low
            if "Price_Diff" in feature_cols:
                current_features["Price_Diff"] = est_high - est_low
            if "Avg_Price" in feature_cols:
                current_features["Avg_Price"] = (est_high + est_low) / 2
            
            # Update last_close for next iteration's constraint
            last_close = pred
        
        return predictions
    
    @classmethod
    def predict_multi_step_dl(
        cls,
        model,
        preprocessor,
        scaler,
        num_steps: int,
        time_step: int
    ) -> List[float]:
        """
        Multi-step prediction for Deep Learning models with drift correction.
        
        Uses momentum-based anchoring to reduce error accumulation in
        recursive forecasting. Key features:
        1. Calculates recent momentum from historical data
        2. Uses exponential decay for model weight
        3. Anchors predictions to momentum-projected trajectory
        4. Constrains predictions within reasonable range

        Args:
            model: Trained Keras/TensorFlow model.
            preprocessor: DataPreprocessor with loaded data.
            scaler: Fitted MinMaxScaler for the target column.
            num_steps: Number of prediction steps.
            time_step: Lookback window size.

        Returns:
            List of predicted prices for each step.
        """
        predictions = []
        
        # Get initial data
        data = preprocessor.df[TARGET_COLUMN].values.reshape(-1, 1)
        last_actual_price = float(data[-1][0])
        scaled_data = scaler.transform(data)
        current_sequence = scaled_data[-time_step:].flatten().tolist()
        
        # Calculate momentum from recent price history
        avg_daily_return = cls._calculate_momentum(data)
        
        for step in range(num_steps):
            # Prepare input
            X_input = np.array(current_sequence[-time_step:]).reshape(1, time_step, 1)
            
            # Model prediction
            pred_scaled = model.predict(X_input, verbose=0)
            pred_value = pred_scaled[0][0]
            
            # Inverse transform to get actual price
            pred_actual = scaler.inverse_transform([[pred_value]])[0][0]
            
            # Apply drift correction
            corrected_pred = cls._apply_drift_correction(
                pred_actual=pred_actual,
                last_actual_price=last_actual_price,
                avg_daily_return=avg_daily_return,
                step=step
            )
            
            predictions.append(float(corrected_pred))
            
            # Feed corrected prediction back (prevents drift accumulation)
            corrected_scaled = scaler.transform([[corrected_pred]])[0][0]
            current_sequence.append(corrected_scaled)
        
        return predictions
    
    @classmethod
    def _calculate_momentum(cls, data: np.ndarray) -> float:
        """
        Calculate average daily return from recent price history.
        
        Uses weighted average of short-term (5 days) and medium-term (20 days)
        momentum.
        
        Args:
            data: Price data array.
            
        Returns:
            Average daily return, clipped to [-0.015, 0.015].
        """
        recent_prices = data[-30:].flatten()
        
        if len(recent_prices) < 5:
            return 0.0
        
        # Short-term momentum (5 days)
        short_returns = np.diff(recent_prices[-6:]) / recent_prices[-6:-1]
        short_momentum = np.mean(short_returns)
        
        # Medium-term momentum (20 days)
        if len(recent_prices) >= 21:
            med_returns = np.diff(recent_prices[-21:]) / recent_prices[-21:-1]
            med_momentum = np.mean(med_returns)
        else:
            med_momentum = short_momentum
        
        # Weighted average (favor short-term)
        avg_daily_return = 0.6 * short_momentum + 0.4 * med_momentum
        
        # Limit to realistic daily range
        return np.clip(avg_daily_return, -0.015, 0.015)
    
    @classmethod
    def _apply_drift_correction(
        cls,
        pred_actual: float,
        last_actual_price: float,
        avg_daily_return: float,
        step: int
    ) -> float:
        """
        Apply drift correction to model prediction.
        
        Uses exponential decay to reduce model weight over time and
        anchors predictions to momentum-based trajectory.
        
        Args:
            pred_actual: Raw model prediction.
            last_actual_price: Last known actual price.
            avg_daily_return: Calculated momentum.
            step: Current prediction step (0-indexed).
            
        Returns:
            Corrected prediction.
        """
        # Model weight decreases exponentially as steps increase
        # Starts at 0.7, decays to ~0.26 at step 30
        model_weight = 0.7 * np.exp(-step * 0.03)
        anchor_weight = 1 - model_weight
        
        # Momentum-based anchor price
        anchor_price = last_actual_price * (1 + avg_daily_return) ** (step + 1)
        
        # Blend model prediction with anchor
        corrected_pred = pred_actual * model_weight + anchor_price * anchor_weight
        
        # Constrain to +/-15% from last actual price
        lower_bound = last_actual_price * 0.85
        upper_bound = last_actual_price * 1.15
        
        return np.clip(corrected_pred, lower_bound, upper_bound)

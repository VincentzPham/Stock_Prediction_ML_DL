"""
Prediction Routes.

Endpoints for stock price prediction operations.
"""

import pandas as pd
from fastapi import APIRouter, HTTPException

from backend.config import TICKERS, MODEL_NAMES, TARGET_COLUMN, TIME_STEP
from backend.data.preprocessor import DataPreprocessor
from backend.training.trainer import MODEL_REGISTRY
from backend.api.schemas import PredictRequest, PredictResponse, PredictionDay
from backend.api.services import PredictionService, MarketService


router = APIRouter(prefix="/predictions", tags=["Predictions"])


@router.post("", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """
    Predict stock prices for multiple trading days ahead.
    
    Uses different prediction strategies based on model type:
    - Deep Learning: Iterative multi-step with drift correction
    - Machine Learning: Rolling feature update
    - Time Series: Native multi-step forecasting
    
    Args:
        request: PredictRequest containing ticker, model, and horizon.
    
    Returns:
        PredictResponse containing predictions for each trading day.
        
    Raises:
        HTTPException: 400 if inputs invalid, 404 if model not found, 500 on error.
    """
    # Validate inputs
    if request.ticker not in TICKERS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid ticker. Available: {TICKERS}"
        )
    
    if request.model not in MODEL_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model. Available: {MODEL_NAMES}"
        )
    
    try:
        # Load and prepare data
        preprocessor = DataPreprocessor(request.ticker)
        preprocessor.load_data()
        preprocessor.add_features()
        
        last_date = preprocessor.df.index[-1]
        last_price = float(preprocessor.df[TARGET_COLUMN].iloc[-1])
        
        # Load model
        model_instance = _load_model(request.ticker, request.model)
        model_path = model_instance._loaded_model_path
        
        # Get trading days for predictions
        trading_days = MarketService.get_trading_days(
            last_date, request.horizon, request.ticker
        )
        
        # Generate predictions based on model type
        predictions_list = _generate_predictions(
            model_instance=model_instance,
            preprocessor=preprocessor,
            trading_days=trading_days,
            model_name=request.model,
            horizon=request.horizon
        )
        
        return PredictResponse(
            ticker=request.ticker,
            model=request.model,
            last_actual_date=str(last_date.date()),
            last_actual_price=round(last_price, 2),
            horizon=request.horizon,
            predictions=predictions_list,
            model_path=str(model_path),
        )
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _load_model(ticker: str, model_name: str):
    """
    Load a trained model from disk.
    
    Args:
        ticker: Stock ticker symbol.
        model_name: Name of the model.
        
    Returns:
        Model instance with loaded weights.
        
    Raises:
        FileNotFoundError: If no saved model exists.
    """
    ModelClass = MODEL_REGISTRY[model_name]
    model_instance = ModelClass(ticker)
    
    # Find latest model file
    files = list(model_instance.model_dir.glob(f"*{model_name}*"))
    valid_exts = [".keras", ".pkl"]
    files = [f for f in files if f.suffix in valid_exts]
    
    if not files:
        raise FileNotFoundError(
            f"No saved model found for {ticker} - {model_name}"
        )
    
    model_path = max(files, key=lambda x: x.stat().st_mtime)
    model_instance.load_model(model_path)
    
    # Store path for response
    model_instance._loaded_model_path = model_path
    
    return model_instance


def _generate_predictions(
    model_instance,
    preprocessor,
    trading_days,
    model_name: str,
    horizon: int
):
    """
    Generate predictions based on model type.
    
    Args:
        model_instance: Loaded model instance.
        preprocessor: DataPreprocessor with loaded data.
        trading_days: List of trading day timestamps.
        model_name: Name of the model.
        horizon: Number of days to predict.
        
    Returns:
        List of PredictionDay objects.
    """
    model_type = model_instance.MODEL_TYPE
    predictions_list = []
    
    if model_type == "deep_learning":
        predictions_list = _predict_deep_learning(
            model_instance, preprocessor, trading_days, horizon
        )
    
    elif model_type == "time_series":
        predictions_list = _predict_time_series(
            model_instance, preprocessor, trading_days, model_name, horizon
        )
    
    elif model_type == "ml":
        predictions_list = _predict_ml(
            model_instance, preprocessor, trading_days, horizon
        )
    
    else:
        # Fallback for unknown model types
        predictions_list = _predict_fallback(
            model_instance, preprocessor, trading_days
        )
    
    return predictions_list


def _predict_deep_learning(model_instance, preprocessor, trading_days, horizon):
    """Generate predictions using Deep Learning models."""
    time_step = getattr(model_instance, "time_step", TIME_STEP)
    _, _, _, _, scaler = preprocessor.prepare_lstm_data(time_step=time_step)
    
    pred_values = PredictionService.predict_multi_step_dl(
        model_instance.model, preprocessor, scaler, horizon, time_step
    )
    
    predictions_list = []
    for i, (date, pred_price) in enumerate(zip(trading_days, pred_values)):
        predictions_list.append(
            PredictionDay(
                day=i + 1,
                date=str(date.date()),
                predicted_price=round(pred_price, 2),
            )
        )
    
    return predictions_list


def _predict_time_series(model_instance, preprocessor, trading_days, model_name, horizon):
    """Generate predictions using Time Series models."""
    import numpy as np
    
    predictions_list = []
    
    if hasattr(model_instance, "model") and model_instance.model is not None:
        try:
            # Prophet needs special handling with specific future dates
            if model_name == "Prophet":
                future_df = pd.DataFrame({
                    "ds": [d.to_pydatetime() for d in trading_days]
                })
                forecast = model_instance.model.predict(future_df)
                forecast_values = forecast["yhat"].values
            else:
                # ARIMA, SARIMA, Exponential Smoothing
                forecast = model_instance.model.forecast(steps=horizon)
                forecast_values = np.array(forecast).flatten()
            
            # Handle log transform for SARIMA
            if hasattr(model_instance, "use_log") and model_instance.use_log:
                forecast_values = np.exp(forecast_values)
            
            for i, (date, pred_price) in enumerate(zip(trading_days, forecast_values)):
                predictions_list.append(
                    PredictionDay(
                        day=i + 1,
                        date=str(date.date()),
                        predicted_price=round(float(pred_price), 2),
                    )
                )
        
        except Exception:
            # Fallback to single-step prediction
            pred = model_instance.predict_next(preprocessor, horizon=1)
            for i, date in enumerate(trading_days):
                predictions_list.append(
                    PredictionDay(
                        day=i + 1,
                        date=str(date.date()),
                        predicted_price=round(pred, 2),
                    )
                )
    
    return predictions_list


def _predict_ml(model_instance, preprocessor, trading_days, horizon):
    """Generate predictions using Machine Learning models."""
    pred_values = PredictionService.predict_multi_step_ml(
        model_instance, preprocessor, horizon
    )
    
    predictions_list = []
    for i, (date, pred_price) in enumerate(zip(trading_days, pred_values)):
        predictions_list.append(
            PredictionDay(
                day=i + 1,
                date=str(date.date()),
                predicted_price=round(pred_price, 2),
            )
        )
    
    return predictions_list


def _predict_fallback(model_instance, preprocessor, trading_days):
    """Fallback prediction for unknown model types."""
    pred = model_instance.predict_next(preprocessor, horizon=1)
    
    predictions_list = []
    for i, date in enumerate(trading_days):
        predictions_list.append(
            PredictionDay(
                day=i + 1,
                date=str(date.date()),
                predicted_price=round(pred, 2),
            )
        )
    
    return predictions_list

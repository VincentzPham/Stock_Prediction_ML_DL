from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
import pandas as pd
import numpy as np

from backend.training.trainer import MODEL_REGISTRY
from backend.data.preprocessor import DataPreprocessor
from backend.config import TICKERS, MODEL_NAMES, TARGET_COLUMN, RESULTS_DIR, TIME_STEP

app = FastAPI(
    title="Stock Prediction API",
    description="API for predicting stock prices using trained models",
    version="2.0.0",
)


class PredictRequest(BaseModel):
    ticker: str
    model: str
    horizon: int = 7


class PredictionDay(BaseModel):
    day: int
    date: str
    predicted_price: float


class PredictResponse(BaseModel):
    ticker: str
    model: str
    last_actual_date: str
    last_actual_price: float
    horizon: int
    predictions: List[PredictionDay]
    currency: str = "USD"
    model_path: str


class HistoricalDataPoint(BaseModel):
    date: str
    actual: float


class HistoricalResponse(BaseModel):
    ticker: str
    data: List[HistoricalDataPoint]


class MetricsResponse(BaseModel):
    ticker: str
    model: str
    mse: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    mape: Optional[float] = None
    r2: Optional[float] = None


def get_trading_days(
    start_date: pd.Timestamp, num_days: int, ticker: str = "AAPL"
) -> List[pd.Timestamp]:
    """
    Get a list of actual trading days using market calendars.
    Excludes weekends AND market holidays (MLK Day, Presidents Day, etc.)

    Args:
        start_date: Starting date (exclusive - will start from next trading day)
        num_days: Number of trading days to return
        ticker: Ticker symbol to determine exchange (BTC-USD uses 24/7)

    Returns:
        List of trading day timestamps
    """
    import pandas_market_calendars as mcal

    # Crypto trades 24/7, use simple business days
    if ticker in ["BTC-USD", "ETH-USD"]:
        business_days = []
        current_date = start_date
        while len(business_days) < num_days:
            current_date += pd.Timedelta(days=1)
            business_days.append(current_date)
        return business_days

    # For stocks, use NYSE calendar
    try:
        calendar = mcal.get_calendar("NYSE")

        # Get schedule for next 90 days (buffer for holidays)
        end_date = start_date + pd.Timedelta(days=num_days * 2 + 30)
        schedule = calendar.schedule(
            start_date=start_date + pd.Timedelta(days=1), end_date=end_date
        )

        if len(schedule) >= num_days:
            trading_days = [pd.Timestamp(d) for d in schedule.index[:num_days]]
            return trading_days
    except Exception:
        pass

    # Fallback to simple business days (weekdays only)
    business_days = []
    current_date = start_date
    while len(business_days) < num_days:
        current_date += pd.Timedelta(days=1)
        if current_date.weekday() < 5:  # Monday=0, Friday=4
            business_days.append(current_date)

    return business_days


def predict_multi_step_ml(model_instance, preprocessor, num_steps: int) -> List[float]:
    """
    Multi-step prediction for ML models (Random Forest, Decision Tree, Linear Regression)
    with rolling feature update for more realistic multi-day forecasting.

    Instead of using random variation, this updates features based on predicted prices
    to simulate how the model would predict each subsequent day.
    """
    predictions = []

    # Feature columns used by ML models
    feature_cols = [
        "Open",
        "High",
        "Low",
        "Volume",
        "Price_Diff",
        "Avg_Price",
        "Volume_Ratio",
    ]
    feature_cols = [c for c in feature_cols if c in preprocessor.df.columns]

    # Get last row features and price
    current_features = preprocessor.df.iloc[-1][feature_cols].copy()
    last_close = float(preprocessor.df[TARGET_COLUMN].iloc[-1])

    # Calculate typical daily volatility for realistic High/Low estimates
    daily_range_pct = (preprocessor.df["High"] - preprocessor.df["Low"]).tail(
        20
    ).mean() / last_close

    for step in range(num_steps):
        # Predict using current features
        X_input = current_features.values.reshape(1, -1)
        pred = float(model_instance.model.predict(X_input)[0])

        # Constrain prediction to reasonable range (±15% from last actual)
        pred = np.clip(pred, last_close * 0.85, last_close * 1.15)
        predictions.append(pred)

        # Update features for next step based on prediction
        # Estimate next day's OHLC based on predicted close
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


def predict_multi_step_dl(
    model, preprocessor, scaler, num_steps: int, time_step: int
) -> List[float]:
    """
    Perform iterative multi-step prediction for Deep Learning models with enhanced drift correction.
    Uses momentum-based anchoring to reduce error accumulation in recursive forecasting.

    Key improvements:
    1. Calculates recent momentum from historical data
    2. Uses exponential decay for model weight (reduces reliance on drifting predictions)
    3. Anchors predictions to momentum-projected trajectory
    4. Constrains predictions within reasonable range (±15% from last actual)
    """
    predictions = []

    # Get initial data
    data = preprocessor.df[TARGET_COLUMN].values.reshape(-1, 1)
    last_actual_price = float(data[-1][0])
    scaled_data = scaler.transform(data)
    current_sequence = scaled_data[-time_step:].flatten().tolist()

    # Calculate recent momentum from multiple timeframes
    recent_prices = data[-30:].flatten()
    if len(recent_prices) >= 5:
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
        avg_daily_return = np.clip(avg_daily_return, -0.015, 0.015)
    else:
        avg_daily_return = 0.0

    for step in range(num_steps):
        # Prepare input
        X_input = np.array(current_sequence[-time_step:]).reshape(1, time_step, 1)

        # Model prediction
        pred_scaled = model.predict(X_input, verbose=0)
        pred_value = pred_scaled[0][0]

        # Inverse transform to get actual price
        pred_actual = scaler.inverse_transform([[pred_value]])[0][0]

        # Enhanced drift correction using exponential decay
        # Model weight decreases exponentially as steps increase
        model_weight = 0.7 * np.exp(
            -step * 0.03
        )  # Starts at 0.7, decays to ~0.26 at step 30
        anchor_weight = 1 - model_weight

        # Momentum-based anchor price
        anchor_price = last_actual_price * (1 + avg_daily_return) ** (step + 1)

        # Blend model prediction with anchor
        corrected_pred = pred_actual * model_weight + anchor_price * anchor_weight

        # Constrain to tighter range (±15% from last actual price)
        lower_bound = last_actual_price * 0.85
        upper_bound = last_actual_price * 1.15
        corrected_pred = np.clip(corrected_pred, lower_bound, upper_bound)

        predictions.append(float(corrected_pred))

        # Feed corrected prediction back (prevents drift accumulation)
        corrected_scaled = scaler.transform([[corrected_pred]])[0][0]
        current_sequence.append(corrected_scaled)

    return predictions


@app.get("/")
def read_root():
    return {"message": "Welcome to Stock Prediction API v2.0"}


@app.get("/tickers")
def get_tickers():
    """Get list of available tickers."""
    return {"tickers": TICKERS}


@app.get("/models")
def get_models():
    """Get list of available models."""
    return {"models": MODEL_NAMES}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Predict stock prices for multiple business days ahead.
    Uses iterative forecasting for multi-step prediction.
    """
    # Validate inputs
    if request.ticker not in TICKERS:
        raise HTTPException(
            status_code=400, detail=f"Invalid ticker. Available: {TICKERS}"
        )

    if request.model not in MODEL_NAMES:
        raise HTTPException(
            status_code=400, detail=f"Invalid model. Available: {MODEL_NAMES}"
        )

    try:
        # Load data
        preprocessor = DataPreprocessor(request.ticker)
        preprocessor.load_data()
        preprocessor.add_features()

        last_date = preprocessor.df.index[-1]
        last_price = float(preprocessor.df[TARGET_COLUMN].iloc[-1])

        # Load model
        ModelClass = MODEL_REGISTRY[request.model]
        model_instance = ModelClass(request.ticker)

        # Find latest model file
        files = list(model_instance.model_dir.glob(f"*{request.model}*"))
        valid_exts = [".keras", ".pkl"]
        files = [f for f in files if f.suffix in valid_exts]

        if not files:
            raise FileNotFoundError(
                f"No saved model found for {request.ticker} - {request.model}"
            )

        model_path = max(files, key=lambda x: x.stat().st_mtime)
        model_instance.load_model(model_path)

        # Get proper trading days for predictions (excludes holidays)
        trading_days = get_trading_days(last_date, request.horizon, request.ticker)

        # Determine model type and predict accordingly
        model_type = model_instance.MODEL_TYPE
        predictions_list = []

        if model_type == "deep_learning":
            # Use iterative multi-step prediction for DL models
            time_step = getattr(model_instance, "time_step", TIME_STEP)
            _, _, _, _, scaler = preprocessor.prepare_lstm_data(time_step=time_step)

            pred_values = predict_multi_step_dl(
                model_instance.model, preprocessor, scaler, request.horizon, time_step
            )

            for i, (date, pred_price) in enumerate(zip(trading_days, pred_values)):
                predictions_list.append(
                    PredictionDay(
                        day=i + 1,
                        date=str(date.date()),
                        predicted_price=round(pred_price, 2),
                    )
                )

        elif model_type == "time_series":
            # Time series models can forecast multiple steps directly
            if hasattr(model_instance, "model") and model_instance.model is not None:
                try:
                    # Prophet needs special handling with specific future dates
                    if request.model == "Prophet":
                        future_df = pd.DataFrame(
                            {"ds": [d.to_pydatetime() for d in trading_days]}
                        )
                        forecast = model_instance.model.predict(future_df)
                        forecast_values = forecast["yhat"].values
                    else:
                        # ARIMA, SARIMA, Exponential Smoothing
                        forecast = model_instance.model.forecast(steps=request.horizon)
                        forecast_values = np.array(forecast).flatten()

                    # Handle log transform for SARIMA
                    if hasattr(model_instance, "use_log") and model_instance.use_log:
                        forecast_values = np.exp(forecast_values)

                    for i, (date, pred_price) in enumerate(
                        zip(trading_days, forecast_values)
                    ):
                        predictions_list.append(
                            PredictionDay(
                                day=i + 1,
                                date=str(date.date()),
                                predicted_price=round(float(pred_price), 2),
                            )
                        )
                except Exception:
                    # Fallback
                    pred = model_instance.predict_next(preprocessor, horizon=1)
                    for i, date in enumerate(trading_days):
                        predictions_list.append(
                            PredictionDay(
                                day=i + 1,
                                date=str(date.date()),
                                predicted_price=round(pred, 2),
                            )
                        )

        elif model_type == "ml":
            # ML models: use rolling feature update for multi-step prediction
            pred_values = predict_multi_step_ml(
                model_instance, preprocessor, request.horizon
            )

            for i, (date, pred_price) in enumerate(zip(trading_days, pred_values)):
                predictions_list.append(
                    PredictionDay(
                        day=i + 1,
                        date=str(date.date()),
                        predicted_price=round(pred_price, 2),
                    )
                )

        else:
            # Fallback
            pred = model_instance.predict_next(preprocessor, horizon=1)
            for i, date in enumerate(trading_days):
                predictions_list.append(
                    PredictionDay(
                        day=i + 1, date=str(date.date()), predicted_price=round(pred, 2)
                    )
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


@app.get("/historical/{ticker}", response_model=HistoricalResponse)
def get_historical(ticker: str, days: int = 60):
    """
    Get historical price data for a ticker.
    """
    if ticker not in TICKERS:
        raise HTTPException(
            status_code=400, detail=f"Invalid ticker. Available: {TICKERS}"
        )

    try:
        preprocessor = DataPreprocessor(ticker)
        preprocessor.load_data()

        # Get last N days
        df = preprocessor.df.tail(days)

        data = []
        for idx, row in df.iterrows():
            data.append(
                HistoricalDataPoint(
                    date=str(idx.date()), actual=round(float(row[TARGET_COLUMN]), 2)
                )
            )

        return HistoricalResponse(ticker=ticker, data=data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/{ticker}/{model}", response_model=MetricsResponse)
def get_metrics(ticker: str, model: str):
    """Get model metrics for a specific ticker and model."""
    if ticker not in TICKERS:
        raise HTTPException(
            status_code=400, detail=f"Invalid ticker. Available: {TICKERS}"
        )

    if model not in MODEL_NAMES:
        raise HTTPException(
            status_code=400, detail=f"Invalid model. Available: {MODEL_NAMES}"
        )

    try:
        import json

        result_dir = RESULTS_DIR / ticker / model

        if not result_dir.exists():
            return MetricsResponse(ticker=ticker, model=model)

        metrics_files = list(result_dir.glob("*_metrics_*.json"))

        if not metrics_files:
            return MetricsResponse(ticker=ticker, model=model)

        # Get latest metrics file
        latest_file = max(metrics_files, key=lambda x: x.stat().st_mtime)

        with open(latest_file, "r") as f:
            metrics = json.load(f)

        # Extract from nested 'metrics' key (file format has nested structure)
        metrics_data = metrics.get("metrics", metrics)

        return MetricsResponse(
            ticker=ticker,
            model=model,
            mse=metrics_data.get("MSE"),
            rmse=metrics_data.get("RMSE"),
            mae=metrics_data.get("MAE"),
            mape=metrics_data.get("MAPE"),
            r2=metrics_data.get("R2"),
        )

    except Exception:
        return MetricsResponse(ticker=ticker, model=model)


@app.get("/latest-price/{ticker}")
def get_latest_price(ticker: str):
    """
    Get the latest price for a ticker.
    """
    if ticker not in TICKERS:
        raise HTTPException(
            status_code=400, detail=f"Invalid ticker. Available: {TICKERS}"
        )

    try:
        preprocessor = DataPreprocessor(ticker)
        preprocessor.load_data()

        last_row = preprocessor.df.iloc[-1]
        last_date = preprocessor.df.index[-1]

        return {
            "ticker": ticker,
            "date": str(last_date.date()),
            "close": round(float(last_row[TARGET_COLUMN]), 2),
            "open": round(float(last_row["Open"]), 2),
            "high": round(float(last_row["High"]), 2),
            "low": round(float(last_row["Low"]), 2),
            "volume": int(last_row["Volume"]),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

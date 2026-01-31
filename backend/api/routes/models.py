"""
Model Routes.

Endpoints for model-related operations (list models, get metrics).
"""

import json
from fastapi import APIRouter, HTTPException

from backend.config import TICKERS, MODEL_NAMES, RESULTS_DIR
from backend.api.schemas import ModelsResponse, MetricsResponse


router = APIRouter(prefix="/models", tags=["Models"])


@router.get("", response_model=ModelsResponse)
def get_models() -> ModelsResponse:
    """
    Get list of available prediction models.
    
    Returns:
        ModelsResponse containing list of supported model names.
    """
    return ModelsResponse(models=MODEL_NAMES)


@router.get("/{ticker}/{model}/metrics", response_model=MetricsResponse)
def get_metrics(ticker: str, model: str) -> MetricsResponse:
    """
    Get evaluation metrics for a specific ticker and model.
    
    Retrieves the latest metrics from saved JSON files.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL').
        model: Model name (e.g., 'LSTM').
    
    Returns:
        MetricsResponse containing MAE, MSE, RMSE, MAPE, R2 values.
        Returns empty metrics if no saved metrics found.
        
    Raises:
        HTTPException: 400 if ticker or model is invalid.
    """
    if ticker not in TICKERS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid ticker. Available: {TICKERS}"
        )
    
    if model not in MODEL_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model. Available: {MODEL_NAMES}"
        )
    
    try:
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

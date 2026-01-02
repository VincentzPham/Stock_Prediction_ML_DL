from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from training.trainer import ModelTrainer
from config import TICKERS, MODEL_NAMES

app = FastAPI(
    title="Stock Prediction API",
    description="API for predicting stock prices using trained models",
    version="1.0.0"
)

class PredictRequest(BaseModel):
    ticker: str
    model: str
    horizon: int = 1

class PredictResponse(BaseModel):
    ticker: str
    model: str
    date: str
    horizon: int
    prediction: float
    currency: str = "USD"
    model_path: str

@app.get("/")
def read_root():
    return {"message": "Welcome to Stock Prediction API"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Predict stock price for a given ticker, model, and horizon.
    """
    # Validate inputs
    if request.ticker not in TICKERS:
        raise HTTPException(status_code=400, detail=f"Invalid ticker. Available: {TICKERS}")
    
    if request.model not in MODEL_NAMES:
        raise HTTPException(status_code=400, detail=f"Invalid model. Available: {MODEL_NAMES}")
    
    try:
        trainer = ModelTrainer(verbose=False)
        result = trainer.predict_horizon(
            ticker=request.ticker,
            model_name=request.model,
            horizon=request.horizon
        )
        
        return PredictResponse(
            ticker=result['ticker'],
            model=result['model'],
            date=result['date'],
            horizon=result['horizon'],
            prediction=result['prediction'],
            model_path=result['model_path']
        )
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

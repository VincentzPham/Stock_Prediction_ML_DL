"""
Stock Prediction API Application.

FastAPI application for stock price prediction using various ML/DL models.
This module serves as the main entry point and assembles all route handlers.

Usage:
    uv run uvicorn backend.api.app:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import (
    root_router,
    tickers_router,
    models_router,
    predictions_router,
)


def create_app() -> FastAPI:
    """
    Application factory for creating the FastAPI instance.
    
    This function creates and configures the FastAPI application,
    including middleware setup and route registration.
    
    Returns:
        Configured FastAPI application instance.
    """
    application = FastAPI(
        title="Stock Prediction API",
        description=(
            "RESTful API for predicting stock prices using trained machine learning "
            "and deep learning models. Supports multiple tickers and prediction models."
        ),
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    
    # Configure CORS
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Register routers
    _register_routes(application)
    
    return application


def _register_routes(application: FastAPI) -> None:
    """
    Register all API routes with the application.
    
    Args:
        application: FastAPI application instance.
    """
    # Root routes (/)
    application.include_router(root_router)
    
    # Ticker routes (/tickers)
    application.include_router(tickers_router)
    
    # Model routes (/models)
    application.include_router(models_router)
    
    # Prediction routes (/predictions)
    application.include_router(predictions_router)
    
    # Legacy routes for backward compatibility
    _register_legacy_routes(application)


def _register_legacy_routes(application: FastAPI) -> None:
    """
    Register legacy routes for backward compatibility.
    
    These routes maintain compatibility with older API consumers.
    New clients should use the routes under /tickers, /models, /predictions.
    
    Args:
        application: FastAPI application instance.
    """
    from backend.api.routes.tickers import get_tickers, get_historical, get_latest_price
    from backend.api.routes.models import get_models, get_metrics
    from backend.api.routes.predictions import predict
    
    # Legacy: GET /tickers -> GET /tickers
    application.get("/tickers", tags=["Legacy"])(
        lambda: get_tickers()
    )
    
    # Legacy: GET /models -> GET /models
    application.get("/models", tags=["Legacy"])(
        lambda: get_models()
    )
    
    # Legacy: POST /predict -> POST /predictions
    application.post("/predict", tags=["Legacy"])(predict)
    
    # Legacy: GET /historical/{ticker} -> GET /tickers/{ticker}/historical
    application.get("/historical/{ticker}", tags=["Legacy"])(
        lambda ticker, days=60: get_historical(ticker, days)
    )
    
    # Legacy: GET /metrics/{ticker}/{model} -> GET /models/{ticker}/{model}/metrics
    application.get("/metrics/{ticker}/{model}", tags=["Legacy"])(
        lambda ticker, model: get_metrics(ticker, model)
    )
    
    # Legacy: GET /latest-price/{ticker} -> GET /tickers/{ticker}/latest
    application.get("/latest-price/{ticker}", tags=["Legacy"])(
        lambda ticker: get_latest_price(ticker)
    )


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

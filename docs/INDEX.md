# Stock Price Prediction Project

## Tổng Quan
Dự án dự đoán giá cổ phiếu/crypto sử dụng nhiều mô hình Machine Learning và Deep Learning.
Hệ thống bao gồm Training Pipeline, FastAPI Backend, và Streamlit UI với Model Comparison Dashboard.

## Tech Stack
- **Python**: 3.12 (Recommended for TensorFlow compatibility)
- **Package Manager**: uv
- **ML/DL**: TensorFlow, scikit-learn, statsmodels, Prophet
- **Backend**: FastAPI, Uvicorn
- **Frontend**: Streamlit (Multi-page App)
- **DevOps**: Docker, GitHub Actions (CI/CD)
- **MLOps**: DVC (Data Version Control), MLflow (Experiment Tracking)
- **Auto-ML**: Optuna (Bayesian Optimization)
- **Sentiment Analysis**: VADER, BeautifulSoup4 (FinViz news)
- **Caching**: Streamlit cache_data, cachetools TTLCache

## Cấu Trúc Thư Mục

```
Stock/
├── backend/                     # Source code chính
│   ├── api/                     # FastAPI Backend (RESTful)
│   │   ├── app.py               # Application factory
│   │   ├── cache.py             # Backend caching (TTLCache)
│   │   ├── routes/              # API route handlers
│   │   │   ├── root.py          # GET /
│   │   │   ├── tickers.py       # /tickers endpoints
│   │   │   ├── models.py        # /models endpoints
│   │   │   ├── predictions.py   # /predictions endpoints
│   │   │   ├── comparison.py    # /compare endpoints
│   │   │   └── sentiment.py     # /sentiment endpoints
│   │   ├── schemas/             # Pydantic models
│   │   │   ├── requests.py      # Request validation
│   │   │   └── responses.py     # Response serialization
│   │   └── services/            # Business logic
│   │       ├── market_service.py      # Trading calendar
│   │       ├── prediction_service.py  # Prediction algorithms
│   │       └── comparison_service.py  # Model comparison
│   ├── config.py                # Cấu hình toàn cục
│   ├── data/                    # Data processing
│   │   ├── preprocessor.py
│   │   └── sentiment_analyzer.py
│   ├── models/                  # Model definitions
│   │   ├── base.py
│   │   ├── deep_learning/       # LSTM, BiLSTM, RNN, ANN, LSTM-GRU
│   │   ├── machine_learning/    # Random Forest, Decision Tree, Linear Regression
│   │   └── time_series/         # ARIMA, SARIMA, Prophet, Exp Smoothing
│   └── training/                # Training logic (with MLflow & Optuna)
│       └── trainer.py
│
├── frontend/                    # Streamlit Frontend (Multi-page)
│   ├── app.py                   # Home page with navigation
│   ├── config.py                # Configuration constants
│   ├── api_client.py            # API communication client
│   ├── styles.py                # CSS styles (animations, responsive)
│   ├── components/              # Reusable UI components
│   │   ├── charts.py            # Plotly chart components
│   │   └── metrics.py           # Metrics display components
│   ├── pages/                   # Multi-page app pages
│   │   ├── 1_Dashboard.py       # Market overview, leaderboard, sentiment
│   │   ├── 2_Model_Comparison.py # Compare models for a ticker
│   │   └── 3_Predictions.py     # Generate price predictions
│   └── utils/                   # Utility modules
│       ├── cache.py             # Frontend caching
│       └── export.py            # CSV/JSON export helpers
│
├── scripts/                     # Scripts
│   └── train_all.py             # Training CLI
│
├── tests/                       # Test suite
│   ├── unit/
│   └── integration/
│
├── Data/                        # CSV Data (Tracked by DVC)
│   └── sentiment/               # Sentiment data per ticker
├── Models/                      # Trained Models (Tracked by DVC)
├── Result/                      # Evaluation Results
├── docs/                        # Documentation
├── .github/                     # CI/CD Workflows
├── .dvc/                        # DVC Config
├── Dockerfile                   # Docker config
└── pyproject.toml               # Dependencies
```

## Danh Sách Models (12 models)

| Model                       | Type              | Library      | Support Tuning |
|-----------------------------|-------------------|--------------|:--------------:|
| LSTM, BiLSTM, LSTM-GRU      | Deep Learning     | tensorflow   |      Yes       |
| RNN, ANN                    | Deep Learning     | tensorflow   |      Yes       |
| ARIMA, SARIMA               | Time Series       | statsmodels  |       No       |
| Prophet                     | Time Series       | prophet      |       No       |
| Exponential Smoothing       | Time Series       | statsmodels  |       No       |
| Random Forest               | ML                | scikit-learn |      Yes       |
| Decision Tree               | ML                | scikit-learn |      Yes       |
| Multiple Linear Regression  | ML                | scikit-learn |       No       |

## Commands

### Setup
```bash
uv sync
uv run dvc pull  # Pull data/models from remote if configured
```

### Hyperparameter Tuning
```bash
# Tune và Train model với Optuna (20 trials)
uv run python scripts/train_all.py -t AAPL -m LSTM --tune --trials 20

# Tune nhanh (5 trials)
uv run python scripts/train_all.py -t AAPL -m "Random Forest" --tune --trials 5
```

### Training
```bash
# Train ML và DL models cho AAPL và MSFT
uv run python scripts/train_all.py --ml-only --dl-only --tickers AAPL,MSFT

# Train tất cả (ML only để nhanh)
uv run python scripts/train_all.py --ml-only
```

### MLflow UI
```bash
uv run mlflow ui
```

### Run App
```bash
# API (Backend)
uv run uvicorn backend.api.app:app --reload

# UI (Frontend)
uv run streamlit run frontend/app.py
```

### Docker
```bash
docker build -t stock-api .
docker run -p 8000:8000 stock-api
```

## Architecture

```
User -> Streamlit UI -> FastAPI -> ModelTrainer -> BaseModel -> [Specific Model]
                                        ^
                                        |
                                  DataPreprocessor
```

## Features
- **Multi-Horizon Forecast**: Dự đoán giá cho 1, 3, 7, 14, 30, 60 ngày tiếp theo.
- **Sentiment Analysis**: Tích hợp tin tức từ FinViz và Google News để tính điểm sentiment.
- **Model Comparison**: Dashboard so sánh hiệu suất tất cả models (MAPE, RMSE, R²).
- **Unified Interface**: Tất cả models đều có method `predict_next(horizon)`.
- **Auto-Update**: API tự động load model mới nhất.
- **Returns-based Training**: DL models có option train trên returns thay vì giá raw.
- **Caching**: Frontend và Backend đều có caching layer (5 min TTL).
- **Export**: Xuất kết quả dự đoán và so sánh dưới dạng CSV/JSON.

## Known Issues
1. **TensorFlow**: Khuyến nghị dùng Python 3.12.
2. **Prophet**: Cần compile Stan models (tốn thời gian lần đầu).
3. **Sentiment Data**: Hiện chỉ có AAPL có sentiment data đầy đủ.

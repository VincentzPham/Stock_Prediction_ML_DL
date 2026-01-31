# Stock Price Prediction Project

## Tổng Quan
Dự án dự đoán giá cổ phiếu/crypto sử dụng nhiều mô hình Machine Learning và Deep Learning.
Hệ thống bao gồm Training Pipeline, FastAPI Backend, và Streamlit UI.

## Tech Stack
- **Python**: 3.12 (Recommended for TensorFlow compatibility)
- **Package Manager**: uv
- **ML/DL**: TensorFlow, scikit-learn, statsmodels, Prophet
- **Backend**: FastAPI, Uvicorn
- **Frontend**: Streamlit
- **DevOps**: Docker, GitHub Actions (CI/CD)
- **MLOps**: DVC (Data Version Control), MLflow (Experiment Tracking)
- **Auto-ML**: Optuna (Bayesian Optimization)
- **Sentiment Analysis**: VADER, BeautifulSoup4 (FinViz news)

## Cấu Trúc Thư Mục

```
Stock/
├── backend/                     # Source code chính
│   ├── api/                     # FastAPI Backend
│   │   └── app.py
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
├── frontend/                    # Streamlit Frontend
│   └── app.py
│
├── scripts/                     # Scripts
│   └── train_all.py             # Training CLI
│
├── tests/                       # Test suite
│   ├── unit/
│   └── integration/
│
├── Data/                        # CSV Data (Tracked by DVC)
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
- **Unified Interface**: Tất cả models đều có method `predict_next(horizon)`.
- **Auto-Update**: API tự động load model mới nhất.

## Known Issues
1. **TensorFlow**: Khuyến nghị dùng Python 3.12.
2. **Prophet**: Cần compile Stan models (tốn thời gian lần đầu).

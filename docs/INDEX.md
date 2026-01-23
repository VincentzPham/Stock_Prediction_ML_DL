# 📊 Stock Price Prediction Project

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

## Cấu Trúc Thư Mục

```
Stock/
├── src/                         # Source code chính
│   ├── api/                     # FastAPI Backend
│   ├── ui/                      # Streamlit Frontend
│   ├── data/                    # Data processing
│   ├── models/                  # Model definitions
│   └── training/                # Training logic (with MLflow & Optuna)
│
├── scripts/                     # Scripts
│   └── train_all.py             # Training CLI
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
| LSTM, BiLSTM, LSTM-GRU      | Deep Learning     | tensorflow   |       ✅       |
| RNN, ANN                    | Deep Learning     | tensorflow   |       ✅       |
| ARIMA, SARIMA               | Time Series       | statsmodels  |       ❌       |
| Prophet                     | Time Series       | prophet      |       ❌       |
| Exponential Smoothing       | Time Series       | statsmodels  |       ❌       |
| Random Forest               | ML                | scikit-learn |       ✅       |
| Decision Tree               | ML                | scikit-learn |       ✅       |
| Multiple Linear Regression  | ML                | scikit-learn |       ❌       |

## Commands

### Setup
```bash
uv sync
uv run dvc pull  # Pull data/models from remote if configured
```

### Hyperparameter Tuning (NEW)
```bash
# Tune và Train model với Optuna (20 trials)
uv run python scripts/train_all.py -t AAPL -m LSTM --tune --trials 20

# Tune nhanh (5 trials)
uv run python scripts/train_all.py -t AAPL -m "Random Forest" --tune --trials 5
```

### Training (Standard)
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
uv run python src/api/app.py

# UI (Frontend)
uv run streamlit run src/ui/app.py
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
- **Multi-Horizon Forecast**: Dự đoán giá cho 1, 3, 7, 30 ngày tiếp theo.
- **Unified Interface**: Tất cả models đều có method `predict_next(horizon)`.
- **Auto-Update**: API tự động load model mới nhất.

## Known Issues
1. **TensorFlow**: Khuyến nghị dùng Python 3.12.
2. **Prophet**: Cần compile Stan models (tốn thời gian lần đầu).

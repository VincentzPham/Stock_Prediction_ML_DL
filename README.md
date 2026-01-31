# Stock Price Prediction Project

## Overview

Dự án dự đoán giá cổ phiếu/crypto sử dụng **12 mô hình** Machine Learning và Deep Learning.  
Hệ thống bao gồm Training Pipeline, FastAPI Backend, và Streamlit UI.

## Tech Stack

- **Python**: 3.12
- **Package Manager**: uv
- **ML/DL**: TensorFlow, scikit-learn, statsmodels, Prophet
- **Backend**: FastAPI, Uvicorn
- **Frontend**: Streamlit
- **Sentiment Analysis**: VADER, BeautifulSoup4 (FinViz + Google News)

## Project Structure

```
Stock/
├── backend/              # Backend source code
│   ├── api/              # FastAPI endpoints
│   ├── config.py         # Global configuration
│   ├── data/             # Data processing
│   ├── models/           # Model definitions
│   │   ├── deep_learning/
│   │   ├── machine_learning/
│   │   └── time_series/
│   └── training/         # Training logic
├── frontend/             # Streamlit UI
├── tests/                # Unit and integration tests
├── scripts/              # Training scripts
├── Data/                 # Raw stock data (CSV)
├── Models/               # Saved trained models
├── Result/               # Evaluation results
└── docs/                 # Documentation
```

---

## Quick Start

### 1. Setup
```bash
# Install dependencies
uv sync
```

### 2. Run Application
```bash
# Start Backend (FastAPI) - port 8000
uv run uvicorn backend.api.app:app --reload --host 0.0.0.0 --port 8000

# Start Frontend (Streamlit) - port 8501
uv run streamlit run frontend/app.py
```

---

## Training Models

### Train All Models (All Tickers)
```bash
# Train all 12 models for all tickers (takes a long time!)
uv run python scripts/train_all.py
```

### Train by Model Type
```bash
# ML Models only (fastest - ~5 min)
uv run python scripts/train_all.py --ml-only

# Deep Learning models only (~30 min per ticker)
uv run python scripts/train_all.py --dl-only

# Time Series models only (~10 min)
uv run python scripts/train_all.py --ts-only
```

### Train Specific Model + Ticker
```bash
# Train LSTM for AAPL
uv run python scripts/train_all.py -t AAPL -m LSTM

# Train Random Forest for NVDA
uv run python scripts/train_all.py -t NVDA -m "Random Forest"

# Train multiple tickers and models
uv run python scripts/train_all.py --tickers AAPL,MSFT,NVDA --models LSTM,ARIMA
```

### With Hyperparameter Tuning (Optuna)
```bash
# Tune LSTM with 20 trials
uv run python scripts/train_all.py -t AAPL -m LSTM --tune --trials 20

# Quick tune with 5 trials
uv run python scripts/train_all.py -t AAPL -m "Random Forest" --tune --trials 5
```

### List Available Options
```bash
# List all available models
uv run python scripts/train_all.py --list-models

# List all available tickers
uv run python scripts/train_all.py --list-tickers
```

---

## Docker

### Build and Run
```bash
# Build images
docker compose build

# Start all services (API + UI)
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down
```

### Access Services
- **API**: http://localhost:8000
- **UI**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs

---

## Models (12 total)

### Deep Learning (5)
| Model     | Description                    |
|-----------|--------------------------------|
| LSTM      | Long Short-Term Memory         |
| BiLSTM    | Bidirectional LSTM             |
| RNN       | Simple Recurrent Neural Network|
| ANN       | Artificial Neural Network      |
| LSTM-GRU  | LSTM-GRU Hybrid                |

### Machine Learning (3)
| Model           | Description           |
|-----------------|-----------------------|
| Random Forest   | Ensemble method       |
| Decision Tree   | Tree-based regressor  |
| Linear Regression | Multiple regression |

### Time Series (4)
| Model               | Description                    |
|---------------------|--------------------------------|
| ARIMA               | AutoRegressive Integrated MA   |
| SARIMA              | Seasonal ARIMA                 |
| Prophet             | Facebook Prophet               |
| Exponential Smoothing | Holt-Winters                 |

---

## Features

- **Multi-horizon prediction**: 1, 3, 7, 14, 30, 60 days
- **Sentiment analysis**: FinViz + Google News integration
- **Hyperparameter tuning**: Optuna integration
- **MLflow tracking**: Experiment logging
- **Walk-forward validation**: For Time Series
- **Holiday-aware**: NYSE calendar integration

---

## Documentation

See detailed API reference at [docs/FUNCTIONS.md](docs/FUNCTIONS.md).

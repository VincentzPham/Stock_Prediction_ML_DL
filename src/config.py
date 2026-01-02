"""
Configuration file for Stock Price Prediction project.
Chứa tất cả hyperparameters, paths, và constants.
"""

import os
from pathlib import Path

# ============================================
# PATHS
# ============================================
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "Data"
MODELS_DIR = ROOT_DIR / "Models"
RESULTS_DIR = ROOT_DIR / "Result"
LOGS_DIR = ROOT_DIR / "logs"

# ============================================
# TICKERS
# ============================================
TICKERS = ['AAPL', 'AMZN', 'AVGO', 'BTC-USD', 'GOOG', 'META', 'MSFT', 'NVDA', 'SAP', 'TSLA', 'TSM']

IPO_DATES = {
    'AAPL': '1980-12-12',
    'AMZN': '1997-05-15',
    'AVGO': '2009-08-06',
    'BTC-USD': '2014-09-17',
    'GOOG': '2004-08-19',
    'META': '2012-05-18',
    'MSFT': '1986-03-13',
    'NVDA': '1999-01-22',
    'SAP': '1988-11-04',
    'TSLA': '2010-06-29',
    'TSM': '1994-09-05'
}

# ============================================
# MODEL NAMES
# ============================================
MODEL_NAMES = [
    'LSTM',
    'BiLSTM', 
    'LSTM-GRU',
    'RNN',
    'ANN',
    'ARIMA',
    'SARIMA',
    'Prophet',
    'Random Forest',
    'Decision Tree',
    'Multiple Linear Regression',
    'Exponential Smoothing'
]

# ============================================
# DATA PROCESSING
# ============================================
TRAIN_TEST_SPLIT = 0.8
TIME_STEP = 60  # Số ngày lookback cho LSTM/RNN
FEATURE_COLUMNS = ['Close', 'High', 'Low', 'Open', 'Volume']
TARGET_COLUMN = 'Close'

# ============================================
# DEEP LEARNING HYPERPARAMETERS
# ============================================
DL_CONFIG = {
    'LSTM': {
        'units': [50, 50],
        'dropout': 0.2,
        'epochs': 50,
        'batch_size': 32,
        'patience': 10,
        'optimizer': 'adam',
        'loss': 'mean_squared_error'
    },
    'BiLSTM': {
        'units': [50, 50],
        'dropout': 0.2,
        'epochs': 50,
        'batch_size': 32,
        'patience': 10,
        'optimizer': 'adam',
        'loss': 'mean_squared_error'
    },
    'LSTM-GRU': {
        'lstm_units': 50,
        'gru_units': 50,
        'dropout': 0.2,
        'epochs': 50,
        'batch_size': 32,
        'patience': 10,
        'optimizer': 'adam',
        'loss': 'mean_squared_error'
    },
    'RNN': {
        'units': [50, 50],
        'dropout': 0.2,
        'epochs': 50,
        'batch_size': 32,
        'patience': 10,
        'optimizer': 'adam',
        'loss': 'mean_squared_error'
    },
    'ANN': {
        'hidden_layers': [64, 32],
        'dropout': 0.2,
        'epochs': 50,
        'batch_size': 32,
        'patience': 10,
        'optimizer': 'adam',
        'loss': 'mean_squared_error'
    }
}

# ============================================
# TIME SERIES HYPERPARAMETERS
# ============================================
TS_CONFIG = {
    'ARIMA': {
        'order': (4, 1, 0),  # (p, d, q)
        'trend': 'n'
    },
    'SARIMA': {
        'order': (4, 1, 0),
        'seasonal_order': (1, 1, 1, 7),  # (P, D, Q, m)
        'trend': 'n'
    },
    'Exponential Smoothing': {
        'trend': 'add',
        'seasonal': 'add',
        'seasonal_periods': 252  # Trading days per year
    },
    'Prophet': {
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': False,
        'changepoint_prior_scale': 0.05
    }
}

# ============================================
# ML HYPERPARAMETERS
# ============================================
ML_CONFIG = {
    'Random Forest': {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42
    },
    'Decision Tree': {
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42
    },
    'Multiple Linear Regression': {
        'fit_intercept': True,
        'normalize': False
    }
}

# ============================================
# FEATURE ENGINEERING
# ============================================
ENGINEERED_FEATURES = [
    'Price_Diff',      # High - Low
    'Avg_Price',       # (High + Low) / 2
    'Volume_Ratio',    # Volume / Mean(Volume)
    'Price_Change',    # Close - Open
    'Returns',         # (Close - Prev_Close) / Prev_Close
    'MA_7',            # Moving Average 7 days
    'MA_21',           # Moving Average 21 days
    'MA_50',           # Moving Average 50 days
    'Volatility'       # Rolling std of returns
]

# ============================================
# OUTPUT SETTINGS
# ============================================
SAVE_PLOTS = True
SAVE_METRICS = True
VERBOSE = True


def get_model_path(ticker: str, model_name: str) -> Path:
    """Trả về đường dẫn lưu model."""
    path = MODELS_DIR / ticker / model_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_result_path(ticker: str, model_name: str) -> Path:
    """Trả về đường dẫn lưu kết quả."""
    path = RESULTS_DIR / ticker / model_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_path(ticker: str) -> Path:
    """Trả về đường dẫn file CSV của ticker."""
    return DATA_DIR / f"{ticker}.csv"

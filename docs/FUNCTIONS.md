# Functions and API Reference

## Cấu Trúc Source Code

```
backend/
├── api/                        # FastAPI Backend (RESTful)
│   ├── app.py                  # Application factory, route registration
│   ├── routes/                 # API route handlers
│   │   ├── __init__.py         # Router exports
│   │   ├── root.py             # GET / (health check)
│   │   ├── tickers.py          # /tickers endpoints
│   │   ├── models.py           # /models endpoints
│   │   └── predictions.py      # /predictions endpoints
│   ├── schemas/                # Pydantic models for validation
│   │   ├── __init__.py         # Schema exports
│   │   ├── requests.py         # Request models (PredictRequest)
│   │   └── responses.py        # Response models (PredictResponse, etc.)
│   └── services/               # Business logic layer
│       ├── __init__.py         # Service exports
│       ├── market_service.py   # Trading calendar utilities
│       └── prediction_service.py # Prediction algorithms
├── config.py                   # Cấu hình toàn cục
├── data/
│   ├── preprocessor.py         # DataPreprocessor class
│   └── sentiment_analyzer.py   # SentimentAnalyzer class
├── models/
│   ├── base.py                 # BaseModel (abstract)
│   ├── deep_learning/          # LSTM, BiLSTM, RNN, ANN, LSTM-GRU
│   ├── machine_learning/       # Random Forest, Decision Tree, Linear Regression
│   └── time_series/            # ARIMA, SARIMA, Prophet, Exponential Smoothing
└── training/
    └── trainer.py              # ModelTrainer class

frontend/
├── app.py                      # Main Streamlit application entry point
├── config.py                   # Configuration constants (API URL, colors)
├── api_client.py               # API client for backend communication
├── styles.py                   # CSS styles and HTML templates
└── components/                 # Reusable UI components
    ├── __init__.py             # Component exports
    ├── charts.py               # Plotly chart components
    └── metrics.py              # Metrics display components

scripts/
└── train_all.py                # Training script

tests/
├── unit/                       # Unit tests
└── integration/                # Integration tests
```

---

## API Endpoints

### RESTful Structure (New)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check, welcome message |
| GET | `/tickers` | List available tickers |
| GET | `/tickers/{ticker}/historical` | Get historical prices |
| GET | `/tickers/{ticker}/latest` | Get latest price |
| GET | `/models` | List available models |
| GET | `/models/{ticker}/{model}/metrics` | Get model metrics |
| POST | `/predictions` | Create new prediction |

### Legacy Endpoints (Backward Compatible)

| Method | Endpoint | Maps To |
|--------|----------|---------|
| GET | `/historical/{ticker}` | `/tickers/{ticker}/historical` |
| GET | `/metrics/{ticker}/{model}` | `/models/{ticker}/{model}/metrics` |
| GET | `/latest-price/{ticker}` | `/tickers/{ticker}/latest` |
| POST | `/predict` | `/predictions` |

### `POST /predictions`

Dự đoán giá cổ phiếu.

**Request:**
```json
{
  "ticker": "AAPL",
  "model": "Random Forest",
  "horizon": 7
}
```

**Response:**
```json
{
  "ticker": "AAPL",
  "model": "Random Forest",
  "last_actual_date": "2025-01-30",
  "last_actual_price": 236.40,
  "horizon": 7,
  "predictions": [
    {"day": 1, "date": "2025-01-31", "predicted_price": 237.50},
    ...
  ],
  "currency": "USD",
  "model_path": "..."
}
```

### `GET /tickers/{ticker}/historical`

Get historical price data.

**Parameters:**
- `ticker` (path): Stock symbol (e.g., "AAPL")
- `days` (query, optional): Number of days (default: 60)

**Response:**
```json
{
  "ticker": "AAPL",
  "data": [
    {"date": "2025-01-30", "actual": 236.40},
    {"date": "2025-01-29", "actual": 235.10},
    ...
  ]
}
```

### `GET /tickers/{ticker}/latest`

Get latest price data.

**Response:**
```json
{
  "ticker": "AAPL",
  "date": "2025-01-30",
  "close": 236.40,
  "open": 234.50,
  "high": 237.80,
  "low": 233.90,
  "volume": 45678900
}
```

### `GET /models/{ticker}/{model}/metrics`

Get evaluation metrics for a trained model.

**Response:**
```json
{
  "ticker": "AAPL",
  "model": "LSTM",
  "mse": 12.34,
  "rmse": 3.51,
  "mae": 2.89,
  "mape": 1.23,
  "r2": 0.95
}
```

---

## API Module (`backend/api/`)

### Application Factory

```python
from backend.api import app, create_app

# Default app instance
app  # FastAPI application

# Or create new instance
my_app = create_app()
```

### Schemas

```python
from backend.api.schemas import (
    PredictRequest,
    PredictResponse,
    PredictionDay,
    HistoricalResponse,
    HistoricalDataPoint,
    MetricsResponse,
    LatestPriceResponse,
    TickersResponse,
    ModelsResponse,
)

# Request validation
request = PredictRequest(ticker="AAPL", model="LSTM", horizon=7)

# Response serialization
response = PredictResponse(
    ticker="AAPL",
    model="LSTM",
    last_actual_date="2025-01-30",
    last_actual_price=236.40,
    horizon=7,
    predictions=[PredictionDay(day=1, date="2025-01-31", predicted_price=237.50)],
    model_path="/path/to/model"
)
```

### Services

```python
from backend.api.services import PredictionService, MarketService

# Get trading days (excludes weekends and holidays)
trading_days = MarketService.get_trading_days(
    start_date=pd.Timestamp("2025-01-30"),
    num_days=7,
    ticker="AAPL"
)

# Multi-step prediction for ML models
predictions = PredictionService.predict_multi_step_ml(
    model_instance=model,
    preprocessor=preprocessor,
    num_steps=7
)

# Multi-step prediction for DL models (with drift correction)
predictions = PredictionService.predict_multi_step_dl(
    model=model.model,
    preprocessor=preprocessor,
    scaler=scaler,
    num_steps=7,
    time_step=60
)
```

---

## Config Module (`backend/config.py`)

### Constants
```python
from backend.config import (
    ROOT_DIR,           # Path đến thư mục gốc
    DATA_DIR,           # Path đến Data/
    MODELS_DIR,         # Path đến Models/
    RESULTS_DIR,        # Path đến Result/
    TICKERS,            # ['AAPL', 'AMZN', ...]
    IPO_DATES,          # {'AAPL': '1980-12-12', ...}
    MODEL_NAMES,        # List tên các models
    TRAIN_TEST_SPLIT,   # 0.8
    TIME_STEP,          # 60
)
```

### Hyperparameters
```python
DL_CONFIG = {
    'LSTM': {'units': [50, 50], 'dropout': 0.2, 'epochs': 50, ...},
    'BiLSTM': {...},
    ...
}

TS_CONFIG = {
    'ARIMA': {'order': (4, 1, 0)},
    'SARIMA': {'order': (4, 1, 0), 'seasonal_order': (1, 1, 1, 7)},
    ...
}

ML_CONFIG = {
    'Random Forest': {'n_estimators': 100, 'random_state': 42},
    ...
}
```

### Helper Functions
```python
get_model_path(ticker, model_name) -> Path  # Models/{ticker}/{model}/
get_result_path(ticker, model_name) -> Path # Result/{ticker}/{model}/
get_data_path(ticker) -> Path               # Data/{ticker}.csv
```

---

## Data Module (`backend/data/`)

### DataPreprocessor

```python
from backend.data import DataPreprocessor

# Khởi tạo
preprocessor = DataPreprocessor(ticker='AAPL')

# Load data
df = preprocessor.load_data()              # Đọc CSV, xử lý dates
df = preprocessor.add_features()           # Thêm technical indicators

# Prepare data cho các loại model
X_train, X_test, y_train, y_test, scaler = preprocessor.prepare_lstm_data(time_step=60)
X_train, X_test, y_train, y_test = preprocessor.prepare_ml_data(feature_cols=None)
train_series, test_series = preprocessor.prepare_timeseries_data()
train_df, test_df = preprocessor.prepare_prophet_data()  # {'ds': date, 'y': value}

# Utilities
dates = preprocessor.get_dates('test')     # DatetimeIndex
original = preprocessor.inverse_transform(scaled_data)
info = preprocessor.get_info()             # Dict với metadata
```

### SentimentAnalyzer

```python
from backend.data import SentimentAnalyzer

analyzer = SentimentAnalyzer()

# Phân tích
daily_sentiment = analyzer.get_daily_sentiment(ticker='AAPL')
# Trả về DataFrame với index là Date và cột Sentiment_Score
```

---

## Models Module (`backend/models/`)

### BaseModel (Abstract)

Tất cả models kế thừa từ `BaseModel`:

```python
class BaseModel(ABC):
    MODEL_NAME: str   # Tên model
    MODEL_TYPE: str   # 'deep_learning', 'time_series', 'ml'
    
    # Abstract methods
    def build(self, **kwargs) -> None
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs) -> Dict
    def predict(self, X) -> np.ndarray
    
    # Implemented methods
    def evaluate(self, y_true, y_pred) -> Dict[str, float]  # MAE, MSE, RMSE, MAPE, R2
    def save_model(self, filename=None) -> Path
    def load_model(self, filepath) -> None
    def save_results(self, dates, actuals, predictions) -> Path
    def save_metrics() -> Path
    def plot_predictions(self, dates, actuals, predictions, save=True) -> Path
    def plot_residuals(self, actuals, predictions, save=True) -> Path
    def summary() -> Dict
```

### Deep Learning Models

```python
from backend.models import LSTMModel, BiLSTMModel, LSTMGRUModel, RNNModel, ANNModel

model = LSTMModel(ticker='AAPL')
model.build(input_shape=(60, 1), units=[50, 50], dropout=0.2)
model.train(X_train, y_train, X_test, y_test, epochs=50, batch_size=32, patience=10)
predictions = model.predict(X_test)
model.plot_training_history(save=True)
```

### Time Series Models

```python
from backend.models import ARIMAModel, SARIMAModel, ProphetModel, ExponentialSmoothingModel

# ARIMA
model = ARIMAModel(ticker='AAPL')
model.build(order=(4, 1, 0))
model.train(train_series)
predictions = model.predict(steps=100)  # hoặc model.predict(test_series=test_series)

# SARIMA
model = SARIMAModel(ticker='AAPL')
model.build(order=(4, 1, 0), seasonal_order=(1, 1, 1, 7))
model.train(train_series, use_log=True)

# Prophet
model = ProphetModel(ticker='AAPL')
model.build(yearly_seasonality=True)
model.train(train_df)  # DataFrame với columns 'ds', 'y'
predictions = model.predict(test_df=test_df)
model.plot_components()
```

### ML Models

```python
from backend.models import RandomForestModel, DecisionTreeModel, LinearRegressionModel

# Random Forest
model = RandomForestModel(ticker='AAPL')
model.build(n_estimators=100, max_depth=None)
model.train(X_train, y_train)
predictions = model.predict(X_test)
model.plot_feature_importance(feature_names=['Open', 'High', ...])

# Linear Regression
model = LinearRegressionModel(ticker='AAPL')
model.train(X_train, y_train)
equation = model.get_equation(feature_names)
```

---

## Training Module (`backend/training/`)

### ModelTrainer

```python
from backend.training import ModelTrainer

trainer = ModelTrainer(verbose=True)

# Train một model
result = trainer.train_single(
    ticker='AAPL',
    model_name='LSTM',
    save_model=True,
    save_results=True
)

# Train tất cả models cho một ticker
results = trainer.train_all_models(
    ticker='AAPL',
    models=['LSTM', 'Random Forest'],  # None = tất cả
    skip_on_error=True
)

# Train một model cho tất cả tickers
results = trainer.train_all_tickers(
    model_name='Random Forest',
    tickers=['AAPL', 'MSFT'],  # None = tất cả
    skip_on_error=True
)

# Train TẤT CẢ (12 models × 11 tickers = 132 runs)
results = trainer.train_all(
    tickers=None,    # None = TICKERS
    models=None,     # None = tất cả models
    skip_on_error=True
)

# Get summary
summary_df = trainer.get_summary()
```

### Model Registry

```python
from backend.training.trainer import MODEL_REGISTRY, DEEP_LEARNING_MODELS, TIME_SERIES_MODELS, ML_MODELS

MODEL_REGISTRY = {
    'LSTM': LSTMModel,
    'BiLSTM': BiLSTMModel,
    ...
}

DEEP_LEARNING_MODELS = ['LSTM', 'BiLSTM', 'LSTM-GRU', 'RNN', 'ANN']
TIME_SERIES_MODELS = ['ARIMA', 'SARIMA', 'Prophet', 'Exponential Smoothing']
ML_MODELS = ['Random Forest', 'Decision Tree', 'Multiple Linear Regression']
```

---

## Scripts (`scripts/`)

### train_all.py

```bash
# Xem help
uv run python scripts/train_all.py --help

# Train một ticker + model
uv run python scripts/train_all.py -t AAPL -m "Random Forest"

# Train nhiều
uv run python scripts/train_all.py --tickers AAPL,MSFT --models LSTM,ARIMA

# Train theo loại
uv run python scripts/train_all.py --ml-only      # Chỉ ML models
uv run python scripts/train_all.py --dl-only      # Chỉ Deep Learning
uv run python scripts/train_all.py --ts-only      # Chỉ Time Series

# List available
uv run python scripts/train_all.py --list-models
uv run python scripts/train_all.py --list-tickers
```

---

## Common Patterns

### 1. Full Training Pipeline

```python
from backend.data import DataPreprocessor
from backend.models import LSTMModel

# Load data
prep = DataPreprocessor('AAPL')
prep.load_data()
prep.add_features()

# Prepare
X_train, X_test, y_train, y_test, scaler = prep.prepare_lstm_data()

# Train
model = LSTMModel('AAPL')
model.build()
model.train(X_train, y_train, X_test, y_test)

# Predict
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
actuals = scaler.inverse_transform(y_test.reshape(-1, 1))

# Evaluate
metrics = model.evaluate(actuals, predictions)
print(metrics)

# Save
model.save_model()
model.save_results(prep.get_dates('test'), actuals, predictions)
model.plot_predictions(prep.get_dates('test'), actuals, predictions)
```

### 2. Quick ML Training

```python
from backend.training import ModelTrainer

trainer = ModelTrainer()

# Train tất cả ML models cho AAPL (nhanh nhất)
results = trainer.train_all_models(
    ticker='AAPL',
    models=['Random Forest', 'Decision Tree', 'Multiple Linear Regression']
)

# Xem kết quả
print(trainer.get_summary())
```

---

## Error-Prone Areas (Lưu ý)

1. **Time Series Split**: KHÔNG shuffle:
   ```python
   train_test_split(X, y, test_size=0.2, shuffle=False)
   ```

2. **LSTM Input Shape**: `(samples, time_steps, features)`:
   ```python
   X = X.reshape(X.shape[0], X.shape[1], 1)
   ```

3. **Inverse Transform**: Nhớ chuyển về giá gốc sau predict:
   ```python
   predictions = scaler.inverse_transform(predictions)
   ```

---

## Frontend Module (`frontend/`)

### API Client

The `APIClient` class handles all communication with the backend API.

```python
from frontend.api_client import api_client

# Get available tickers
tickers = api_client.get_tickers()  # ['AAPL', 'AMZN', ...]

# Get available models
models = api_client.get_models()  # ['LSTM', 'Random Forest', ...]

# Get historical data
df = api_client.get_historical_data('AAPL', days=60)
# DataFrame with columns: ['date', 'actual']

# Get latest price
latest = api_client.get_latest_price('AAPL')
# {'ticker': 'AAPL', 'date': '2025-01-30', 'close': 236.40, ...}

# Get model metrics
metrics = api_client.get_metrics('AAPL', 'LSTM')
# {'mse': 12.34, 'rmse': 3.51, 'mae': 2.89, 'mape': 1.23, 'r2': 0.95}

# Generate prediction
result = api_client.predict('AAPL', 'LSTM', horizon=7)
# {'ticker': 'AAPL', 'predictions': [...], ...}
```

### Chart Components

```python
from frontend.components import (
    create_historical_chart,
    create_prediction_chart,
)

# Historical chart
fig = create_historical_chart(historical_df, ticker='AAPL', days=90)
st.plotly_chart(fig)

# Prediction chart with historical + forecast
fig = create_prediction_chart(historical_df, predictions, ticker='AAPL')
st.plotly_chart(fig)
```

### Metrics Components

```python
from frontend.components import display_metrics_cards, display_price_card

# Display metrics in styled cards
display_metrics_cards(metrics)  # Shows MSE, RMSE, MAE, MAPE, R2

# Display current price
display_price_card(ticker='AAPL', date='2025-01-30', price=236.40)
```

### Configuration

```python
from frontend.config import (
    API_URL,           # Backend API URL
    HORIZON_OPTIONS,   # Prediction horizon choices
    CHART_COLORS,      # Color palette for charts
    REQUEST_TIMEOUT,   # API timeout in seconds
)
```

### Styles

```python
from frontend.styles import get_custom_css, get_hero_html, get_footer_html

# Apply custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Render hero section
st.markdown(get_hero_html(), unsafe_allow_html=True)

# Render footer
st.markdown(get_footer_html(), unsafe_allow_html=True)
```
   ```

4. **Prophet Format**: Cần DataFrame với columns `ds` và `y`:
   ```python
   df = pd.DataFrame({'ds': dates, 'y': values})
   ```

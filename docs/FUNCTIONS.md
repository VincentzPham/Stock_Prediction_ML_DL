# 📚 Functions & API Reference

## Cấu Trúc Source Code

```
src/
├── api/                # FastAPI Backend
│   └── app.py          # Entry point
├── ui/                 # Streamlit Frontend
│   └── app.py          # Entry point
├── config.py           # Cấu hình toàn cục
├── data/
│   ├── preprocessor.py # DataPreprocessor class
│   └── downloader.py   # DataDownloader class
├── models/
│   ├── base.py         # BaseModel (abstract)
│   └── *.py            # Các model implementations
└── training/
    └── trainer.py      # ModelTrainer class
```

---

## API Endpoints

### `POST /predict`

Dự đoán giá cổ phiếu.

**Request:**
```json
{
  "ticker": "AAPL",
  "model": "Random Forest",
  "horizon": 1
}
```

**Response:**
```json
{
  "ticker": "AAPL",
  "model": "Random Forest",
  "date": "2023-12-31",
  "horizon": 1,
  "prediction": 150.5,
  "currency": "USD",
  "model_path": "..."
}
```

---

## Models Module (`src/models/`)

### BaseModel

```python
class BaseModel(ABC):
    # ... existing methods ...
    
    @abstractmethod
    def predict_next(self, preprocessor, horizon: int = 1) -> float:
        """
        Dự đoán giá trị tương lai (horizon days ahead).
        """
        pass
```

### Implementations

Tất cả models (LSTM, ARIMA, Random Forest,...) đều implement `predict_next` để hỗ trợ unified interface cho API.

---

## Config Module (`src/config.py`)

### Constants
```python
from src.config import (
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

## Data Module (`src/data/`)

### DataPreprocessor

```python
from src.data import DataPreprocessor

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

### DataDownloader

```python
from src.data import DataDownloader

downloader = DataDownloader(save_dir=DATA_DIR)

# Download
df = downloader.download(ticker='AAPL', start_date=None, end_date=None, save=True)
results = downloader.download_all(tickers=TICKERS, save=True)
df = downloader.update_data(ticker='AAPL')  # Cập nhật data mới
```

---

## Models Module (`src/models/`)

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
    def save_metrics(self) -> Path
    def plot_predictions(self, dates, actuals, predictions, save=True) -> Path
    def plot_residuals(self, actuals, predictions, save=True) -> Path
    def summary(self) -> Dict
```

### Deep Learning Models

```python
from src.models import LSTMModel, BiLSTMModel, LSTMGRUModel, RNNModel, ANNModel

model = LSTMModel(ticker='AAPL')
model.build(input_shape=(60, 1), units=[50, 50], dropout=0.2)
model.train(X_train, y_train, X_test, y_test, epochs=50, batch_size=32, patience=10)
predictions = model.predict(X_test)
model.plot_training_history(save=True)
```

### Time Series Models

```python
from src.models import ARIMAModel, SARIMAModel, ProphetModel, ExponentialSmoothingModel

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
from src.models import RandomForestModel, DecisionTreeModel, LinearRegressionModel

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

## Training Module (`src/training/`)

### ModelTrainer

```python
from src.training import ModelTrainer

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
from src.training.trainer import MODEL_REGISTRY, DEEP_LEARNING_MODELS, TIME_SERIES_MODELS, ML_MODELS

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
from src.data import DataPreprocessor
from src.models import LSTMModel

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
from src.training import ModelTrainer

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

1. **TensorFlow + Python 3.13**: Chưa tương thích tốt. Dùng `--ml-only` hoặc `--ts-only`.

2. **CSV Reading**: File từ yfinance cần `skiprows=2`:
   ```python
   df = pd.read_csv(path, skiprows=2)
   df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
   ```

3. **Time Series Split**: KHÔNG shuffle:
   ```python
   train_test_split(X, y, test_size=0.2, shuffle=False)
   ```

4. **LSTM Input Shape**: `(samples, time_steps, features)`:
   ```python
   X = X.reshape(X.shape[0], X.shape[1], 1)
   ```

5. **Inverse Transform**: Nhớ chuyển về giá gốc sau predict:
   ```python
   predictions = scaler.inverse_transform(predictions)
   ```

6. **Prophet Format**: Cần DataFrame với columns `ds` và `y`:
   ```python
   df = pd.DataFrame({'ds': dates, 'y': values})
   ```

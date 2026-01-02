"""
Model Trainer Module
Unified training interface for all models.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Type
import pandas as pd
import numpy as np
from datetime import datetime
import json
import traceback

sys.path.append(str(Path(__file__).parent.parent))

from config import (
    TICKERS, MODEL_NAMES, RESULTS_DIR, MODELS_DIR,
    TIME_STEP, TRAIN_TEST_SPLIT, TARGET_COLUMN
)
from data.preprocessor import DataPreprocessor
from models import (
    BaseModel,
    LSTMModel, BiLSTMModel, LSTMGRUModel, RNNModel, ANNModel,
    ARIMAModel, SARIMAModel, ProphetModel, ExponentialSmoothingModel,
    RandomForestModel, DecisionTreeModel, LinearRegressionModel
)


# Model registry
MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    'LSTM': LSTMModel,
    'BiLSTM': BiLSTMModel,
    'LSTM-GRU': LSTMGRUModel,
    'RNN': RNNModel,
    'ANN': ANNModel,
    'ARIMA': ARIMAModel,
    'SARIMA': SARIMAModel,
    'Prophet': ProphetModel,
    'Exponential Smoothing': ExponentialSmoothingModel,
    'Random Forest': RandomForestModel,
    'Decision Tree': DecisionTreeModel,
    'Multiple Linear Regression': LinearRegressionModel
}

# Model types
DEEP_LEARNING_MODELS = ['LSTM', 'BiLSTM', 'LSTM-GRU', 'RNN', 'ANN']
TIME_SERIES_MODELS = ['ARIMA', 'SARIMA', 'Prophet', 'Exponential Smoothing']
ML_MODELS = ['Random Forest', 'Decision Tree', 'Multiple Linear Regression']


class ModelTrainer:
    """
    Unified trainer cho tất cả models.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: Dict[str, Dict[str, Any]] = {}
        
    def train_single(
        self, 
        ticker: str, 
        model_name: str,
        save_model: bool = True,
        save_results: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train một model cho một ticker.
        
        Args:
            ticker: Mã chứng khoán
            model_name: Tên model
            save_model: Có lưu model không
            save_results: Có lưu kết quả không
            
        Returns:
            Dict chứa kết quả training và metrics.
        """
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
        
        print(f"\n{'='*60}")
        print(f"Training {model_name} for {ticker}")
        print(f"{'='*60}")
        
        result = {
            'ticker': ticker,
            'model': model_name,
            'status': 'pending',
            'metrics': {},
            'error': None
        }
        
        try:
            # Load and prepare data
            preprocessor = DataPreprocessor(ticker)
            preprocessor.load_data()
            preprocessor.add_features()
            
            # Get model class
            ModelClass = MODEL_REGISTRY[model_name]
            model = ModelClass(ticker)
            
            # Train based on model type
            if model_name in DEEP_LEARNING_MODELS:
                result = self._train_deep_learning(model, preprocessor, **kwargs)
            elif model_name in TIME_SERIES_MODELS:
                result = self._train_time_series(model, preprocessor, model_name, **kwargs)
            elif model_name in ML_MODELS:
                result = self._train_ml(model, preprocessor, **kwargs)
            
            result['status'] = 'success'
            
            # Save model
            if save_model:
                model.save_model()
            
            # Save results
            if save_results and 'predictions' in result:
                dates = preprocessor.get_dates('test')
                model.save_results(
                    dates=dates,
                    actuals=result['actuals'],
                    predictions=result['predictions']
                )
                model.save_metrics()
                
                # Plot
                model.plot_predictions(
                    dates=dates,
                    actuals=result['actuals'],
                    predictions=result['predictions']
                )
            
            print(f"\n✓ {model_name} for {ticker} completed successfully!")
            print(f"  Metrics: {result['metrics']}")
            
        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
            print(f"\n✗ Error training {model_name} for {ticker}: {e}")
            if self.verbose:
                traceback.print_exc()
        
        # Store result
        key = f"{ticker}_{model_name}"
        self.results[key] = result
        
        return result
    
    def _train_deep_learning(
        self, 
        model: BaseModel, 
        preprocessor: DataPreprocessor,
        **kwargs
    ) -> Dict[str, Any]:
        """Train deep learning models (LSTM, RNN, etc)."""
        # Prepare data
        X_train, X_test, y_train, y_test, scaler = preprocessor.prepare_lstm_data()
        
        # Build and train
        model.build()
        model.train(X_train, y_train, X_test, y_test, **kwargs)
        
        # Predict
        predictions = model.predict(X_test)
        
        # Inverse transform
        predictions = scaler.inverse_transform(predictions)
        actuals = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Evaluate
        metrics = model.evaluate(actuals, predictions)
        
        return {
            'ticker': model.ticker,
            'model': model.MODEL_NAME,
            'predictions': predictions.flatten(),
            'actuals': actuals.flatten(),
            'metrics': metrics,
            'history': model.history
        }
    
    def _train_time_series(
        self, 
        model: BaseModel, 
        preprocessor: DataPreprocessor,
        model_name: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Train time series models (ARIMA, Prophet, etc)."""
        if model_name == 'Prophet':
            # Prophet cần format đặc biệt
            train_df, test_df = preprocessor.prepare_prophet_data()
            
            model.build()
            model.train(train_df, **kwargs)
            predictions = model.predict(test_df=test_df)
            actuals = test_df['y'].values
            
        else:
            # ARIMA, SARIMA, Exponential Smoothing
            train_series, test_series = preprocessor.prepare_timeseries_data()
            
            model.build()
            model.train(train_series, **kwargs)
            
            if model_name in ['ARIMA', 'SARIMA']:
                # Rolling forecast cho ARIMA/SARIMA (slower but more accurate)
                # Để tiết kiệm thời gian, dùng simple forecast
                predictions = model.predict(steps=len(test_series))
            else:
                predictions = model.predict(test_series=test_series)
            
            actuals = test_series.values
        
        # Evaluate
        metrics = model.evaluate(actuals, predictions)
        
        return {
            'ticker': model.ticker,
            'model': model.MODEL_NAME,
            'predictions': np.array(predictions).flatten(),
            'actuals': np.array(actuals).flatten(),
            'metrics': metrics
        }
    
    def _train_ml(
        self, 
        model: BaseModel, 
        preprocessor: DataPreprocessor,
        **kwargs
    ) -> Dict[str, Any]:
        """Train ML models (Random Forest, etc)."""
        # Prepare data
        X_train, X_test, y_train, y_test = preprocessor.prepare_ml_data()
        
        # Build and train
        model.build()
        model.train(X_train, y_train, **kwargs)
        
        # Predict
        predictions = model.predict(X_test)
        
        # Evaluate
        metrics = model.evaluate(y_test, predictions)
        
        return {
            'ticker': model.ticker,
            'model': model.MODEL_NAME,
            'predictions': predictions,
            'actuals': y_test,
            'metrics': metrics
        }
    
    def train_all_models(
        self, 
        ticker: str,
        models: List[str] = None,
        skip_on_error: bool = True,
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train tất cả models cho một ticker.
        
        Args:
            ticker: Mã chứng khoán
            models: Danh sách models. Nếu None, train tất cả.
            skip_on_error: Bỏ qua nếu có lỗi và tiếp tục model tiếp theo.
        """
        if models is None:
            models = list(MODEL_REGISTRY.keys())
        
        results = {}
        
        print(f"\n{'#'*60}")
        print(f"Training ALL models for {ticker}")
        print(f"Models: {models}")
        print(f"{'#'*60}")
        
        for model_name in models:
            try:
                result = self.train_single(ticker, model_name, **kwargs)
                results[model_name] = result
            except Exception as e:
                if skip_on_error:
                    print(f"Skipping {model_name} due to error: {e}")
                    results[model_name] = {'status': 'failed', 'error': str(e)}
                else:
                    raise
        
        return results

    def predict_horizon(
        self, 
        ticker: str, 
        model_name: str,
        horizon: int = 1,
        model_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Predict future values (next day/horizon) using a trained model.
        
        Args:
            ticker: Stock ticker symbol.
            model_name: Name of the model to use.
            horizon: Number of steps to predict into the future (default 1).
            model_path: Path to a specific saved model. If None, finds the latest.
            
        Returns:
            Dictionary containing prediction details.
        """
        print(f"\nPredicting horizon={horizon} for {ticker} using {model_name}...")
        
        # 1. Load & Prepare Data
        preprocessor = DataPreprocessor(ticker)
        preprocessor.load_data()
        preprocessor.add_features()
        
        # 2. Load Model
        if model_name not in MODEL_REGISTRY:
             raise ValueError(f"Unknown model: {model_name}")

        ModelClass = MODEL_REGISTRY[model_name]
        model = ModelClass(ticker)
        
        if model_path is None:
            # Find latest model in the model directory
            files = list(model.model_dir.glob(f"*{model_name}*"))
            valid_exts = ['.keras', '.pkl']
            files = [f for f in files if f.suffix in valid_exts]
            
            if not files:
                raise FileNotFoundError(f"No saved model found for {ticker} - {model_name}")
            
            model_path = max(files, key=lambda x: x.stat().st_mtime)
            
        model.load_model(model_path)
        
        # 3. Predict
        prediction = None
        last_date = preprocessor.df.index[-1]
        
        try:
            # Use the unified predict_next interface
            prediction = model.predict_next(preprocessor, horizon=horizon)
        except Exception as e:
            print(f"Error predicting with {model_name}: {e}")
            if self.verbose:
                traceback.print_exc()
            prediction = 0.0

        print(f"Prediction for {ticker} (+{horizon} days): {prediction}")
        
        return {
            'ticker': ticker,
            'model': model_name,
            'date': str(last_date.date()),
            'horizon': horizon,
            'prediction': float(prediction),
            'model_path': str(model_path)
        }
    
    def train_all_tickers(
        self, 
        model_name: str,
        tickers: List[str] = None,
        skip_on_error: bool = True,
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train một model cho tất cả tickers.
        """
        if tickers is None:
            tickers = TICKERS
        
        results = {}
        
        print(f"\n{'#'*60}")
        print(f"Training {model_name} for ALL tickers")
        print(f"Tickers: {tickers}")
        print(f"{'#'*60}")
        
        for ticker in tickers:
            try:
                result = self.train_single(ticker, model_name, **kwargs)
                results[ticker] = result
            except Exception as e:
                if skip_on_error:
                    print(f"Skipping {ticker} due to error: {e}")
                    results[ticker] = {'status': 'failed', 'error': str(e)}
                else:
                    raise
        
        return results
    
    def train_all(
        self,
        tickers: List[str] = None,
        models: List[str] = None,
        skip_on_error: bool = True,
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train TẤT CẢ models cho TẤT CẢ tickers.
        
        Total: 12 models × 11 tickers = 132 training runs
        """
        if tickers is None:
            tickers = TICKERS
        if models is None:
            models = list(MODEL_REGISTRY.keys())
        
        total = len(tickers) * len(models)
        print(f"\n{'#'*60}")
        print(f"TRAINING ALL: {len(models)} models × {len(tickers)} tickers = {total} runs")
        print(f"{'#'*60}")
        
        all_results = {}
        completed = 0
        
        for ticker in tickers:
            for model_name in models:
                completed += 1
                print(f"\n[{completed}/{total}] {ticker} - {model_name}")
                
                try:
                    result = self.train_single(ticker, model_name, **kwargs)
                    all_results[f"{ticker}_{model_name}"] = result
                except Exception as e:
                    if skip_on_error:
                        print(f"Skipping due to error: {e}")
                        all_results[f"{ticker}_{model_name}"] = {'status': 'failed', 'error': str(e)}
                    else:
                        raise
        
        # Save summary
        self._save_summary(all_results)
        
        return all_results
    
    def _save_summary(self, results: Dict[str, Dict[str, Any]]) -> Path:
        """Lưu summary của tất cả training runs."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Tạo summary DataFrame
        summary_data = []
        for key, result in results.items():
            row = {
                'ticker': result.get('ticker', key.split('_')[0]),
                'model': result.get('model', key.split('_')[1]),
                'status': result.get('status', 'unknown'),
                'error': result.get('error', None)
            }
            # Add metrics
            metrics = result.get('metrics', {})
            for metric_name, metric_value in metrics.items():
                row[metric_name] = metric_value
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save
        filepath = RESULTS_DIR / f"training_summary_{timestamp}.csv"
        summary_df.to_csv(filepath, index=False)
        print(f"\nSummary saved to {filepath}")
        
        # Also save as JSON
        json_path = RESULTS_DIR / f"training_summary_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return filepath
    
    def get_summary(self) -> pd.DataFrame:
        """Trả về summary DataFrame của các runs đã thực hiện."""
        if not self.results:
            return pd.DataFrame()
        
        summary_data = []
        for key, result in self.results.items():
            row = {
                'ticker': result.get('ticker', ''),
                'model': result.get('model', ''),
                'status': result.get('status', ''),
            }
            metrics = result.get('metrics', {})
            for metric_name, metric_value in metrics.items():
                row[metric_name] = metric_value
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)

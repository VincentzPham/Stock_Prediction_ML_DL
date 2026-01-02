#!/usr/bin/env python
"""
Quick Test Script
Test nhanh một model để verify setup.

Usage:
    uv run python scripts/test_quick.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def test_data_loading():
    """Test data loading."""
    print("\n1. Testing Data Loading...")
    from data.preprocessor import DataPreprocessor
    
    preprocessor = DataPreprocessor('AAPL')
    df = preprocessor.load_data()
    print(f"   ✓ Loaded {len(df)} rows")
    print(f"   ✓ Columns: {list(df.columns)}")
    print(f"   ✓ Date range: {df.index.min()} to {df.index.max()}")
    
    return preprocessor


def test_feature_engineering(preprocessor):
    """Test feature engineering."""
    print("\n2. Testing Feature Engineering...")
    df = preprocessor.add_features()
    print(f"   ✓ Added features, now {len(df.columns)} columns")
    print(f"   ✓ New columns: {[c for c in df.columns if c not in ['Close', 'High', 'Low', 'Open', 'Volume']]}")


def test_data_preparation(preprocessor):
    """Test data preparation."""
    print("\n3. Testing Data Preparation...")
    
    # LSTM data
    X_train, X_test, y_train, y_test, scaler = preprocessor.prepare_lstm_data()
    print(f"   ✓ LSTM data: X_train={X_train.shape}, X_test={X_test.shape}")
    
    # ML data
    X_train_ml, X_test_ml, y_train_ml, y_test_ml = preprocessor.prepare_ml_data()
    print(f"   ✓ ML data: X_train={X_train_ml.shape}, X_test={X_test_ml.shape}")
    
    # Time series data
    train_series, test_series = preprocessor.prepare_timeseries_data()
    print(f"   ✓ Time series: train={len(train_series)}, test={len(test_series)}")
    
    return X_train, X_test, y_train, y_test, scaler


def test_lstm_model(X_train, X_test, y_train, y_test, scaler):
    """Test LSTM model."""
    print("\n4. Testing LSTM Model...")
    from models.lstm import LSTMModel
    
    model = LSTMModel('AAPL')
    model.build(input_shape=(X_train.shape[1], X_train.shape[2]))
    print("   ✓ Model built")
    
    # Train với ít epochs để test nhanh
    model.train(X_train, y_train, X_test, y_test, epochs=2, patience=0)
    print("   ✓ Model trained (2 epochs)")
    
    # Predict
    predictions = model.predict(X_test)
    print(f"   ✓ Predictions shape: {predictions.shape}")
    
    # Inverse transform
    predictions = scaler.inverse_transform(predictions)
    actuals = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Evaluate
    metrics = model.evaluate(actuals, predictions)
    print(f"   ✓ Metrics: RMSE={metrics['RMSE']:.2f}, MAPE={metrics['MAPE']:.2f}%")
    
    return model


def test_ml_model():
    """Test ML model."""
    print("\n5. Testing Random Forest Model...")
    from data.preprocessor import DataPreprocessor
    from models.random_forest import RandomForestModel
    
    preprocessor = DataPreprocessor('AAPL')
    preprocessor.load_data()
    preprocessor.add_features()
    
    X_train, X_test, y_train, y_test = preprocessor.prepare_ml_data()
    
    model = RandomForestModel('AAPL')
    model.build(n_estimators=10)  # Ít cây để test nhanh
    model.train(X_train, y_train)
    
    predictions = model.predict(X_test)
    metrics = model.evaluate(y_test, predictions)
    
    print(f"   ✓ Predictions shape: {predictions.shape}")
    print(f"   ✓ Metrics: RMSE={metrics['RMSE']:.2f}, MAPE={metrics['MAPE']:.2f}%")


def test_trainer():
    """Test trainer."""
    print("\n6. Testing Trainer...")
    from training.trainer import ModelTrainer
    
    trainer = ModelTrainer()
    result = trainer.train_single(
        ticker='AAPL',
        model_name='Multiple Linear Regression',  # Nhanh nhất
        save_model=False,
        save_results=False
    )
    
    print(f"   ✓ Status: {result['status']}")
    print(f"   ✓ Metrics: {result['metrics']}")


def main():
    print("=" * 60)
    print("QUICK TEST - Stock Prediction Project")
    print("=" * 60)
    
    try:
        preprocessor = test_data_loading()
        test_feature_engineering(preprocessor)
        X_train, X_test, y_train, y_test, scaler = test_data_preparation(preprocessor)
        test_lstm_model(X_train, X_test, y_train, y_test, scaler)
        test_ml_model()
        test_trainer()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

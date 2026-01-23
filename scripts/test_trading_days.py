"""
Test script for verifying trading days fix and prediction improvements.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np

from api.app import get_trading_days, predict_multi_step_dl, predict_multi_step_ml
from data.preprocessor import DataPreprocessor
from config import TARGET_COLUMN, TIME_STEP


def test_trading_days():
    """Test that trading days properly exclude market holidays."""
    print("=" * 60)
    print("TEST 1: Trading Days Generation")
    print("=" * 60)
    
    # Test for stocks (should exclude MLK Day 01/19/2026)
    start = pd.Timestamp('2026-01-08')
    trading_days = get_trading_days(start, 30, 'AAPL')
    
    print(f"\nStock (AAPL - NYSE Calendar):")
    print(f"Start date: {start.date()}")
    print(f"First 15 trading days:")
    for i, d in enumerate(trading_days[:15]):
        print(f"  {i+1:2d}. {d.date()} ({d.day_name()})")
    
    # Check if MLK Day (2026-01-19) is excluded
    # Note: MLK Day is always the third Monday of January
    mlk_day = pd.Timestamp('2026-01-19')  # 3rd Monday of Jan 2026
    if mlk_day in trading_days:
        print(f"\n❌ FAIL: MLK Day ({mlk_day.date()}) should be excluded!")
    else:
        print(f"\n✓ PASS: MLK Day ({mlk_day.date()}) properly excluded")
    
    # Check for weekends
    weekend_days = [d for d in trading_days if d.weekday() >= 5]
    if weekend_days:
        print(f"❌ FAIL: Weekend days found: {[str(d.date()) for d in weekend_days]}")
    else:
        print("✓ PASS: No weekend days in trading days")
    
    # Test for crypto (should include all days including weekends)
    print(f"\nCrypto (BTC-USD - 24/7):")
    btc_days = get_trading_days(start, 10, 'BTC-USD')
    for i, d in enumerate(btc_days):
        print(f"  {i+1:2d}. {d.date()} ({d.day_name()})")
    
    # BTC should have weekend days
    btc_weekend = [d for d in btc_days if d.weekday() >= 5]
    if btc_weekend:
        print(f"✓ PASS: BTC-USD includes weekend: {[str(d.date()) for d in btc_weekend]}")
    else:
        print("Note: No weekends in BTC sample (depends on start date)")


def test_lstm_prediction():
    """Test LSTM prediction with drift correction."""
    print("\n" + "=" * 60)
    print("TEST 2: LSTM Prediction with Drift Correction")
    print("=" * 60)
    
    from models.lstm import LSTMModel
    
    # Load data
    preprocessor = DataPreprocessor('AAPL')
    preprocessor.load_data()
    preprocessor.add_features()
    
    # Prepare data
    _, _, _, _, scaler = preprocessor.prepare_lstm_data(time_step=TIME_STEP)
    
    # Load model
    model_instance = LSTMModel('AAPL')
    model_files = list(model_instance.model_dir.glob('*LSTM*'))
    model_files = [f for f in model_files if f.suffix == '.keras']
    if not model_files:
        print("No LSTM model found, skipping test")
        return
    
    model_path = max(model_files, key=lambda x: x.stat().st_mtime)
    model_instance.load_model(model_path)
    
    last_price = float(preprocessor.df[TARGET_COLUMN].iloc[-1])
    print(f"\nLast actual price: ${last_price:.2f}")
    
    # Test prediction
    predictions = predict_multi_step_dl(
        model_instance.model, preprocessor, scaler, 30, TIME_STEP
    )
    
    print(f"\nPredictions:")
    print(f"  Day 1:  ${predictions[0]:.2f} ({(predictions[0]/last_price-1)*100:+.1f}%)")
    print(f"  Day 15: ${predictions[14]:.2f} ({(predictions[14]/last_price-1)*100:+.1f}%)")
    print(f"  Day 30: ${predictions[29]:.2f} ({(predictions[29]/last_price-1)*100:+.1f}%)")
    
    # Check bounds
    min_pred = min(predictions)
    max_pred = max(predictions)
    lower_bound = last_price * 0.85
    upper_bound = last_price * 1.15
    
    if min_pred >= lower_bound and max_pred <= upper_bound:
        print(f"\n✓ PASS: All predictions within ±15% range [${lower_bound:.2f}, ${upper_bound:.2f}]")
    else:
        print(f"\n⚠ WARNING: Some predictions outside ±15% range")
        print(f"  Min: ${min_pred:.2f}, Max: ${max_pred:.2f}")


def test_summary():
    """Print summary of all tests."""
    print("\n" + "=" * 60)
    print("IMPLEMENTATION SUMMARY")
    print("=" * 60)
    print("""
Changes made to fix the two issues:

1. TRADING DAYS FIX (src/api/app.py):
   - Added pandas_market_calendars integration
   - NYSE calendar used for stocks (excludes MLK Day, Presidents Day, etc.)
   - Crypto (BTC-USD) uses 24/7 calendar
   - Fallback to simple weekday check if calendar fails

2. PREDICTION QUALITY FIX (src/api/app.py):
   
   a) Deep Learning Models (LSTM, BiLSTM, LSTM-GRU, RNN, ANN):
      - Added momentum-based drift correction
      - Model weight decays exponentially (70% -> 26% over 30 steps)
      - Constrained to ±15% range from last actual price
   
   b) Time Series Models (ARIMA, SARIMA, Prophet, Exponential Smoothing):
      - Prophet now uses specific trading days instead of calendar days
      - Other time series models use native multi-step forecasting
   
   c) ML Models (Random Forest, Decision Tree, Linear Regression):
      - New predict_multi_step_ml function with rolling feature update
      - Features updated based on predictions for each step
      - Constrained to ±15% range

3. CONFIGURATION (src/config.py):
   - Added CRYPTO_TICKERS list for calendar logic

Files modified:
   - pyproject.toml (pandas-market-calendars already included)
   - src/config.py (added CRYPTO_TICKERS)
   - src/api/app.py (main prediction logic fixes)
""")


if __name__ == "__main__":
    test_trading_days()
    test_lstm_prediction()
    test_summary()

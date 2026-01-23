"""
Test script to verify predict_next() works for all models.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.training.trainer import ModelTrainer

def test_predict():
    t = ModelTrainer(verbose=False)
    
    # Test với một vài model khác nhau
    test_models = ['LSTM', 'RNN', 'BiLSTM', 'Random Forest', 'Decision Tree']
    
    print('Testing predict_next() for multiple models...')
    print('='*60)
    
    success = 0
    failed = 0
    no_model = 0
    
    for model_name in test_models:
        try:
            result = t.predict_horizon('AAPL', model_name, horizon=1)
            print(f'OK {model_name}: ${result["prediction"]:.2f}')
            success += 1
        except FileNotFoundError as e:
            print(f'SKIP {model_name}: No saved model found')
            no_model += 1
        except Exception as e:
            print(f'FAIL {model_name}: ERROR - {e}')
            failed += 1
    
    print('='*60)
    print(f'Results: {success} OK, {failed} FAIL, {no_model} SKIP (no model)')
    
    return failed == 0

if __name__ == "__main__":
    success = test_predict()
    sys.exit(0 if success else 1)

#!/usr/bin/env python
"""
Train All Script
Script tự động train tất cả models cho tất cả tickers.

Usage:
    # Train tất cả
    uv run python scripts/train_all.py
    
    # Train một ticker cụ thể
    uv run python scripts/train_all.py --ticker AAPL
    
    # Train một model cụ thể  
    uv run python scripts/train_all.py --model LSTM
    
    # Train một ticker với một model
    uv run python scripts/train_all.py --ticker AAPL --model LSTM
    
    # Train nhiều tickers và models
    uv run python scripts/train_all.py --tickers AAPL,MSFT,NVDA --models LSTM,ARIMA
    
    # Skip models bị lỗi
    uv run python scripts/train_all.py --skip-errors
    
    # Chỉ train ML models (nhanh hơn)
    uv run python scripts/train_all.py --ml-only
    
    # Chỉ train Deep Learning models
    uv run python scripts/train_all.py --dl-only
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import TICKERS, MODEL_NAMES
from data.downloader import DataDownloader
from training.trainer import (
    ModelTrainer, 
    MODEL_REGISTRY,
    DEEP_LEARNING_MODELS,
    TIME_SERIES_MODELS,
    ML_MODELS
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train stock prediction models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--ticker', '-t',
        type=str,
        help='Single ticker to train (e.g., AAPL)'
    )
    
    parser.add_argument(
        '--tickers',
        type=str,
        help='Comma-separated list of tickers (e.g., AAPL,MSFT,NVDA)'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        help='Single model to train (e.g., LSTM)'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        help='Comma-separated list of models (e.g., LSTM,ARIMA)'
    )
    
    parser.add_argument(
        '--skip-errors',
        action='store_true',
        help='Skip models that fail and continue with others'
    )

    parser.add_argument(
        '--no-update-data',
        action='store_true',
        help='Do not download latest CSV data before training'
    )
    
    parser.add_argument(
        '--ml-only',
        action='store_true',
        help='Train only ML models (Random Forest, Decision Tree, Linear Regression)'
    )
    
    parser.add_argument(
        '--dl-only',
        action='store_true',
        help='Train only Deep Learning models (LSTM, RNN, etc.)'
    )
    
    parser.add_argument(
        '--ts-only',
        action='store_true',
        help='Train only Time Series models (ARIMA, Prophet, etc.)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save models and results'
    )
    
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List all available models'
    )
    
    parser.add_argument(
        '--list-tickers',
        action='store_true',
        help='List all available tickers'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # List commands
    if args.list_models:
        print("\nAvailable models:")
        print("-" * 40)
        print("\nDeep Learning:")
        for m in DEEP_LEARNING_MODELS:
            print(f"  - {m}")
        print("\nTime Series:")
        for m in TIME_SERIES_MODELS:
            print(f"  - {m}")
        print("\nMachine Learning:")
        for m in ML_MODELS:
            print(f"  - {m}")
        return
    
    if args.list_tickers:
        print("\nAvailable tickers:")
        print("-" * 40)
        for t in TICKERS:
            print(f"  - {t}")
        return
    
    # Determine tickers
    if args.ticker:
        tickers = [args.ticker.upper()]
    elif args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(',')]
    else:
        tickers = TICKERS
    
    # Validate tickers
    for t in tickers:
        if t not in TICKERS:
            print(f"Warning: {t} not in default tickers list. Will try anyway.")
    
    # Determine models
    if args.model:
        models = [args.model]
    elif args.models:
        models = [m.strip() for m in args.models.split(',')]
    elif args.ml_only:
        models = ML_MODELS
    elif args.dl_only:
        models = DEEP_LEARNING_MODELS
    elif args.ts_only:
        models = TIME_SERIES_MODELS
    else:
        models = list(MODEL_REGISTRY.keys())
    
    # Validate models
    for m in models:
        if m not in MODEL_REGISTRY:
            print(f"Error: Unknown model '{m}'")
            print(f"Available: {list(MODEL_REGISTRY.keys())}")
            return
    
    # Print summary
    print("\n" + "=" * 60)
    print("STOCK PREDICTION MODEL TRAINING")
    print("=" * 60)
    print(f"Tickers ({len(tickers)}): {', '.join(tickers)}")
    print(f"Models ({len(models)}): {', '.join(models)}")
    print(f"Total training runs: {len(tickers) * len(models)}")
    print(f"Skip errors: {args.skip_errors}")
    print(f"Update data first: {not args.no_update_data}")
    print(f"Save results: {not args.no_save}")
    print("=" * 60)
    
    # Confirm
    if len(tickers) * len(models) > 5:
        response = input("\nProceed? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Train
    start_time = datetime.now()

    # Update data (download latest CSVs)
    if not args.no_update_data:
        print("\nUpdating latest data from yfinance...")
        downloader = DataDownloader()
        downloader.download_all(tickers=tickers, save=True)
    
    trainer = ModelTrainer(verbose=True)
    
    if len(tickers) == 1 and len(models) == 1:
        # Single training
        result = trainer.train_single(
            ticker=tickers[0],
            model_name=models[0],
            save_model=not args.no_save,
            save_results=not args.no_save
        )
    elif len(tickers) == 1:
        # All models for one ticker
        results = trainer.train_all_models(
            ticker=tickers[0],
            models=models,
            skip_on_error=args.skip_errors,
            save_model=not args.no_save,
            save_results=not args.no_save
        )
    elif len(models) == 1:
        # One model for all tickers
        results = trainer.train_all_tickers(
            model_name=models[0],
            tickers=tickers,
            skip_on_error=args.skip_errors,
            save_model=not args.no_save,
            save_results=not args.no_save
        )
    else:
        # All combinations
        results = trainer.train_all(
            tickers=tickers,
            models=models,
            skip_on_error=args.skip_errors,
            save_model=not args.no_save,
            save_results=not args.no_save
        )
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"Duration: {duration}")
    
    summary = trainer.get_summary()
    if not summary.empty:
        print(f"\nResults:")
        print(summary.to_string(index=False))
        
        # Stats
        success = (summary['status'] == 'success').sum()
        failed = (summary['status'] == 'failed').sum()
        print(f"\nSuccess: {success}, Failed: {failed}")


if __name__ == '__main__':
    main()

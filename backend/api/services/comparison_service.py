"""
Comparison Service Module.

Provides functionality for comparing model performance across tickers.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional

from backend.config import RESULTS_DIR, TICKERS, MODEL_NAMES
from backend.api.cache import (
    cached_leaderboard,
    cached_market_overview,
    cached_ticker_comparison,
)


class ComparisonService:
    """
    Service for aggregating and comparing model metrics.
    
    Scans the Result/ directory structure to collect metrics
    from all trained models and provides comparison data.
    """
    
    @staticmethod
    def get_latest_metrics_file(ticker: str, model: str) -> Optional[Path]:
        """
        Find the latest metrics JSON file for a ticker/model combination.
        
        Args:
            ticker: Stock ticker symbol.
            model: Model name.
            
        Returns:
            Path to the latest metrics file, or None if not found.
        """
        model_dir = RESULTS_DIR / ticker / model
        
        if not model_dir.exists():
            return None
        
        # Find all metrics JSON files
        metrics_files = list(model_dir.glob(f"{ticker}_{model}_metrics_*.json"))
        
        if not metrics_files:
            return None
        
        # Return the most recent one
        return max(metrics_files, key=lambda x: x.stat().st_mtime)
    
    @staticmethod
    def load_metrics(metrics_path: Path) -> Optional[Dict[str, Any]]:
        """
        Load metrics from a JSON file.
        
        Args:
            metrics_path: Path to the metrics JSON file.
            
        Returns:
            Dictionary containing metrics data, or None on error.
        """
        try:
            with open(metrics_path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    
    @classmethod
    @cached_ticker_comparison
    def get_ticker_comparison(cls, ticker: str) -> Dict[str, Any]:
        """
        Get comparison data for all models for a specific ticker.
        
        Args:
            ticker: Stock ticker symbol.
            
        Returns:
            Dictionary with comparison data for all models.
        """
        results = {
            "ticker": ticker,
            "models": [],
            "trained_count": 0,
            "total_count": len(MODEL_NAMES),
        }
        
        for model_name in MODEL_NAMES:
            model_data = {
                "model": model_name,
                "status": "not_trained",
                "metrics": None,
                "trained_at": None,
            }
            
            metrics_file = cls.get_latest_metrics_file(ticker, model_name)
            
            if metrics_file:
                metrics_data = cls.load_metrics(metrics_file)
                
                if metrics_data and "metrics" in metrics_data:
                    model_data["status"] = "trained"
                    model_data["metrics"] = {
                        "mse": metrics_data["metrics"].get("MSE"),
                        "rmse": metrics_data["metrics"].get("RMSE"),
                        "mae": metrics_data["metrics"].get("MAE"),
                        "mape": metrics_data["metrics"].get("MAPE"),
                        "r2": metrics_data["metrics"].get("R2"),
                    }
                    model_data["trained_at"] = metrics_data.get("timestamp")
                    results["trained_count"] += 1
        
            results["models"].append(model_data)
        
        # Sort by MAPE (lowest first), with not_trained at the end
        results["models"].sort(
            key=lambda x: (
                x["status"] != "trained",  # trained first
                x["metrics"]["mape"] if x["metrics"] else float("inf")
            )
        )
        
        return results
    
    @classmethod
    @cached_leaderboard
    def get_leaderboard(cls) -> Dict[str, Any]:
        """
        Get global leaderboard across all tickers and models.
        
        Returns:
            Dictionary with leaderboard data.
        """
        all_entries = []
        ticker_stats = {}
        model_stats = {}
        
        for ticker in TICKERS:
            ticker_stats[ticker] = {"trained": 0, "total": len(MODEL_NAMES)}
            
            for model_name in MODEL_NAMES:
                if model_name not in model_stats:
                    model_stats[model_name] = {"trained": 0, "total": len(TICKERS)}
                
                metrics_file = cls.get_latest_metrics_file(ticker, model_name)
                
                if metrics_file:
                    metrics_data = cls.load_metrics(metrics_file)
                    
                    if metrics_data and "metrics" in metrics_data:
                        ticker_stats[ticker]["trained"] += 1
                        model_stats[model_name]["trained"] += 1
                        
                        all_entries.append({
                            "ticker": ticker,
                            "model": model_name,
                            "mse": metrics_data["metrics"].get("MSE"),
                            "rmse": metrics_data["metrics"].get("RMSE"),
                            "mae": metrics_data["metrics"].get("MAE"),
                            "mape": metrics_data["metrics"].get("MAPE"),
                            "r2": metrics_data["metrics"].get("R2"),
                            "trained_at": metrics_data.get("timestamp"),
                        })
        
        # Sort by MAPE (lowest = best)
        all_entries.sort(key=lambda x: x["mape"] if x["mape"] else float("inf"))
        
        # Calculate summary stats
        total_combinations = len(TICKERS) * len(MODEL_NAMES)
        trained_count = len(all_entries)
        
        avg_mape = None
        avg_r2 = None
        if all_entries:
            mape_values = [e["mape"] for e in all_entries if e["mape"] is not None]
            r2_values = [e["r2"] for e in all_entries if e["r2"] is not None]
            if mape_values:
                avg_mape = sum(mape_values) / len(mape_values)
            if r2_values:
                avg_r2 = sum(r2_values) / len(r2_values)
        
        # Best model per ticker
        best_per_ticker = {}
        for ticker in TICKERS:
            ticker_entries = [e for e in all_entries if e["ticker"] == ticker]
            if ticker_entries:
                best_per_ticker[ticker] = ticker_entries[0]  # Already sorted by MAPE
        
        # Best ticker per model
        best_per_model = {}
        for model in MODEL_NAMES:
            model_entries = [e for e in all_entries if e["model"] == model]
            if model_entries:
                model_entries.sort(key=lambda x: x["mape"] if x["mape"] else float("inf"))
                best_per_model[model] = model_entries[0]
        
        return {
            "summary": {
                "total_combinations": total_combinations,
                "trained_count": trained_count,
                "not_trained_count": total_combinations - trained_count,
                "coverage_percent": round(trained_count / total_combinations * 100, 1),
                "avg_mape": round(avg_mape, 2) if avg_mape else None,
                "avg_r2": round(avg_r2, 4) if avg_r2 else None,
            },
            "leaderboard": all_entries[:20],  # Top 20
            "ticker_stats": ticker_stats,
            "model_stats": model_stats,
            "best_per_ticker": best_per_ticker,
            "best_per_model": best_per_model,
        }
    
    @classmethod
    @cached_market_overview
    def get_market_overview(cls) -> List[Dict[str, Any]]:
        """
        Get latest price data for all tickers (from CSV files).
        
        Returns:
            List of ticker data with latest prices.
        """
        import pandas as pd
        from backend.config import DATA_DIR
        
        overview = []
        
        for ticker in TICKERS:
            data = {
                "ticker": ticker,
                "latest_price": None,
                "previous_price": None,
                "change": None,
                "change_percent": None,
                "date": None,
            }
            
            csv_path = DATA_DIR / f"{ticker}.csv"
            
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path, skiprows=2)
                    df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
                    df["Date"] = pd.to_datetime(df["Date"])
                    df = df.sort_values("Date")
                    
                    if len(df) >= 2:
                        latest = df.iloc[-1]
                        previous = df.iloc[-2]
                        
                        data["latest_price"] = round(float(latest["Close"]), 2)
                        data["previous_price"] = round(float(previous["Close"]), 2)
                        data["change"] = round(data["latest_price"] - data["previous_price"], 2)
                        data["change_percent"] = round(
                            (data["change"] / data["previous_price"]) * 100, 2
                        )
                        data["date"] = str(latest["Date"].date())
                except Exception:
                    pass
            
            overview.append(data)
        
        return overview

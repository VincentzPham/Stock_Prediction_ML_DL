"""
Export utilities for frontend.

Provides helper functions for exporting data as CSV and JSON.
"""

import json
import pandas as pd
import streamlit as st
from typing import Any, Dict, List, Union
from datetime import datetime


def export_to_csv(
    data: Union[pd.DataFrame, List[Dict[str, Any]]],
    filename_prefix: str,
    label: str = "Download CSV"
) -> None:
    """
    Create a CSV download button for data.
    
    Args:
        data: DataFrame or list of dictionaries to export.
        filename_prefix: Prefix for the downloaded filename.
        label: Button label text.
    """
    if isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data
    
    if df.empty:
        st.warning("No data available to export.")
        return
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.csv"
    
    # Convert to CSV
    csv_data = df.to_csv(index=False)
    
    st.download_button(
        label=f"📥 {label}",
        data=csv_data,
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


def export_to_json(
    data: Union[Dict[str, Any], List[Dict[str, Any]]],
    filename_prefix: str,
    label: str = "Download JSON"
) -> None:
    """
    Create a JSON download button for data.
    
    Args:
        data: Dictionary or list to export.
        filename_prefix: Prefix for the downloaded filename.
        label: Button label text.
    """
    if not data:
        st.warning("No data available to export.")
        return
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.json"
    
    # Convert to JSON with formatting
    json_data = json.dumps(data, indent=2, default=str)
    
    st.download_button(
        label=f"📥 {label}",
        data=json_data,
        file_name=filename,
        mime="application/json",
        use_container_width=True,
    )


def export_predictions(
    predictions: List[Dict[str, Any]],
    ticker: str,
    model: str,
    format_type: str = "csv"
) -> None:
    """
    Export prediction results.
    
    Args:
        predictions: List of prediction dictionaries.
        ticker: Stock ticker symbol.
        model: Model name used for prediction.
        format_type: Export format ('csv' or 'json').
    """
    if not predictions:
        st.warning("No predictions to export.")
        return
    
    filename_prefix = f"{ticker}_{model}_predictions"
    
    if format_type == "csv":
        export_to_csv(predictions, filename_prefix, "Download Predictions (CSV)")
    else:
        export_data = {
            "ticker": ticker,
            "model": model,
            "generated_at": datetime.now().isoformat(),
            "predictions": predictions,
        }
        export_to_json(export_data, filename_prefix, "Download Predictions (JSON)")


def export_comparison(
    comparison_data: Dict[str, Any],
    ticker: str,
    format_type: str = "csv"
) -> None:
    """
    Export model comparison results.
    
    Args:
        comparison_data: Comparison data dictionary.
        ticker: Stock ticker symbol.
        format_type: Export format ('csv' or 'json').
    """
    if not comparison_data or "models" not in comparison_data:
        st.warning("No comparison data to export.")
        return
    
    filename_prefix = f"{ticker}_model_comparison"
    
    if format_type == "csv":
        # Flatten models data for CSV
        models = comparison_data.get("models", [])
        rows = []
        for model_data in models:
            row = {
                "Model": model_data["model"],
                "Status": model_data["status"],
            }
            if model_data["metrics"]:
                row["MSE"] = model_data["metrics"].get("mse")
                row["RMSE"] = model_data["metrics"].get("rmse")
                row["MAE"] = model_data["metrics"].get("mae")
                row["MAPE"] = model_data["metrics"].get("mape")
                row["R2"] = model_data["metrics"].get("r2")
            rows.append(row)
        
        export_to_csv(rows, filename_prefix, "Download Comparison (CSV)")
    else:
        export_to_json(comparison_data, filename_prefix, "Download Comparison (JSON)")


def export_leaderboard(
    leaderboard_data: Dict[str, Any],
    format_type: str = "csv"
) -> None:
    """
    Export global leaderboard data.
    
    Args:
        leaderboard_data: Leaderboard data dictionary.
        format_type: Export format ('csv' or 'json').
    """
    if not leaderboard_data or "leaderboard" not in leaderboard_data:
        st.warning("No leaderboard data to export.")
        return
    
    filename_prefix = "model_leaderboard"
    
    if format_type == "csv":
        leaderboard = leaderboard_data.get("leaderboard", [])
        export_to_csv(leaderboard, filename_prefix, "Download Leaderboard (CSV)")
    else:
        export_to_json(leaderboard_data, filename_prefix, "Download Leaderboard (JSON)")

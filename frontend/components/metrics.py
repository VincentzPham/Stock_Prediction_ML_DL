"""
Metrics Display Components Module.

Provides functions for displaying model evaluation metrics.
"""

import streamlit as st
from typing import Dict, Any

from frontend.config import CHART_COLORS


def display_metrics_cards(metrics: Dict[str, Any]) -> None:
    """
    Display metrics in styled visual cards.
    
    Args:
        metrics: Dictionary containing metric values (mse, rmse, mae, mape, r2).
    """
    if not metrics or all(
        v is None for k, v in metrics.items() if k not in ["ticker", "model"]
    ):
        st.info(
            "No metrics available for this model. Train the model to generate metrics."
        )
        return
    
    # Build metrics HTML cards
    metrics_html = '<div class="metrics-grid">'
    
    if metrics.get("mse") is not None:
        metrics_html += f"""
        <div class="metric-item">
            <div class="metric-item-value">{metrics['mse']:.2f}</div>
            <div class="metric-item-label">MSE</div>
        </div>"""
    
    if metrics.get("rmse") is not None:
        metrics_html += f"""
        <div class="metric-item">
            <div class="metric-item-value">{metrics['rmse']:.2f}</div>
            <div class="metric-item-label">RMSE</div>
        </div>"""
    
    if metrics.get("mae") is not None:
        metrics_html += f"""
        <div class="metric-item">
            <div class="metric-item-value">{metrics['mae']:.2f}</div>
            <div class="metric-item-label">MAE</div>
        </div>"""
    
    if metrics.get("mape") is not None:
        metrics_html += f"""
        <div class="metric-item">
            <div class="metric-item-value">{metrics['mape']:.2f}%</div>
            <div class="metric-item-label">MAPE</div>
        </div>"""
    
    if metrics.get("r2") is not None:
        r2_value = metrics["r2"]
        r2_color = _get_r2_color(r2_value)
        metrics_html += f"""
        <div class="metric-item" style="border-color: {r2_color};">
            <div class="metric-item-value" style="color: {r2_color};">{r2_value:.4f}</div>
            <div class="metric-item-label">R2 Score</div>
        </div>"""
    
    metrics_html += "</div>"
    
    st.markdown(metrics_html, unsafe_allow_html=True)


def _get_r2_color(r2_value: float) -> str:
    """
    Get color for R2 score based on value.
    
    Args:
        r2_value: R2 score value.
        
    Returns:
        Hex color string.
    """
    if r2_value > 0.9:
        return CHART_COLORS["success"]
    elif r2_value > 0.7:
        return CHART_COLORS["warning"]
    else:
        return CHART_COLORS["error"]


def display_price_card(
    ticker: str,
    date: str,
    price: float
) -> None:
    """
    Display current price in a styled card.
    
    Args:
        ticker: Stock ticker symbol.
        date: Price date.
        price: Current price value.
    """
    st.markdown(
        f"""
        <div class="panel panel-tight">
            <div class="panel-label">{ticker} - {date}</div>
            <div class="panel-value">${price:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

"""
Chart Components Module.

Provides functions for creating Plotly charts.
"""

import plotly.graph_objects as go
import pandas as pd
from typing import List, Dict, Any

from frontend.config import CHART_COLORS


def create_historical_chart(
    historical_df: pd.DataFrame,
    ticker: str,
    days: int = 90
) -> go.Figure:
    """
    Create a chart showing historical price data.
    
    Args:
        historical_df: DataFrame with columns ['date', 'actual'].
        ticker: Stock ticker symbol for title.
        days: Number of days shown (for title).
        
    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()
    
    if not historical_df.empty:
        fig.add_trace(
            go.Scatter(
                x=historical_df["date"],
                y=historical_df["actual"],
                mode="lines",
                name="Historical Price",
                line=dict(color=CHART_COLORS["primary"], width=2),
                fill="tozeroy",
                fillcolor="rgba(15, 118, 110, 0.12)",
            )
        )
    
    fig.update_layout(
        title=dict(
            text=f"{ticker} Stock Price - Last {days} Days",
            font=dict(size=18, color=CHART_COLORS["text"]),
        ),
        font=dict(family="Space Grotesk", color=CHART_COLORS["text"]),
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        plot_bgcolor=CHART_COLORS["background"],
        paper_bgcolor=CHART_COLORS["background"],
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor=CHART_COLORS["grid"]),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor=CHART_COLORS["grid"],
            tickprefix="$"
        ),
        margin=dict(l=60, r=40, t=80, b=60),
        height=400,
    )
    
    return fig


def create_prediction_chart(
    historical_df: pd.DataFrame,
    predictions: List[Dict[str, Any]],
    ticker: str
) -> go.Figure:
    """
    Create a chart with historical data and predictions.
    
    Args:
        historical_df: DataFrame with columns ['date', 'actual'].
        predictions: List of prediction dictionaries with 'date' and 'predicted_price'.
        ticker: Stock ticker symbol for title.
        
    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()
    
    # Historical data
    if not historical_df.empty:
        fig.add_trace(
            go.Scatter(
                x=historical_df["date"],
                y=historical_df["actual"],
                mode="lines",
                name="Historical Price",
                line=dict(color=CHART_COLORS["primary"], width=2),
                hovertemplate="Date: %{x}<br>Price: $%{y:.2f}<extra></extra>",
            )
        )
    
    # Predictions
    if predictions:
        pred_dates = [p["date"] for p in predictions]
        pred_values = [p["predicted_price"] for p in predictions]
        
        # Connect last historical point to first prediction
        if not historical_df.empty:
            last_hist_date = historical_df["date"].iloc[-1]
            last_hist_value = historical_df["actual"].iloc[-1]
            
            # Add connecting line
            fig.add_trace(
                go.Scatter(
                    x=[last_hist_date, pred_dates[0]],
                    y=[last_hist_value, pred_values[0]],
                    mode="lines",
                    name="Transition",
                    line=dict(color=CHART_COLORS["muted"], width=2, dash="dot"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        
        # Predicted line
        fig.add_trace(
            go.Scatter(
                x=pred_dates,
                y=pred_values,
                mode="lines+markers",
                name="Predicted Price",
                line=dict(color=CHART_COLORS["secondary"], width=2, dash="dash"),
                marker=dict(size=8, color=CHART_COLORS["secondary"]),
                hovertemplate="Date: %{x}<br>Predicted: $%{y:.2f}<extra></extra>",
            )
        )
    
    # Layout
    fig.update_layout(
        title=dict(
            text=f"{ticker} Stock Price - Historical & Predicted",
            font=dict(size=18, color=CHART_COLORS["text"]),
        ),
        font=dict(family="Space Grotesk", color=CHART_COLORS["text"]),
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor=CHART_COLORS["background"],
        paper_bgcolor=CHART_COLORS["background"],
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor=CHART_COLORS["grid"]),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor=CHART_COLORS["grid"],
            tickprefix="$"
        ),
        margin=dict(l=60, r=40, t=80, b=60),
        height=450,
    )
    
    return fig

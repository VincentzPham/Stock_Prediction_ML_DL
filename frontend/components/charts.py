"""
Chart Components Module.

Provides functions for creating Plotly charts with enhanced styling and interactivity.
"""

import plotly.graph_objects as go
import pandas as pd
from typing import List, Dict, Any

from frontend.config import CHART_COLORS


# Common chart configuration for consistent styling
CHART_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {
        "format": "png",
        "filename": "chart",
        "height": 600,
        "width": 1000,
        "scale": 2,
    },
}


def _get_common_layout(title: str, height: int = 400) -> dict:
    """
    Get common layout configuration for charts.
    
    Args:
        title: Chart title.
        height: Chart height in pixels.
        
    Returns:
        Dictionary with layout configuration.
    """
    return dict(
        title=dict(
            text=title,
            font=dict(size=18, color=CHART_COLORS["text"], family="Space Grotesk"),
            x=0,
            xanchor="left",
        ),
        font=dict(family="Space Grotesk", color=CHART_COLORS["text"]),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="white",
            font_size=13,
            font_family="Space Grotesk",
            bordercolor=CHART_COLORS["grid"],
        ),
        plot_bgcolor=CHART_COLORS["background"],
        paper_bgcolor=CHART_COLORS["background"],
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor=CHART_COLORS["grid"],
            showline=True,
            linewidth=1,
            linecolor=CHART_COLORS["grid"],
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor=CHART_COLORS["grid"],
            tickprefix="$",
            showline=True,
            linewidth=1,
            linecolor=CHART_COLORS["grid"],
        ),
        margin=dict(l=60, r=40, t=60, b=50),
        height=height,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=CHART_COLORS["grid"],
            borderwidth=1,
        ),
        # Animation settings
        transition=dict(
            duration=500,
            easing="cubic-in-out",
        ),
    )


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
                line=dict(
                    color=CHART_COLORS["primary"],
                    width=2.5,
                    shape="spline",  # Smooth curve
                ),
                fill="tozeroy",
                fillcolor="rgba(15, 118, 110, 0.08)",
                hovertemplate="<b>%{x|%b %d, %Y}</b><br>Price: $%{y:,.2f}<extra></extra>",
            )
        )
    
    layout = _get_common_layout(f"{ticker} Stock Price - Last {days} Days", height=400)
    layout["xaxis"]["title"] = "Date"
    layout["yaxis"]["title"] = "Price (USD)"
    
    # Add range slider for zoom functionality
    layout["xaxis"]["rangeslider"] = dict(
        visible=True,
        thickness=0.05,
        bgcolor=CHART_COLORS["grid"],
    )
    layout["height"] = 450  # Extra height for range slider
    
    fig.update_layout(**layout)
    
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
                name="Historical",
                line=dict(
                    color=CHART_COLORS["primary"],
                    width=2.5,
                    shape="spline",
                ),
                hovertemplate="<b>%{x|%b %d, %Y}</b><br>Price: $%{y:,.2f}<extra></extra>",
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
                    line=dict(
                        color=CHART_COLORS["muted"],
                        width=2,
                        dash="dot",
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        
        # Prediction confidence band (visual only - simulated ±5%)
        upper_band = [v * 1.05 for v in pred_values]
        lower_band = [v * 0.95 for v in pred_values]
        
        # Upper bound (invisible, for fill)
        fig.add_trace(
            go.Scatter(
                x=pred_dates,
                y=upper_band,
                mode="lines",
                name="Upper Bound",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        
        # Lower bound with fill
        fig.add_trace(
            go.Scatter(
                x=pred_dates,
                y=lower_band,
                mode="lines",
                name="Confidence Band",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(197, 139, 42, 0.15)",
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
                name="Predicted",
                line=dict(
                    color=CHART_COLORS["secondary"],
                    width=2.5,
                    dash="dash",
                    shape="spline",
                ),
                marker=dict(
                    size=8,
                    color=CHART_COLORS["secondary"],
                    line=dict(width=2, color="white"),
                ),
                hovertemplate="<b>%{x|%b %d, %Y}</b><br>"
                              "Predicted: $%{y:,.2f}<extra></extra>",
            )
        )
    
    # Layout
    layout = _get_common_layout(f"{ticker} - Historical & Predicted", height=480)
    layout["xaxis"]["title"] = "Date"
    layout["yaxis"]["title"] = "Price (USD)"
    
    fig.update_layout(**layout)
    
    return fig


def get_chart_config() -> dict:
    """
    Get Plotly chart configuration for consistent interactivity.
    
    Returns:
        Dictionary with chart config options.
    """
    return CHART_CONFIG

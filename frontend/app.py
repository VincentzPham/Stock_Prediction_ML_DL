"""
Stock Price Prediction - Frontend UI
Professional Streamlit application for stock price prediction.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import os
from pathlib import Path

from backend.config import TICKERS, MODEL_NAMES

# API URL - support Docker environment variable
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Horizon options
HORIZON_OPTIONS = {
    "1 Days": 1,
    "3 Days": 3,
    "7 Days": 7,
    "14 Days": 14,
    "30 Days": 30,
    "60 Days": 60,
}

# Page config
st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for professional look
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@400;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    :root {
        --font-sans: 'Space Grotesk', 'Segoe UI', sans-serif;
        --font-serif: 'Fraunces', Georgia, serif;
        --bg: #f4f1ec;
        --bg-2: #faf7f2;
        --panel: #ffffff;
        --panel-soft: #f8f6f2;
        --line: #e6dfd6;
        --ink: #1b2430;
        --muted: #5f6b7a;
        --accent: #0f766e;
        --accent-strong: #0b5f59;
        --accent-warm: #c58b2a;
        --shadow-sm: 0 6px 18px rgba(15, 23, 42, 0.08);
        --shadow-lg: 0 20px 50px rgba(15, 23, 42, 0.15);
        --radius-lg: 18px;
        --radius-md: 12px;
    }
    
    .stApp {
        font-family: var(--font-sans);
        color: var(--ink);
        background:
            radial-gradient(900px circle at 5% -10%, rgba(15, 118, 110, 0.18), transparent 55%),
            radial-gradient(800px circle at 110% 10%, rgba(197, 139, 42, 0.18), transparent 60%),
            linear-gradient(180deg, var(--bg) 0%, var(--bg-2) 100%);
    }
    
    .main .block-container {
        padding: 2.2rem 2.8rem;
        max-width: 1400px;
    }
    
    h1 {
        font-family: var(--font-serif);
        color: var(--ink);
        font-weight: 600;
        font-size: 2.6rem !important;
        letter-spacing: -0.02em;
        margin: 0;
    }
    
    h2, h3 {
        color: var(--ink);
        font-weight: 600;
        letter-spacing: -0.01em;
    }
    
    .stCaption, .stMarkdown, p {
        color: var(--muted);
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f2f2e 0%, #10262b 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p {
        color: rgba(236, 242, 241, 0.92);
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: #f8fbfa !important;
        -webkit-text-fill-color: #f8fbfa !important;
    }
    
    .hero {
        position: relative;
        border-radius: var(--radius-lg);
        padding: 1.6rem 1.9rem;
        border: 1px solid var(--line);
        background: linear-gradient(135deg, rgba(15, 118, 110, 0.12), rgba(197, 139, 42, 0.12));
        box-shadow: var(--shadow-sm);
        overflow: hidden;
        margin-bottom: 1.4rem;
    }
    
    .hero::after {
        content: "";
        position: absolute;
        inset: 0;
        background: radial-gradient(500px circle at 85% 15%, rgba(255, 255, 255, 0.6), transparent 60%);
        opacity: 0.6;
        pointer-events: none;
    }
    
    .hero-content {
        position: relative;
        z-index: 1;
    }
    
    .hero-subtitle {
        font-size: 1.05rem;
        margin: 0.4rem 0 1rem 0;
        color: var(--muted);
        max-width: 720px;
    }
    
    .hero-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    
    .chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: rgba(15, 118, 110, 0.14);
        color: var(--accent);
        border: 1px solid rgba(15, 118, 110, 0.28);
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    
    .panel {
        background: var(--panel);
        border-radius: var(--radius-md);
        border: 1px solid var(--line);
        box-shadow: var(--shadow-sm);
        padding: 1.2rem 1.4rem;
    }
    
    .panel-tight {
        padding: 1.1rem 1.25rem;
    }
    
    .panel-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--muted);
        margin-bottom: 0.35rem;
        font-weight: 600;
    }
    
    .panel-value {
        font-size: 2.1rem;
        font-weight: 600;
        color: var(--ink);
        margin: 0;
    }
    
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .metric-item {
        background: var(--panel);
        padding: 1.1rem 1.2rem;
        border-radius: var(--radius-md);
        text-align: center;
        border: 1px solid var(--line);
        transition: all 0.3s ease;
        box-shadow: var(--shadow-sm);
    }
    
    .metric-item:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
    }
    
    .metric-item-value {
        font-size: 1.6rem;
        font-weight: 600;
        color: var(--ink);
        margin-bottom: 0.25rem;
    }
    
    .metric-item-label {
        font-size: 0.7rem;
        color: var(--muted);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    
    .stDataFrame {
        border-radius: var(--radius-md);
        overflow: hidden;
        box-shadow: var(--shadow-sm);
    }
    
    .dataframe {
        font-size: 0.875rem;
        border-collapse: separate;
        border-spacing: 0;
    }
    
    .dataframe th {
        background: #123b3a;
        color: white !important;
        font-weight: 600;
        padding: 0.9rem 1rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.72rem;
    }
    
    .dataframe td {
        padding: 0.8rem 1rem;
        border-bottom: 1px solid #f0e9df;
    }
    
    .dataframe tr:hover td {
        background: #fbf8f3;
    }
    
    div[data-testid="stTable"] table {
        border-radius: var(--radius-md);
        overflow: hidden;
        border: 1px solid var(--line);
        box-shadow: var(--shadow-sm);
    }
    
    div[data-testid="stMetric"] {
        background: var(--panel);
        border-radius: var(--radius-md);
        border: 1px solid var(--line);
        padding: 1rem 1.1rem;
        box-shadow: var(--shadow-sm);
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--ink);
        font-weight: 600;
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
        color: var(--muted);
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    
    .stSuccess, .stError, .stInfo {
        border-radius: var(--radius-md);
        border: 1px solid var(--line);
        box-shadow: var(--shadow-sm);
    }
    
    .stSuccess {
        background: rgba(15, 118, 110, 0.08);
        border-color: rgba(15, 118, 110, 0.3);
    }
    
    .stError {
        background: rgba(180, 35, 24, 0.08);
        border-color: rgba(180, 35, 24, 0.28);
    }
    
    .stInfo {
        background: rgba(197, 139, 42, 0.08);
        border-color: rgba(197, 139, 42, 0.3);
    }
    
    .stButton > button {
        width: 100%;
        border-radius: var(--radius-md);
        font-weight: 600;
        padding: 0.85rem 1.4rem;
        font-size: 0.95rem;
        background: linear-gradient(135deg, var(--accent), var(--accent-strong));
        border: none;
        color: white;
        box-shadow: 0 12px 30px rgba(15, 118, 110, 0.25);
        transition: all 0.3s ease;
        letter-spacing: 0.02em;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 16px 32px rgba(15, 118, 110, 0.3);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    .stSelectbox > div > div {
        border-radius: 10px;
        border: 1.5px solid var(--line);
        background: var(--panel);
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: rgba(15, 118, 110, 0.7);
        box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.12);
    }
    
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #e3d9cd, transparent);
    }
    
    .stSpinner > div {
        border-color: var(--accent) transparent var(--accent) transparent;
    }
    
    footer {
        background: linear-gradient(180deg, transparent, rgba(0, 0, 0, 0.03));
        padding-top: 2rem;
    }
    
    @keyframes rise {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-in {
        animation: rise 0.6s ease-out forwards;
    }
    
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1.2rem;
        }
        
        h1 {
            font-size: 1.9rem !important;
        }
        
        .hero {
            padding: 1.2rem 1.3rem;
        }
        
        .panel-value {
            font-size: 1.6rem;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)


def get_historical_data(ticker: str, days: int = 60) -> pd.DataFrame:
    """Fetch historical data from API."""
    try:
        response = requests.get(f"{API_URL}/historical/{ticker}?days={days}")
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data["data"])
            df["date"] = pd.to_datetime(df["date"])
            return df
    except Exception as e:
        st.error(f"Error fetching historical data: {e}")
    return pd.DataFrame()


def get_metrics(ticker: str, model: str) -> dict:
    """Fetch model metrics from API."""
    try:
        response = requests.get(f"{API_URL}/metrics/{ticker}/{model}")
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return {}


def get_latest_price(ticker: str) -> dict:
    """Fetch latest price from API."""
    try:
        response = requests.get(f"{API_URL}/latest-price/{ticker}")
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return {}


def create_prediction_chart(
    historical_df: pd.DataFrame, predictions: list, ticker: str
):
    """Create a professional chart with historical and predicted prices."""
    fig = go.Figure()

    # Historical data
    if not historical_df.empty:
        fig.add_trace(
            go.Scatter(
                x=historical_df["date"],
                y=historical_df["actual"],
                mode="lines",
                name="Historical Price",
                line=dict(color="#0f766e", width=2),
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
                    line=dict(color="#9aa6b2", width=2, dash="dot"),
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
                line=dict(color="#c58b2a", width=2, dash="dash"),
                marker=dict(size=8, color="#c58b2a"),
                hovertemplate="Date: %{x}<br>Predicted: $%{y:.2f}<extra></extra>",
            )
        )

    # Layout
    fig.update_layout(
        title=dict(
            text=f"{ticker} Stock Price - Historical & Predicted",
            font=dict(size=18, color="#1b2430"),
        ),
        font=dict(family="Space Grotesk", color="#1b2430"),
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor="#eee5db"),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor="#eee5db", tickprefix="$"),
        margin=dict(l=60, r=40, t=80, b=60),
        height=450,
    )

    return fig


def display_metrics_table(metrics: dict):
    """Display metrics in beautiful visual cards."""
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
        r2_color = (
            "#0f766e"
            if metrics["r2"] > 0.9
            else ("#c58b2a" if metrics["r2"] > 0.7 else "#b42318")
        )
        metrics_html += f"""
        <div class="metric-item" style="border-color: {r2_color};">
            <div class="metric-item-value" style="color: {r2_color};">{metrics['r2']:.4f}</div>
            <div class="metric-item-label">R2 Score</div>
        </div>"""

    metrics_html += "</div>"

    st.markdown(metrics_html, unsafe_allow_html=True)


def main():
    # Header
    st.markdown(
        """
    <div class="hero animate-in">
        <div class="hero-content">
            <h1>Stock Price Prediction</h1>
            <p class="hero-subtitle">
                Professional forecasting with machine learning and deep learning models.
                Explore historical context and generate multi-day price projections.
            </p>
            <div class="hero-chips">
                <span class="chip">ML and DL Models</span>
                <span class="chip">FastAPI Backend</span>
                <span class="chip">Streamlit UI</span>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        st.markdown("---")

        # Ticker selection
        ticker = st.selectbox(
            "Select Stock Ticker",
            options=TICKERS,
            index=0,
            help="Choose the stock ticker to predict",
        )

        st.markdown("")

        # Model selection
        model_name = st.selectbox(
            "Select Prediction Model",
            options=MODEL_NAMES,
            index=MODEL_NAMES.index("LSTM") if "LSTM" in MODEL_NAMES else 0,
            help="Choose the ML/DL model for prediction",
        )

        st.markdown("")

        # Horizon selection
        horizon_label = st.selectbox(
            "Forecast Horizon",
            options=list(HORIZON_OPTIONS.keys()),
            index=1,  # Default to 7 days
            help="Number of days to forecast",
        )
        horizon = HORIZON_OPTIONS[horizon_label]

        st.markdown("")
        st.markdown("---")

        # Predict button
        predict_clicked = st.button(
            "Generate Prediction", type="primary", use_container_width=True
        )

        st.markdown("---")

        # Model info
        st.markdown("**Model Information**")
        st.caption(f"Ticker: {ticker}")
        st.caption(f"Model: {model_name}")
        st.caption(f"Horizon: {horizon} days")

    # Main content area
    col_main, col_side = st.columns([3, 1])

    with col_side:
        # Latest price card
        latest = get_latest_price(ticker)
        if latest:
            st.markdown("### Current Price")
            st.markdown(
                f"""
            <div class="panel panel-tight">
                <div class="panel-label">{ticker} - {latest.get('date', 'N/A')}</div>
                <div class="panel-value">${latest.get('close', 0):.2f}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Price details
            st.markdown("**Price Details**")
            price_df = pd.DataFrame(
                {
                    "": ["Open", "High", "Low", "Volume"],
                    "Value": [
                        f"${latest.get('open', 0):.2f}",
                        f"${latest.get('high', 0):.2f}",
                        f"${latest.get('low', 0):.2f}",
                        f"{latest.get('volume', 0):,}",
                    ],
                }
            )
            st.table(price_df)

    with col_main:
        if predict_clicked:
            with st.spinner(
                f"Generating predictions for {ticker} using {model_name}..."
            ):
                try:
                    # Call API
                    payload = {
                        "ticker": ticker,
                        "model": model_name,
                        "horizon": horizon,
                    }
                    response = requests.post(
                        f"{API_URL}/predict", json=payload, timeout=120
                    )

                    if response.status_code == 200:
                        data = response.json()

                        # Success message
                        st.success(
                            f"Prediction generated successfully for {horizon} days ahead"
                        )

                        # Get historical data for chart
                        historical_df = get_historical_data(ticker, days=60)

                        # Chart
                        st.markdown("### Price Forecast Visualization")
                        fig = create_prediction_chart(
                            historical_df, data["predictions"], ticker
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Predictions table
                        st.markdown("### Predicted Prices")
                        pred_df = pd.DataFrame(data["predictions"])
                        pred_df.columns = ["Day", "Date", "Predicted Price ($)"]
                        pred_df["Predicted Price ($)"] = pred_df[
                            "Predicted Price ($)"
                        ].apply(lambda x: f"${x:.2f}")
                        st.dataframe(pred_df, use_container_width=True, hide_index=True)

                        # Summary stats
                        pred_values = [
                            p["predicted_price"] for p in data["predictions"]
                        ]
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric(
                                "Last Actual Price", f"${data['last_actual_price']:.2f}"
                            )
                        with col2:
                            st.metric("First Prediction", f"${pred_values[0]:.2f}")
                        with col3:
                            st.metric("Last Prediction", f"${pred_values[-1]:.2f}")
                        with col4:
                            change = (
                                (pred_values[-1] - data["last_actual_price"])
                                / data["last_actual_price"]
                            ) * 100
                            st.metric("Expected Change", f"{change:+.2f}%")

                        # Model metrics
                        st.markdown("---")
                        st.markdown("### Model Performance Metrics")
                        metrics = get_metrics(ticker, model_name)
                        display_metrics_table(metrics)

                        # Model info
                        st.caption(f"Model path: {data['model_path']}")

                    else:
                        error_detail = response.json().get("detail", response.text)
                        st.error(f"API Error: {error_detail}")

                except requests.exceptions.ConnectionError:
                    st.error(
                        "Cannot connect to API Backend. Please ensure the backend server is running."
                    )
                except requests.exceptions.Timeout:
                    st.error("Request timed out. The prediction is taking too long.")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
        else:
            # Default view - show historical chart
            st.markdown("### Historical Price Data")
            historical_df = get_historical_data(ticker, days=90)

            if not historical_df.empty:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=historical_df["date"],
                        y=historical_df["actual"],
                        mode="lines",
                        name="Historical Price",
                        line=dict(color="#0f766e", width=2),
                        fill="tozeroy",
                        fillcolor="rgba(15, 118, 110, 0.12)",
                    )
                )

                fig.update_layout(
                    title=dict(
                        text=f"{ticker} Stock Price - Last 90 Days",
                        font=dict(size=18, color="#1b2430"),
                    ),
                    font=dict(family="Space Grotesk", color="#1b2430"),
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    hovermode="x unified",
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    xaxis=dict(showgrid=True, gridwidth=1, gridcolor="#eee5db"),
                    yaxis=dict(
                        showgrid=True, gridwidth=1, gridcolor="#eee5db", tickprefix="$"
                    ),
                    margin=dict(l=60, r=40, t=80, b=60),
                    height=400,
                )

                st.plotly_chart(fig, use_container_width=True)

                # Quick stats
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        "Current Price", f"${historical_df['actual'].iloc[-1]:.2f}"
                    )
                with col2:
                    st.metric("90-Day High", f"${historical_df['actual'].max():.2f}")
                with col3:
                    st.metric("90-Day Low", f"${historical_df['actual'].min():.2f}")
                with col4:
                    change_90d = (
                        (
                            historical_df["actual"].iloc[-1]
                            - historical_df["actual"].iloc[0]
                        )
                        / historical_df["actual"].iloc[0]
                    ) * 100
                    st.metric("90-Day Change", f"{change_90d:+.2f}%")
            else:
                st.info(
                    "Select a ticker and click 'Generate Prediction' to view forecasts."
                )

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center; padding: 1rem 0;">
        <p style="color: #64748b; font-size: 0.875rem; margin: 0;">
            <strong>Stock Price Prediction System</strong> | Built with FastAPI and Streamlit
        </p>
        <p style="color: #94a3b8; font-size: 0.75rem; margin-top: 0.5rem;">
            Powered by LSTM, Random Forest, ARIMA and more
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

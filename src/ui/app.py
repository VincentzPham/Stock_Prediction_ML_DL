"""
Stock Price Prediction - Frontend UI
Professional Streamlit application for stock price prediction.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from config import TICKERS, MODEL_NAMES

# API URL - support Docker environment variable
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Horizon options
HORIZON_OPTIONS = {
    "3 Days": 3,
    "7 Days": 7,
    "14 Days": 14,
    "30 Days": 30,
    "60 Days": 60,
    "90 Days": 90
}

# Page config
st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Root variables */
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --success-gradient: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        --warning-gradient: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%);
        --dark-bg: #0f0f23;
        --card-bg: rgba(255, 255, 255, 0.95);
        --glass-bg: rgba(255, 255, 255, 0.1);
        --border-color: rgba(255, 255, 255, 0.18);
        --text-primary: #1a1a2e;
        --text-secondary: #4a5568;
        --shadow-sm: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
    }
    
    /* Global styles */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main container */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    /* Headers */
    h1 {
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    h2, h3 {
        color: var(--text-primary);
        font-weight: 600;
        letter-spacing: -0.01em;
    }
    
    /* Sidebar enhancement */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: rgba(255, 255, 255, 0.9);
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: white !important;
        -webkit-text-fill-color: white !important;
    }
    
    /* Glass card effect */
    .glass-card {
        background: var(--card-bg);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: var(--shadow-lg);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 30px 60px -12px rgba(0, 0, 0, 0.3);
    }
    
    /* Gradient metric card */
    .metric-card {
        background: var(--primary-gradient);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 10px 40px -10px rgba(102, 126, 234, 0.5);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 20px 50px -10px rgba(102, 126, 234, 0.6);
    }
    
    .metric-card-success {
        background: var(--success-gradient);
        box-shadow: 0 10px 40px -10px rgba(17, 153, 142, 0.5);
    }
    
    .metric-card-warning {
        background: var(--warning-gradient);
        box-shadow: 0 10px 40px -10px rgba(242, 153, 74, 0.5);
    }
    
    .metric-card-light {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
        box-shadow: var(--shadow-sm);
        transition: all 0.3s ease;
    }
    
    .metric-card-light:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.15);
    }
    
    .metric-value {
        font-size: 2.25rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.2;
        letter-spacing: -0.02em;
    }
    
    .metric-label {
        font-size: 0.875rem;
        opacity: 0.85;
        margin: 0;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Metrics table styling */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .metric-item {
        background: linear-gradient(145deg, #ffffff 0%, #f1f5f9 100%);
        padding: 1.25rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .metric-item:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px -8px rgba(0, 0, 0, 0.12);
    }
    
    .metric-item-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.25rem;
    }
    
    .metric-item-label {
        font-size: 0.75rem;
        color: var(--text-secondary);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Table styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: var(--shadow-sm);
    }
    
    .dataframe {
        font-size: 0.875rem;
        border-collapse: separate;
        border-spacing: 0;
    }
    
    .dataframe th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        font-weight: 600;
        padding: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.75rem;
    }
    
    .dataframe td {
        padding: 0.875rem 1rem;
        border-bottom: 1px solid #f1f5f9;
    }
    
    .dataframe tr:hover td {
        background: #f8fafc;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: linear-gradient(135deg, rgba(17, 153, 142, 0.1) 0%, rgba(56, 239, 125, 0.1) 100%);
        border: 1px solid rgba(56, 239, 125, 0.3);
        border-radius: 12px;
        padding: 1rem 1.5rem;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(245, 87, 108, 0.1) 0%, rgba(220, 53, 69, 0.1) 100%);
        border: 1px solid rgba(245, 87, 108, 0.3);
        border-radius: 12px;
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        font-weight: 600;
        padding: 0.875rem 1.5rem;
        font-size: 1rem;
        background: var(--primary-gradient);
        border: none;
        color: white;
        box-shadow: 0 8px 25px -8px rgba(102, 126, 234, 0.5);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px -8px rgba(102, 126, 234, 0.6);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        transition: border-color 0.3s ease;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #e2e8f0, transparent);
    }
    
    /* Spinner animation */
    .stSpinner > div {
        border-color: #667eea transparent #667eea transparent;
    }
    
    /* Caption styling */
    .stCaption {
        color: var(--text-secondary);
        font-size: 0.875rem;
    }
    
    /* Footer styling */
    footer {
        background: linear-gradient(180deg, transparent, rgba(0, 0, 0, 0.02));
        padding-top: 2rem;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.5s ease-out forwards;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem;
        }
        
        h1 {
            font-size: 1.75rem !important;
        }
        
        .metric-value {
            font-size: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)


def get_historical_data(ticker: str, days: int = 60) -> pd.DataFrame:
    """Fetch historical data from API."""
    try:
        response = requests.get(f"{API_URL}/historical/{ticker}?days={days}")
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data['data'])
            df['date'] = pd.to_datetime(df['date'])
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
    except Exception as e:
        pass
    return {}


def get_latest_price(ticker: str) -> dict:
    """Fetch latest price from API."""
    try:
        response = requests.get(f"{API_URL}/latest-price/{ticker}")
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        pass
    return {}


def create_prediction_chart(historical_df: pd.DataFrame, predictions: list, ticker: str):
    """Create a professional chart with historical and predicted prices."""
    fig = go.Figure()
    
    # Historical data
    if not historical_df.empty:
        fig.add_trace(go.Scatter(
            x=historical_df['date'],
            y=historical_df['actual'],
            mode='lines',
            name='Historical Price',
            line=dict(color='#3b82f6', width=2),
            hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))
    
    # Predictions
    if predictions:
        pred_dates = [p['date'] for p in predictions]
        pred_values = [p['predicted_price'] for p in predictions]
        
        # Connect last historical point to first prediction
        if not historical_df.empty:
            last_hist_date = historical_df['date'].iloc[-1]
            last_hist_value = historical_df['actual'].iloc[-1]
            
            # Add connecting line
            fig.add_trace(go.Scatter(
                x=[last_hist_date, pred_dates[0]],
                y=[last_hist_value, pred_values[0]],
                mode='lines',
                name='Transition',
                line=dict(color='#94a3b8', width=2, dash='dot'),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Predicted line
        fig.add_trace(go.Scatter(
            x=pred_dates,
            y=pred_values,
            mode='lines+markers',
            name='Predicted Price',
            line=dict(color='#10b981', width=2, dash='dash'),
            marker=dict(size=8, color='#10b981'),
            hovertemplate='Date: %{x}<br>Predicted: $%{y:.2f}<extra></extra>'
        ))
    
    # Layout
    fig.update_layout(
        title=dict(
            text=f'{ticker} Stock Price - Historical & Predicted',
            font=dict(size=18, color='#1f2937')
        ),
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='#f1f5f9'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='#f1f5f9',
            tickprefix='$'
        ),
        margin=dict(l=60, r=40, t=80, b=60),
        height=450
    )
    
    return fig


def display_metrics_table(metrics: dict):
    """Display metrics in beautiful visual cards."""
    if not metrics or all(v is None for k, v in metrics.items() if k not in ['ticker', 'model']):
        st.info("📊 No metrics available for this model. Train the model first to generate metrics.")
        return
    
    # Build metrics HTML cards
    metrics_html = '<div class="metrics-grid">'
    
    if metrics.get('mse') is not None:
        metrics_html += f'''
        <div class="metric-item">
            <div class="metric-item-value">{metrics['mse']:.2f}</div>
            <div class="metric-item-label">MSE</div>
        </div>'''
    
    if metrics.get('rmse') is not None:
        metrics_html += f'''
        <div class="metric-item">
            <div class="metric-item-value">{metrics['rmse']:.2f}</div>
            <div class="metric-item-label">RMSE</div>
        </div>'''
    
    if metrics.get('mae') is not None:
        metrics_html += f'''
        <div class="metric-item">
            <div class="metric-item-value">{metrics['mae']:.2f}</div>
            <div class="metric-item-label">MAE</div>
        </div>'''
    
    if metrics.get('mape') is not None:
        metrics_html += f'''
        <div class="metric-item">
            <div class="metric-item-value">{metrics['mape']:.2f}%</div>
            <div class="metric-item-label">MAPE</div>
        </div>'''
    
    if metrics.get('r2') is not None:
        r2_color = '#10b981' if metrics['r2'] > 0.9 else ('#f59e0b' if metrics['r2'] > 0.7 else '#ef4444')
        metrics_html += f'''
        <div class="metric-item" style="border-color: {r2_color};">
            <div class="metric-item-value" style="color: {r2_color};">{metrics['r2']:.4f}</div>
            <div class="metric-item-label">R² Score</div>
        </div>'''
    
    metrics_html += '</div>'
    
    st.markdown(metrics_html, unsafe_allow_html=True)


def main():
    # Header
    st.title("Stock Price Prediction")
    st.markdown("""<p style="font-size: 1.125rem; color: #64748b; margin-top: -0.5rem; margin-bottom: 1.5rem;">
        🚀 Predict future stock prices using Machine Learning and Deep Learning models
    </p>""", unsafe_allow_html=True)
    
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
            help="Choose the stock ticker to predict"
        )
        
        st.markdown("")
        
        # Model selection
        model_name = st.selectbox(
            "Select Prediction Model",
            options=MODEL_NAMES,
            index=MODEL_NAMES.index('LSTM') if 'LSTM' in MODEL_NAMES else 0,
            help="Choose the ML/DL model for prediction"
        )
        
        st.markdown("")
        
        # Horizon selection
        horizon_label = st.selectbox(
            "Forecast Horizon",
            options=list(HORIZON_OPTIONS.keys()),
            index=1,  # Default to 7 days
            help="Number of days to forecast"
        )
        horizon = HORIZON_OPTIONS[horizon_label]
        
        st.markdown("")
        st.markdown("---")
        
        # Predict button
        predict_clicked = st.button("Generate Prediction", type="primary", use_container_width=True)
        
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
            st.markdown(f"""
            <div class="metric-card-light">
                <p class="metric-label">{ticker} - {latest.get('date', 'N/A')}</p>
                <p class="metric-value">${latest.get('close', 0):.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Price details
            st.markdown("**Price Details**")
            price_df = pd.DataFrame({
                "": ["Open", "High", "Low", "Volume"],
                "Value": [
                    f"${latest.get('open', 0):.2f}",
                    f"${latest.get('high', 0):.2f}",
                    f"${latest.get('low', 0):.2f}",
                    f"{latest.get('volume', 0):,}"
                ]
            })
            st.table(price_df)
    
    with col_main:
        if predict_clicked:
            with st.spinner(f"Generating predictions for {ticker} using {model_name}..."):
                try:
                    # Call API
                    payload = {
                        "ticker": ticker,
                        "model": model_name,
                        "horizon": horizon
                    }
                    response = requests.post(f"{API_URL}/predict", json=payload, timeout=120)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Success message
                        st.success(f"Prediction generated successfully for {horizon} days ahead")
                        
                        # Get historical data for chart
                        historical_df = get_historical_data(ticker, days=60)
                        
                        # Chart
                        st.markdown("### Price Forecast Visualization")
                        fig = create_prediction_chart(historical_df, data['predictions'], ticker)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Predictions table
                        st.markdown("### Predicted Prices")
                        pred_df = pd.DataFrame(data['predictions'])
                        pred_df.columns = ['Day', 'Date', 'Predicted Price ($)']
                        pred_df['Predicted Price ($)'] = pred_df['Predicted Price ($)'].apply(lambda x: f"${x:.2f}")
                        st.dataframe(pred_df, use_container_width=True, hide_index=True)
                        
                        # Summary stats
                        pred_values = [p['predicted_price'] for p in data['predictions']]
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Last Actual Price", f"${data['last_actual_price']:.2f}")
                        with col2:
                            st.metric("First Prediction", f"${pred_values[0]:.2f}")
                        with col3:
                            st.metric("Last Prediction", f"${pred_values[-1]:.2f}")
                        with col4:
                            change = ((pred_values[-1] - data['last_actual_price']) / data['last_actual_price']) * 100
                            st.metric("Expected Change", f"{change:+.2f}%")
                        
                        # Model metrics
                        st.markdown("---")
                        st.markdown("### Model Performance Metrics")
                        metrics = get_metrics(ticker, model_name)
                        display_metrics_table(metrics)
                        
                        # Model info
                        st.caption(f"Model path: {data['model_path']}")
                        
                    else:
                        error_detail = response.json().get('detail', response.text)
                        st.error(f"API Error: {error_detail}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API Backend. Please ensure the backend server is running.")
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
                fig.add_trace(go.Scatter(
                    x=historical_df['date'],
                    y=historical_df['actual'],
                    mode='lines',
                    name='Historical Price',
                    line=dict(color='#3b82f6', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(59, 130, 246, 0.1)'
                ))
                
                fig.update_layout(
                    title=dict(
                        text=f'{ticker} Stock Price - Last 90 Days',
                        font=dict(size=18, color='#1f2937')
                    ),
                    xaxis_title='Date',
                    yaxis_title='Price (USD)',
                    hovermode='x unified',
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#f1f5f9'),
                    yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#f1f5f9', tickprefix='$'),
                    margin=dict(l=60, r=40, t=80, b=60),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Quick stats
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Price", f"${historical_df['actual'].iloc[-1]:.2f}")
                with col2:
                    st.metric("90-Day High", f"${historical_df['actual'].max():.2f}")
                with col3:
                    st.metric("90-Day Low", f"${historical_df['actual'].min():.2f}")
                with col4:
                    change_90d = ((historical_df['actual'].iloc[-1] - historical_df['actual'].iloc[0]) / historical_df['actual'].iloc[0]) * 100
                    st.metric("90-Day Change", f"{change_90d:+.2f}%")
            else:
                st.info("Select a ticker and click 'Generate Prediction' to view forecasts.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <p style="color: #64748b; font-size: 0.875rem; margin: 0;">
            📈 <strong>Stock Price Prediction System</strong> | Built with FastAPI & Streamlit
        </p>
        <p style="color: #94a3b8; font-size: 0.75rem; margin-top: 0.5rem;">
            Powered by LSTM, Random Forest, ARIMA & more
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

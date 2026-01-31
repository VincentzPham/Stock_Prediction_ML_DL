"""
Predictions Page - Generate stock price predictions.
"""

import streamlit as st
import pandas as pd

from frontend.config import HORIZON_OPTIONS, DEFAULT_HISTORICAL_DAYS, DEFAULT_HORIZON_INDEX
from frontend.api_client import api_client
from frontend.components import (
    create_historical_chart,
    create_prediction_chart,
    display_metrics_cards,
    display_price_card,
)
from frontend.utils.cache import get_cached_historical_data, get_cached_metrics
from frontend.utils.export import export_predictions


def render_current_price(ticker: str) -> None:
    """Render current price card."""
    latest = api_client.get_latest_price(ticker)
    if latest:
        display_price_card(
            ticker=ticker,
            date=latest.get("date", "N/A"),
            price=latest.get("close", 0.0)
        )


def render_metrics(ticker: str, model: str) -> None:
    """Render model metrics section."""
    st.markdown("### 📊 Model Performance Metrics")
    metrics = get_cached_metrics(ticker, model)
    display_metrics_cards(metrics)


def render_prediction(ticker: str, model: str, horizon: int) -> None:
    """Run prediction and display results."""
    st.markdown("### 🔮 Price Prediction")
    
    try:
        with st.spinner(f"Generating {horizon}-day prediction using {model}..."):
            result = api_client.predict(ticker, model, horizon)
        
        if result and result.get("predictions"):
            predictions = result["predictions"]
            
            # Display chart
            historical_df = get_cached_historical_data(ticker, DEFAULT_HISTORICAL_DAYS)
            fig = create_prediction_chart(historical_df, predictions, ticker)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display prediction table
            st.markdown("#### Predicted Prices")
            pred_df = pd.DataFrame(predictions)
            pred_df.columns = ["Day", "Date", "Predicted Price ($)"]
            pred_df["Predicted Price ($)"] = pred_df["Predicted Price ($)"].apply(
                lambda x: f"${x:,.2f}"
            )
            st.dataframe(pred_df[["Date", "Predicted Price ($)"]], use_container_width=True, hide_index=True)
            
            # Summary stats
            prices = [p["predicted_price"] for p in predictions]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("First Day", f"${prices[0]:,.2f}")
            with col2:
                st.metric("Last Day", f"${prices[-1]:,.2f}")
            with col3:
                change = ((prices[-1] - prices[0]) / prices[0]) * 100
                st.metric(
                    "Change",
                    f"${prices[-1] - prices[0]:,.2f}",
                    f"{change:+.2f}%"
                )
            
            st.success(
                f"✅ Successfully generated {len(predictions)}-day prediction for {ticker}"
            )
            
            # Export options
            st.markdown("---")
            col_exp1, col_exp2, col_exp3 = st.columns([1, 1, 2])
            with col_exp1:
                export_predictions(predictions, ticker, model, "csv")
            with col_exp2:
                export_predictions(predictions, ticker, model, "json")
        else:
            st.warning("No predictions returned. Please try again.")
            
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        st.info("Please ensure the API server is running and the model is trained.")


def render_historical_chart(ticker: str) -> None:
    """Render historical price chart."""
    st.markdown("### 📈 Historical Price Data")
    
    historical_df = get_cached_historical_data(ticker, 90)
    
    if not historical_df.empty:
        fig = create_historical_chart(historical_df, ticker, 90)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No historical data available for this ticker.")


def main():
    """Predictions page main entry point."""
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(15, 118, 110, 0.12), rgba(197, 139, 42, 0.12));
        border-radius: 18px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        border: 1px solid #e6dfd6;
    ">
        <h1 style="margin: 0; color: #1b2430;">🔮 Predictions</h1>
        <p style="margin: 0.5rem 0 0 0; color: #5f6b7a;">
            Generate stock price predictions using trained ML/DL models.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Configuration in columns
    col1, col2, col3 = st.columns(3)
    
    # Get options from API
    tickers = api_client.get_tickers()
    models = api_client.get_models()
    
    # Fallback
    if not tickers:
        tickers = ["AAPL", "AMZN", "AVGO", "BTC-USD", "GOOG", "META", "MSFT", "NVDA", "SAP", "TSLA", "TSM"]
    if not models:
        models = [
            "LSTM", "BiLSTM", "LSTM-GRU", "RNN", "ANN",
            "Random Forest", "Decision Tree", "Multiple Linear Regression",
            "ARIMA", "SARIMA", "Prophet", "Exponential Smoothing"
        ]
    
    with col1:
        selected_ticker = st.selectbox(
            "Stock Ticker",
            options=tickers,
            help="Select the stock ticker to predict."
        )
    
    with col2:
        selected_model = st.selectbox(
            "Prediction Model",
            options=models,
            help="Select the ML/DL model for prediction."
        )
    
    with col3:
        horizon_label = st.selectbox(
            "Prediction Horizon",
            options=list(HORIZON_OPTIONS.keys()),
            index=DEFAULT_HORIZON_INDEX,
            help="Number of days to predict."
        )
        horizon_days = HORIZON_OPTIONS[horizon_label]
    
    # Predict button
    predict_clicked = st.button("🚀 Generate Prediction", use_container_width=True, type="primary")
    
    if predict_clicked:
        st.session_state["run_prediction"] = True
    
    st.markdown("---")
    
    # Main content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        render_current_price(selected_ticker)
    
    with col2:
        render_metrics(selected_ticker, selected_model)
    
    st.markdown("---")
    
    # Prediction or historical chart
    if st.session_state.get("run_prediction"):
        render_prediction(selected_ticker, selected_model, horizon_days)
        st.session_state["run_prediction"] = False
    else:
        render_historical_chart(selected_ticker)


if __name__ == "__main__":
    main()

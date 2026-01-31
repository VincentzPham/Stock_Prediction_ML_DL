"""
Stock Price Prediction - Streamlit Frontend.

Main application entry point for the Streamlit UI.
"""

import streamlit as st
import pandas as pd

from frontend.config import HORIZON_OPTIONS, DEFAULT_HISTORICAL_DAYS, DEFAULT_HORIZON_INDEX
from frontend.api_client import api_client
from frontend.styles import get_custom_css, get_hero_html, get_footer_html
from frontend.components import (
    create_historical_chart,
    create_prediction_chart,
    display_metrics_cards,
    display_price_card,
)


def configure_page() -> None:
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Stock Prediction",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(get_custom_css(), unsafe_allow_html=True)


def render_sidebar() -> tuple[str, str, int]:
    """
    Render sidebar controls.
    
    Returns:
        Tuple of (selected_ticker, selected_model, horizon_days).
    """
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.caption("Select prediction parameters")
        
        # Get available options from API
        tickers = api_client.get_tickers()
        models = api_client.get_models()
        
        # Fallback if API unavailable
        if not tickers:
            tickers = ["AAPL", "AMZN", "GOOG", "META", "MSFT", "NVDA", "TSLA"]
        if not models:
            models = [
                "LSTM", "BiLSTM", "LSTM-GRU", "RNN", "ANN",
                "Random Forest", "Decision Tree", "Multiple Linear Regression",
                "ARIMA", "SARIMA", "Prophet", "Exponential Smoothing"
            ]
        
        # Ticker selection
        selected_ticker = st.selectbox(
            "Stock Ticker",
            options=tickers,
            help="Select the stock ticker to predict."
        )
        
        # Model selection
        selected_model = st.selectbox(
            "Prediction Model",
            options=models,
            help="Select the ML/DL model for prediction."
        )
        
        # Horizon selection
        horizon_label = st.selectbox(
            "Prediction Horizon",
            options=list(HORIZON_OPTIONS.keys()),
            index=DEFAULT_HORIZON_INDEX,
            help="Number of days to predict into the future."
        )
        horizon_days = HORIZON_OPTIONS[horizon_label]
        
        st.markdown("---")
        
        # Predict button
        predict_clicked = st.button("🚀 Generate Prediction", use_container_width=True)
        
        if predict_clicked:
            st.session_state["run_prediction"] = True
        
        st.markdown("---")
        st.caption("💡 Predictions are generated using trained models.")
    
    return selected_ticker, selected_model, horizon_days


def render_current_price(ticker: str) -> None:
    """
    Render current price card.
    
    Args:
        ticker: Stock ticker symbol.
    """
    latest = api_client.get_latest_price(ticker)
    if latest:
        display_price_card(
            ticker=ticker,
            date=latest.get("date", "N/A"),
            price=latest.get("close", 0.0)
        )


def render_metrics(ticker: str, model: str) -> None:
    """
    Render model metrics section.
    
    Args:
        ticker: Stock ticker symbol.
        model: Model name.
    """
    st.markdown("### 📊 Model Performance Metrics")
    metrics = api_client.get_metrics(ticker, model)
    display_metrics_cards(metrics)


def render_prediction(ticker: str, model: str, horizon: int) -> None:
    """
    Run prediction and display results.
    
    Args:
        ticker: Stock ticker symbol.
        model: Model name.
        horizon: Number of days to predict.
    """
    st.markdown("### 🔮 Price Prediction")
    
    try:
        with st.spinner(f"Generating {horizon}-day prediction using {model}..."):
            result = api_client.predict(ticker, model, horizon)
        
        if result and result.get("predictions"):
            predictions = result["predictions"]
            
            # Display chart
            historical_df = api_client.get_historical_data(ticker, DEFAULT_HISTORICAL_DAYS)
            fig = create_prediction_chart(historical_df, predictions, ticker)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display prediction table
            st.markdown("#### Predicted Prices")
            pred_df = pd.DataFrame(predictions)
            pred_df.columns = ["Date", "Predicted Price ($)"]
            pred_df["Predicted Price ($)"] = pred_df["Predicted Price ($)"].apply(
                lambda x: f"${x:,.2f}"
            )
            st.dataframe(pred_df, use_container_width=True, hide_index=True)
            
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
        else:
            st.warning("No predictions returned. Please try again.")
            
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        st.info("Please ensure the API server is running and the model is trained.")


def render_historical_chart(ticker: str) -> None:
    """
    Render historical price chart.
    
    Args:
        ticker: Stock ticker symbol.
    """
    st.markdown("### 📈 Historical Price Data")
    
    historical_df = api_client.get_historical_data(ticker, 90)
    
    if not historical_df.empty:
        fig = create_historical_chart(historical_df, ticker, 90)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No historical data available for this ticker.")


def main() -> None:
    """Main application entry point."""
    configure_page()
    
    # Hero section
    st.markdown(get_hero_html(), unsafe_allow_html=True)
    
    # Sidebar configuration
    ticker, model, horizon = render_sidebar()
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        render_current_price(ticker)
    
    with col2:
        render_metrics(ticker, model)
    
    st.markdown("---")
    
    # Prediction section
    if st.session_state.get("run_prediction"):
        render_prediction(ticker, model, horizon)
        st.session_state["run_prediction"] = False
    else:
        render_historical_chart(ticker)
    
    # Footer
    st.markdown("---")
    st.markdown(get_footer_html(), unsafe_allow_html=True)


if __name__ == "__main__":
    main()

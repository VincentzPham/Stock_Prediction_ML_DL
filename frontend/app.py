"""
Stock Price Prediction - Streamlit Frontend.

Main application entry point for the Streamlit UI.
This serves as the home page for the multi-page application.
"""

import streamlit as st

from frontend.styles import get_custom_css


def configure_page() -> None:
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Stock Prediction Platform",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(get_custom_css(), unsafe_allow_html=True)


def render_home():
    """Render the home page content."""
    # Hero section
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(15, 118, 110, 0.15), rgba(197, 139, 42, 0.15));
        border-radius: 24px;
        padding: 3rem;
        margin-bottom: 2rem;
        border: 1px solid #e6dfd6;
        text-align: center;
    ">
        <h1 style="
            font-size: 3rem;
            margin: 0;
            color: #1b2430;
            font-weight: 700;
        ">📈 Stock Price Prediction</h1>
        <p style="
            font-size: 1.2rem;
            color: #5f6b7a;
            margin: 1rem 0 1.5rem 0;
            max-width: 700px;
            margin-left: auto;
            margin-right: auto;
        ">
            Professional forecasting platform using Machine Learning and Deep Learning models.
            Analyze market trends and generate multi-day price predictions.
        </p>
        <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
            <span style="
                background: rgba(15, 118, 110, 0.12);
                color: #0f766e;
                padding: 0.5rem 1rem;
                border-radius: 20px;
                font-size: 0.85rem;
                font-weight: 600;
            ">12 ML/DL Models</span>
            <span style="
                background: rgba(197, 139, 42, 0.12);
                color: #c58b2a;
                padding: 0.5rem 1rem;
                border-radius: 20px;
                font-size: 0.85rem;
                font-weight: 600;
            ">11 Stock Tickers</span>
            <span style="
                background: rgba(15, 118, 110, 0.12);
                color: #0f766e;
                padding: 0.5rem 1rem;
                border-radius: 20px;
                font-size: 0.85rem;
                font-weight: 600;
            ">Real-time Predictions</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation cards
    st.markdown("## 🚀 Get Started")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="
            background: white;
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid #e6dfd6;
            height: 200px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        ">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
            <h3 style="margin: 0 0 0.5rem 0; color: #1b2430;">Dashboard</h3>
            <p style="color: #5f6b7a; font-size: 0.9rem; margin: 0;">
                Market overview, model leaderboard, and training coverage statistics.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Dashboard", key="btn_dashboard", use_container_width=True):
            st.switch_page("pages/1_Dashboard.py")
    
    with col2:
        st.markdown("""
        <div style="
            background: white;
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid #e6dfd6;
            height: 200px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        ">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📈</div>
            <h3 style="margin: 0 0 0.5rem 0; color: #1b2430;">Model Comparison</h3>
            <p style="color: #5f6b7a; font-size: 0.9rem; margin: 0;">
                Compare all trained models for a ticker and find the best performer.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Compare Models", key="btn_compare", use_container_width=True):
            st.switch_page("pages/2_Model_Comparison.py")
    
    with col3:
        st.markdown("""
        <div style="
            background: white;
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid #e6dfd6;
            height: 200px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        ">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔮</div>
            <h3 style="margin: 0 0 0.5rem 0; color: #1b2430;">Predictions</h3>
            <p style="color: #5f6b7a; font-size: 0.9rem; margin: 0;">
                Generate stock price predictions using trained ML/DL models.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Make Predictions", key="btn_predict", use_container_width=True):
            st.switch_page("pages/3_Predictions.py")
    
    st.markdown("---")
    
    # Model information
    st.markdown("## 🤖 Available Models")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Deep Learning**
        - LSTM (Long Short-Term Memory)
        - BiLSTM (Bidirectional LSTM)
        - LSTM-GRU Hybrid
        - RNN (Recurrent Neural Network)
        - ANN (Artificial Neural Network)
        """)
    
    with col2:
        st.markdown("""
        **Time Series**
        - ARIMA
        - SARIMA
        - Prophet
        - Exponential Smoothing
        """)
    
    with col3:
        st.markdown("""
        **Machine Learning**
        - Random Forest
        - Decision Tree
        - Multiple Linear Regression
        """)
    
    st.markdown("---")
    
    # Tickers
    st.markdown("## 💹 Supported Tickers")
    
    tickers = [
        "AAPL", "AMZN", "AVGO", "BTC-USD", "GOOG",
        "META", "MSFT", "NVDA", "SAP", "TSLA", "TSM"
    ]
    
    cols = st.columns(len(tickers))
    for i, ticker in enumerate(tickers):
        with cols[i]:
            st.markdown(f"""
            <div style="
                background: white;
                border-radius: 8px;
                padding: 0.5rem;
                text-align: center;
                border: 1px solid #e6dfd6;
                font-weight: 600;
                color: #1b2430;
            ">{ticker}</div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; color: #64748b;">
        <p style="margin: 0;">
            <strong>Stock Price Prediction Platform</strong> | Built with FastAPI + Streamlit
        </p>
        <p style="font-size: 0.8rem; margin-top: 0.5rem; color: #94a3b8;">
            Powered by TensorFlow, scikit-learn, and statsmodels
        </p>
    </div>
    """, unsafe_allow_html=True)


def main() -> None:
    """Main application entry point."""
    configure_page()
    render_home()


if __name__ == "__main__":
    main()

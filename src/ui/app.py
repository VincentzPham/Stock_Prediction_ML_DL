import streamlit as st
import requests
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from config import TICKERS, MODEL_NAMES

# API URL
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Stock Prediction App",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Price Prediction")
st.markdown("Dự đoán giá cổ phiếu sử dụng Machine Learning & Deep Learning")

# Sidebar inputs
st.sidebar.header("Configuration")

ticker = st.sidebar.selectbox(
    "Chọn Mã Cổ Phiếu (Ticker)",
    TICKERS,
    index=0
)

model_name = st.sidebar.selectbox(
    "Chọn Mô Hình (Model)",
    MODEL_NAMES,
    index=MODEL_NAMES.index('Random Forest') if 'Random Forest' in MODEL_NAMES else 0
)

horizon = st.sidebar.number_input(
    "Forecast Horizon (Days)",
    min_value=1,
    max_value=30,
    value=1,
    step=1
)

if st.sidebar.button("Predict", type="primary"):
    with st.spinner(f"Đang dự đoán {ticker} với {model_name}..."):
        try:
            # Call API
            payload = {
                "ticker": ticker,
                "model": model_name,
                "horizon": horizon
            }
            response = requests.post(f"{API_URL}/predict", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                
                # Display results
                st.success("Dự đoán thành công!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Ticker",
                        value=data['ticker']
                    )
                
                with col2:
                    st.metric(
                        label="Model",
                        value=data['model']
                    )
                    
                with col3:
                    st.metric(
                        label=f"Predicted Price (+{data['horizon']} days)",
                        value=f"${data['prediction']:.2f}",
                        delta=None # Có thể thêm delta nếu có giá hôm nay
                    )
                
                st.info(f"Dự đoán cho ngày (dựa trên dữ liệu cuối cùng): {data['date']} + {data['horizon']} ngày")
                st.caption(f"Model Path: {data['model_path']}")
                
            else:
                st.error(f"Lỗi API: {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("Không thể kết nối tới API Backend. Hãy chắc chắn rằng bạn đã chạy `uv run python src/api/app.py`.")
        except Exception as e:
            st.error(f"Lỗi không xác định: {e}")

# Footer
st.markdown("---")
st.markdown("Built with FastAPI & Streamlit")

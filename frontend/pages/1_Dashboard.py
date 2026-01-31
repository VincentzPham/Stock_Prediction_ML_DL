"""
Dashboard Page - Landing page with market overview and model leaderboard.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from frontend.config import CHART_COLORS
from frontend.utils.cache import (
    get_cached_leaderboard,
    get_cached_market_overview,
    get_cached_sentiment_overview,
)
from frontend.utils.export import export_leaderboard


def render_market_overview():
    """Render market overview section with all tickers."""
    st.markdown("### 📊 Market Overview")
    
    market_data = get_cached_market_overview()
    
    if not market_data:
        st.warning("Unable to load market data. Please ensure the API is running.")
        return
    
    # Create cards for each ticker
    cols = st.columns(4)
    
    for i, ticker_data in enumerate(market_data):
        with cols[i % 4]:
            ticker = ticker_data["ticker"]
            price = ticker_data.get("latest_price")
            change = ticker_data.get("change")
            change_pct = ticker_data.get("change_percent")
            
            if price:
                # Determine color based on change
                if change and change > 0:
                    change_color = CHART_COLORS["success"]
                    arrow = "▲"
                elif change and change < 0:
                    change_color = CHART_COLORS["error"]
                    arrow = "▼"
                else:
                    change_color = CHART_COLORS["muted"]
                    arrow = "━"
                
                st.markdown(f"""
                <div style="
                    background: white;
                    border-radius: 12px;
                    padding: 1rem;
                    border: 1px solid #e6dfd6;
                    margin-bottom: 1rem;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                ">
                    <div style="font-weight: 600; color: #1b2430; font-size: 1rem;">{ticker}</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: #1b2430; margin: 0.3rem 0;">
                        ${price:,.2f}
                    </div>
                    <div style="color: {change_color}; font-size: 0.85rem; font-weight: 500;">
                        {arrow} {change:+.2f} ({change_pct:+.2f}%)
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="
                    background: #f8f6f2;
                    border-radius: 12px;
                    padding: 1rem;
                    border: 1px solid #e6dfd6;
                    margin-bottom: 1rem;
                ">
                    <div style="font-weight: 600; color: #1b2430;">{ticker}</div>
                    <div style="color: #9aa6b2; font-size: 0.9rem;">No data</div>
                </div>
                """, unsafe_allow_html=True)


def render_leaderboard():
    """Render model leaderboard section."""
    st.markdown("### 🏆 Model Leaderboard")
    st.caption("Top performing models ranked by MAPE (Mean Absolute Percentage Error)")
    
    leaderboard_data = get_cached_leaderboard()
    
    if not leaderboard_data or "leaderboard" not in leaderboard_data:
        st.warning("Unable to load leaderboard. Please ensure models are trained.")
        return
    
    # Summary stats
    summary = leaderboard_data.get("summary", {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Models Trained",
            f"{summary.get('trained_count', 0)}/{summary.get('total_combinations', 0)}"
        )
    
    with col2:
        coverage = summary.get('coverage_percent', 0)
        st.metric("Coverage", f"{coverage}%")
    
    with col3:
        avg_mape = summary.get('avg_mape')
        st.metric("Avg MAPE", f"{avg_mape:.2f}%" if avg_mape else "N/A")
    
    with col4:
        avg_r2 = summary.get('avg_r2')
        st.metric("Avg R²", f"{avg_r2:.4f}" if avg_r2 else "N/A")
    
    st.markdown("---")
    
    # Leaderboard table
    leaderboard = leaderboard_data.get("leaderboard", [])
    
    if leaderboard:
        # Create DataFrame
        df = pd.DataFrame(leaderboard)
        
        # Add rank column
        df.insert(0, "Rank", range(1, len(df) + 1))
        
        # Format columns
        df["mape"] = df["mape"].apply(lambda x: f"{x:.2f}%" if x else "N/A")
        df["rmse"] = df["rmse"].apply(lambda x: f"{x:.2f}" if x else "N/A")
        df["r2"] = df["r2"].apply(lambda x: f"{x:.4f}" if x else "N/A")
        
        # Select display columns
        display_df = df[["Rank", "ticker", "model", "mape", "rmse", "r2"]].copy()
        display_df.columns = ["Rank", "Ticker", "Model", "MAPE", "RMSE", "R²"]
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=400
        )
        
        # Bar chart - Top 10 by MAPE
        st.markdown("#### Top 10 Models by MAPE")
        
        top_10 = leaderboard[:10]
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=[f"{e['ticker']}/{e['model']}" for e in top_10],
            y=[e['mape'] for e in top_10],
            marker_color=[CHART_COLORS["primary"] if i == 0 else CHART_COLORS["secondary"] 
                         for i in range(len(top_10))],
            text=[f"{e['mape']:.2f}%" for e in top_10],
            textposition='outside',
        ))
        
        fig.update_layout(
            xaxis_title="Ticker / Model",
            yaxis_title="MAPE (%)",
            height=350,
            margin=dict(l=40, r=40, t=40, b=80),
            plot_bgcolor="white",
            xaxis=dict(tickangle=-45),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Export options
        st.markdown("---")
        col_exp1, col_exp2, col_exp3 = st.columns([1, 1, 2])
        with col_exp1:
            export_leaderboard(leaderboard_data, "csv")
        with col_exp2:
            export_leaderboard(leaderboard_data, "json")
    else:
        st.info("No models have been trained yet. Train some models to see the leaderboard.")


def render_model_stats():
    """Render model statistics section."""
    st.markdown("### 📈 Training Coverage")
    
    leaderboard_data = get_cached_leaderboard()
    
    if not leaderboard_data:
        return
    
    col1, col2 = st.columns(2)
    
    # Ticker stats
    with col1:
        st.markdown("**By Ticker**")
        ticker_stats = leaderboard_data.get("ticker_stats", {})
        
        if ticker_stats:
            ticker_df = pd.DataFrame([
                {
                    "Ticker": ticker,
                    "Trained": stats["trained"],
                    "Total": stats["total"],
                    "Coverage": f"{stats['trained']/stats['total']*100:.0f}%"
                }
                for ticker, stats in ticker_stats.items()
            ])
            st.dataframe(ticker_df, use_container_width=True, hide_index=True, height=300)
    
    # Model stats
    with col2:
        st.markdown("**By Model**")
        model_stats = leaderboard_data.get("model_stats", {})
        
        if model_stats:
            model_df = pd.DataFrame([
                {
                    "Model": model,
                    "Trained": stats["trained"],
                    "Total": stats["total"],
                    "Coverage": f"{stats['trained']/stats['total']*100:.0f}%"
                }
                for model, stats in model_stats.items()
            ])
            st.dataframe(model_df, use_container_width=True, hide_index=True, height=300)


def render_sentiment_overview():
    """Render sentiment overview section."""
    st.markdown("### 📰 Market Sentiment")
    st.caption("News sentiment analysis for available tickers (Powered by VADER)")
    
    sentiment_data = get_cached_sentiment_overview()
    
    if not sentiment_data:
        st.info("Sentiment data is being loaded...")
        return
    
    # Filter only tickers with available sentiment
    available_sentiments = [s for s in sentiment_data if s.get("available")]
    
    if not available_sentiments:
        st.info("No sentiment data available yet. Sentiment analysis is only available for AAPL currently.")
        return
    
    # Display sentiment cards
    cols = st.columns(min(4, len(available_sentiments)))
    
    for i, sent_data in enumerate(available_sentiments):
        with cols[i % 4]:
            ticker = sent_data["ticker"]
            score = sent_data.get("sentiment_score", 0)
            label = sent_data.get("sentiment_label", "Neutral")
            news_count = sent_data.get("news_count", 0)
            date = sent_data.get("date", "N/A")
            
            # Color based on sentiment
            if score > 0.05:
                bg_color = "rgba(15, 118, 110, 0.1)"
                border_color = CHART_COLORS["success"]
                emoji = "🟢"
            elif score < -0.05:
                bg_color = "rgba(180, 35, 24, 0.1)"
                border_color = CHART_COLORS["error"]
                emoji = "🔴"
            else:
                bg_color = "rgba(154, 166, 178, 0.1)"
                border_color = CHART_COLORS["muted"]
                emoji = "⚪"
            
            st.markdown(f"""
            <div style="
                background: {bg_color};
                border-radius: 12px;
                padding: 1rem;
                border: 2px solid {border_color};
                margin-bottom: 1rem;
            ">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 600; color: #1b2430; font-size: 1rem;">{ticker}</span>
                    <span style="font-size: 1.2rem;">{emoji}</span>
                </div>
                <div style="font-size: 1.3rem; font-weight: 700; color: #1b2430; margin: 0.3rem 0;">
                    {score:+.3f}
                </div>
                <div style="color: {border_color}; font-size: 0.85rem; font-weight: 600;">
                    {label}
                </div>
                <div style="color: #5f6b7a; font-size: 0.75rem; margin-top: 0.3rem;">
                    {news_count} news | {date}
                </div>
            </div>
            """, unsafe_allow_html=True)


def main():
    """Dashboard page main entry point."""
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(15, 118, 110, 0.12), rgba(197, 139, 42, 0.12));
        border-radius: 18px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        border: 1px solid #e6dfd6;
    ">
        <h1 style="margin: 0; color: #1b2430;">📊 Dashboard</h1>
        <p style="margin: 0.5rem 0 0 0; color: #5f6b7a;">
            Overview of market data and model performance across all tickers.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    render_market_overview()
    
    st.markdown("---")
    
    render_sentiment_overview()
    
    st.markdown("---")
    
    render_leaderboard()
    
    st.markdown("---")
    
    render_model_stats()


if __name__ == "__main__":
    main()

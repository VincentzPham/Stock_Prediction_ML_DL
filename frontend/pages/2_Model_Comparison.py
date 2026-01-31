"""
Model Comparison Page - Compare models for a specific ticker.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from frontend.api_client import api_client
from frontend.config import CHART_COLORS
from frontend.utils.cache import get_cached_ticker_comparison
from frontend.utils.export import export_comparison

# Store selected ticker for export context
_current_ticker = ""


def render_comparison_table(comparison_data: dict):
    """Render comparison table with all models."""
    models = comparison_data.get("models", [])
    
    if not models:
        st.info("No model data available.")
        return
    
    # Create DataFrame
    rows = []
    for model_data in models:
        row = {
            "Model": model_data["model"],
            "Status": "✅ Trained" if model_data["status"] == "trained" else "⏳ Not Trained",
        }
        
        if model_data["metrics"]:
            row["MAPE (%)"] = f"{model_data['metrics']['mape']:.2f}" if model_data['metrics']['mape'] else "N/A"
            row["RMSE"] = f"{model_data['metrics']['rmse']:.2f}" if model_data['metrics']['rmse'] else "N/A"
            row["MAE"] = f"{model_data['metrics']['mae']:.2f}" if model_data['metrics']['mae'] else "N/A"
            row["R²"] = f"{model_data['metrics']['r2']:.4f}" if model_data['metrics']['r2'] else "N/A"
        else:
            row["MAPE (%)"] = "—"
            row["RMSE"] = "—"
            row["MAE"] = "—"
            row["R²"] = "—"
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=450
    )
    
    return comparison_data  # Return for export


def render_export_options(comparison_data: dict, ticker: str):
    """Render export options for comparison data."""
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        export_comparison(comparison_data, ticker, "csv")
    with col2:
        export_comparison(comparison_data, ticker, "json")


def render_comparison_charts(comparison_data: dict):
    """Render comparison charts."""
    models = comparison_data.get("models", [])
    
    # Filter only trained models with metrics
    trained_models = [m for m in models if m["status"] == "trained" and m["metrics"]]
    
    if not trained_models:
        st.info("No trained models to compare. Please train some models first.")
        return
    
    # MAPE Comparison
    st.markdown("#### MAPE Comparison (Lower is Better)")
    
    fig_mape = go.Figure()
    
    mape_values = [m["metrics"]["mape"] for m in trained_models]
    model_names = [m["model"] for m in trained_models]
    
    # Color the best one differently
    colors = [CHART_COLORS["primary"] if v == min(mape_values) else CHART_COLORS["secondary"] 
              for v in mape_values]
    
    fig_mape.add_trace(go.Bar(
        x=model_names,
        y=mape_values,
        marker_color=colors,
        text=[f"{v:.2f}%" for v in mape_values],
        textposition='outside',
    ))
    
    fig_mape.update_layout(
        xaxis_title="Model",
        yaxis_title="MAPE (%)",
        height=350,
        margin=dict(l=40, r=40, t=40, b=80),
        plot_bgcolor="white",
        xaxis=dict(tickangle=-45),
    )
    
    st.plotly_chart(fig_mape, use_container_width=True)
    
    # R² Comparison
    st.markdown("#### R² Score Comparison (Higher is Better)")
    
    fig_r2 = go.Figure()
    
    r2_values = [m["metrics"]["r2"] for m in trained_models if m["metrics"]["r2"]]
    r2_models = [m["model"] for m in trained_models if m["metrics"]["r2"]]
    
    if r2_values:
        colors_r2 = [CHART_COLORS["primary"] if v == max(r2_values) else CHART_COLORS["secondary"] 
                     for v in r2_values]
        
        fig_r2.add_trace(go.Bar(
            x=r2_models,
            y=r2_values,
            marker_color=colors_r2,
            text=[f"{v:.4f}" for v in r2_values],
            textposition='outside',
        ))
        
        fig_r2.update_layout(
            xaxis_title="Model",
            yaxis_title="R² Score",
            height=350,
            margin=dict(l=40, r=40, t=40, b=80),
            plot_bgcolor="white",
            xaxis=dict(tickangle=-45),
        )
        
        st.plotly_chart(fig_r2, use_container_width=True)
    
    # Radar Chart for multi-metric comparison (top 5 models)
    if len(trained_models) >= 2:
        st.markdown("#### Multi-Metric Radar Chart (Top 5 Models)")
        
        top_5 = trained_models[:5]
        
        # Normalize metrics for radar chart
        # For MAPE, MAE, RMSE: lower is better, so we invert
        # For R2: higher is better
        
        categories = ['MAPE Score', 'RMSE Score', 'MAE Score', 'R² Score']
        
        fig_radar = go.Figure()
        
        for model_data in top_5:
            metrics = model_data["metrics"]
            
            # Normalize: convert to 0-100 scale where 100 is best
            # MAPE: assume good range is 0-20%, invert so lower MAPE = higher score
            mape_score = max(0, 100 - (metrics["mape"] or 0) * 5) if metrics["mape"] else 50
            
            # RMSE: assume good range is 0-50, invert
            rmse_score = max(0, 100 - (metrics["rmse"] or 0) * 2) if metrics["rmse"] else 50
            
            # MAE: assume good range is 0-30, invert
            mae_score = max(0, 100 - (metrics["mae"] or 0) * 3) if metrics["mae"] else 50
            
            # R2: already 0-1, scale to 0-100
            r2_score = (metrics["r2"] or 0) * 100 if metrics["r2"] else 50
            
            values = [mape_score, rmse_score, mae_score, r2_score]
            
            fig_radar.add_trace(go.Scatterpolar(
                r=values + [values[0]],  # Close the shape
                theta=categories + [categories[0]],
                fill='toself',
                name=model_data["model"],
                opacity=0.6,
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            height=450,
            margin=dict(l=80, r=80, t=40, b=40),
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)


def render_best_model_card(comparison_data: dict):
    """Render best model recommendation card."""
    models = comparison_data.get("models", [])
    trained_models = [m for m in models if m["status"] == "trained" and m["metrics"]]
    
    if not trained_models:
        return
    
    # Best model is first (sorted by MAPE)
    best = trained_models[0]
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(15, 118, 110, 0.15), rgba(15, 118, 110, 0.05));
        border-radius: 16px;
        padding: 1.5rem;
        border: 2px solid {CHART_COLORS['primary']};
        margin-bottom: 1.5rem;
    ">
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
            <span style="font-size: 1.5rem;">🏆</span>
            <span style="font-size: 1.2rem; font-weight: 700; color: {CHART_COLORS['primary']};">
                Recommended Model
            </span>
        </div>
        <div style="font-size: 1.8rem; font-weight: 700; color: #1b2430; margin: 0.5rem 0;">
            {best['model']}
        </div>
        <div style="display: flex; gap: 2rem; margin-top: 1rem;">
            <div>
                <div style="font-size: 0.75rem; color: #5f6b7a; text-transform: uppercase;">MAPE</div>
                <div style="font-size: 1.2rem; font-weight: 600; color: #1b2430;">
                    {best['metrics']['mape']:.2f}%
                </div>
            </div>
            <div>
                <div style="font-size: 0.75rem; color: #5f6b7a; text-transform: uppercase;">RMSE</div>
                <div style="font-size: 1.2rem; font-weight: 600; color: #1b2430;">
                    {best['metrics']['rmse']:.2f}
                </div>
            </div>
            <div>
                <div style="font-size: 0.75rem; color: #5f6b7a; text-transform: uppercase;">R²</div>
                <div style="font-size: 1.2rem; font-weight: 600; color: #1b2430;">
                    {best['metrics']['r2']:.4f}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Model comparison page main entry point."""
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(15, 118, 110, 0.12), rgba(197, 139, 42, 0.12));
        border-radius: 18px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        border: 1px solid #e6dfd6;
    ">
        <h1 style="margin: 0; color: #1b2430;">📈 Model Comparison</h1>
        <p style="margin: 0.5rem 0 0 0; color: #5f6b7a;">
            Compare all trained models for a specific ticker to find the best performer.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Ticker selector
    tickers = api_client.get_tickers()
    if not tickers:
        tickers = ["AAPL", "AMZN", "AVGO", "BTC-USD", "GOOG", "META", "MSFT", "NVDA", "SAP", "TSLA", "TSM"]
    
    selected_ticker = st.selectbox(
        "Select Ticker",
        options=tickers,
        help="Choose a ticker to compare all its trained models."
    )
    
    # Get comparison data
    comparison_data = get_cached_ticker_comparison(selected_ticker)
    
    if not comparison_data:
        st.error("Unable to load comparison data. Please ensure the API is running.")
        return
    
    # Training status
    trained = comparison_data.get("trained_count", 0)
    total = comparison_data.get("total_count", 0)
    
    st.markdown(f"""
    <div style="
        background: white;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        border: 1px solid #e6dfd6;
        margin-bottom: 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    ">
        <div>
            <span style="font-size: 1.1rem; font-weight: 600; color: #1b2430;">
                {selected_ticker}
            </span>
            <span style="color: #5f6b7a; margin-left: 0.5rem;">
                Model Training Status
            </span>
        </div>
        <div style="
            background: {'#e6f4f1' if trained == total else '#fef3e2'};
            padding: 0.4rem 1rem;
            border-radius: 20px;
            font-weight: 600;
            color: {CHART_COLORS['primary'] if trained == total else CHART_COLORS['warning']};
        ">
            {trained}/{total} Models Trained
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Best model recommendation
    render_best_model_card(comparison_data)
    
    # Tabs for table and charts
    tab1, tab2 = st.tabs(["📊 Comparison Table", "📈 Charts"])
    
    with tab1:
        render_comparison_table(comparison_data)
        render_export_options(comparison_data, selected_ticker)
    
    with tab2:
        render_comparison_charts(comparison_data)


if __name__ == "__main__":
    main()

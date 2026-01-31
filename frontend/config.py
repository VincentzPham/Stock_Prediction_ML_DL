"""
Frontend Configuration.

Contains constants and settings for the Streamlit frontend.
"""

import os

# API URL - support Docker environment variable
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Horizon options for prediction
HORIZON_OPTIONS = {
    "1 Day": 1,
    "3 Days": 3,
    "7 Days": 7,
    "14 Days": 14,
    "30 Days": 30,
    "60 Days": 60,
}

# Chart colors
CHART_COLORS = {
    "primary": "#0f766e",       # Teal - historical data
    "secondary": "#c58b2a",     # Gold - predictions
    "muted": "#9aa6b2",         # Gray - transitions
    "success": "#0f766e",       # Green
    "warning": "#c58b2a",       # Orange
    "error": "#b42318",         # Red
    "background": "white",
    "grid": "#eee5db",
    "text": "#1b2430",
}

# Default values
DEFAULT_HISTORICAL_DAYS = 60
DEFAULT_HORIZON_INDEX = 2  # 7 Days
REQUEST_TIMEOUT = 120  # seconds

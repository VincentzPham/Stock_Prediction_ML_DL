"""
CSS Styles Module.

Contains all custom CSS styles for the Streamlit frontend.
"""


def get_custom_css() -> str:
    """
    Get custom CSS styles for the application.
    
    Returns:
        CSS string to be injected into Streamlit.
    """
    return """
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
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes scaleIn {
        from { opacity: 0; transform: scale(0.95); }
        to { opacity: 1; transform: scale(1); }
    }
    
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    
    .animate-in {
        animation: rise 0.6s ease-out forwards;
    }
    
    .animate-fade {
        animation: fadeIn 0.4s ease-out forwards;
    }
    
    .animate-slide {
        animation: slideUp 0.5s ease-out forwards;
    }
    
    .animate-scale {
        animation: scaleIn 0.3s ease-out forwards;
    }
    
    /* Staggered animation for cards/items */
    .stagger-1 { animation-delay: 0.05s; }
    .stagger-2 { animation-delay: 0.1s; }
    .stagger-3 { animation-delay: 0.15s; }
    .stagger-4 { animation-delay: 0.2s; }
    .stagger-5 { animation-delay: 0.25s; }
    .stagger-6 { animation-delay: 0.3s; }
    
    /* Loading skeleton animation */
    .skeleton {
        background: linear-gradient(
            90deg,
            #f0f0f0 25%,
            #e0e0e0 50%,
            #f0f0f0 75%
        );
        background-size: 200% 100%;
        animation: shimmer 1.5s infinite;
        border-radius: var(--radius-md);
    }
    
    /* Chart container animation */
    .chart-container {
        animation: scaleIn 0.4s ease-out forwards;
    }
    
    /* Hover lift effect for interactive elements */
    .hover-lift {
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .hover-lift:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
    }
    
    /* Smooth transitions for all interactive elements */
    .transition-all {
        transition: all 0.3s ease;
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
        
        .metrics-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    
    /* Tablet breakpoint */
    @media (max-width: 1024px) and (min-width: 769px) {
        .main .block-container {
            padding: 1.8rem 2rem;
        }
        
        h1 {
            font-size: 2.2rem !important;
        }
        
        .metrics-grid {
            grid-template-columns: repeat(3, 1fr);
        }
    }
    
    /* Mobile small */
    @media (max-width: 480px) {
        .main .block-container {
            padding: 0.8rem;
        }
        
        h1 {
            font-size: 1.6rem !important;
        }
        
        .metrics-grid {
            grid-template-columns: 1fr;
        }
        
        .panel {
            padding: 1rem;
        }
    }
</style>
"""


def get_hero_html() -> str:
    """
    Get HTML for the hero section.
    
    Returns:
        HTML string for hero section.
    """
    return """
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
    """


def get_footer_html() -> str:
    """
    Get HTML for the footer section.
    
    Returns:
        HTML string for footer section.
    """
    return """
    <div style="text-align: center; padding: 1rem 0;">
        <p style="color: #64748b; font-size: 0.875rem; margin: 0;">
            <strong>Stock Price Prediction System</strong> | Built with FastAPI and Streamlit
        </p>
        <p style="color: #94a3b8; font-size: 0.75rem; margin-top: 0.5rem;">
            Powered by LSTM, Random Forest, ARIMA and more
        </p>
    </div>
    """

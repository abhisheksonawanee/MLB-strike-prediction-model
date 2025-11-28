"""
MLB Gameday Theme Utilities
"""

import sys
from pathlib import Path
import urllib.request
import urllib.error

# Color palette matching MLB Gameday
COLORS = {
    'background': '#0A0F1A',
    'card_bg': '#111827',
    'header_bg': '#0B1220',
    'mlb_blue': '#0069A6',
    'mlb_red': '#D50032',
    'text_primary': '#FFFFFF',
    'text_secondary': '#A9B3C9',
    'border': 'rgba(255, 255, 255, 0.07)',
    'border_light': 'rgba(255, 255, 255, 0.2)',
}

def get_mlb_logo_path():
    """Get the path to the MLB logo."""
    dashboard_dir = Path(__file__).parent
    assets_dir = dashboard_dir / 'assets'
    assets_dir.mkdir(exist_ok=True)
    return assets_dir / 'mlb_logo.png'

def download_mlb_logo(logo_path: Path):
    """Download MLB logo if it doesn't exist."""
    if logo_path.exists():
        return
    
    # MLB logo URL (using a reliable source)
    logo_url = "https://www.mlbstatic.com/team-logos/league-on-dark/1.svg"
    

    png_urls = [
        "https://www.mlbstatic.com/team-logos/league-on-dark/1.svg",
        "https://www.mlb.com/assets/images/logos/league-dark/1.svg",
    ]
    
   
    try:
        # Try to download from a reliable CDN
        urllib.request.urlretrieve(
            "https://www.mlbstatic.com/team-logos/league-on-dark/1.svg",
            logo_path.with_suffix('.svg')
        )
        # Convert SVG to PNG would require additional libraries
    except (urllib.error.URLError, Exception):
        # If download fails, just continue - the logo path check will handle it
        pass

def get_gameday_css() -> str:
    """Return the complete CSS for MLB Gameday theme."""
    return f"""
    <style>
    /* ========================================================================
       MLB GAMEDAY DARK THEME - Base Styles
       ======================================================================== */
    
    /* Main app background */
    .stApp {{
        background: {COLORS['background']};
        color: {COLORS['text_primary']};
    }}
    
    /* Main content container */
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 100%;
    }}
    
    /* ========================================================================
       Header Bar
       ======================================================================== */
    
    .gameday-header {{
        background: {COLORS['header_bg']};
        border-bottom: 1px solid {COLORS['border']};
        padding: 1rem 2rem;
        margin: -2rem -2rem 2rem -2rem;
        display: flex;
        align-items: center;
        gap: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }}
    
    .gameday-header img {{
        height: 45px;
        width: auto;
        object-fit: contain;
    }}
    
    .gameday-header h1 {{
        color: {COLORS['text_primary']};
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: 0.5px;
    }}
    
    /* ========================================================================
       Card Containers
       ======================================================================== */
    
    .gameday-card {{
        background: {COLORS['card_bg']};
        border-radius: 12px;
        padding: 24px;
        margin: 1rem 0;
        border: 1px solid {COLORS['border']};
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.25);
    }}
    
    .gameday-card h2,
    .gameday-card h3 {{
        color: {COLORS['mlb_blue']};
        margin-top: 0;
        font-weight: 600;
    }}
    
    /* ========================================================================
       Typography
       ======================================================================== */
    
    h1, h2, h3, h4, h5, h6 {{
        color: {COLORS['text_primary']} !important;
    }}
    
    p, span, div, label {{
        color: {COLORS['text_primary']};
    }}
    
    /* ========================================================================
       Sidebar Styling
       ======================================================================== */
    
    [data-testid="stSidebar"] {{
        background: {COLORS['header_bg']};
        border-right: 1px solid {COLORS['border']};
    }}
    
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {{
        color: {COLORS['text_primary']} !important;
    }}
    
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {{
        color: {COLORS['text_secondary']};
    }}
    
    /* ========================================================================
       Tabs Styling
       ======================================================================== */
    
    .stTabs [data-baseweb="tab-list"] {{
        background: {COLORS['card_bg']};
        border-radius: 8px;
        padding: 4px;
        gap: 4px;
        border: 1px solid {COLORS['border']};
    }}
    
    .stTabs [data-baseweb="tab"] {{
        color: {COLORS['text_secondary']} !important;
        background: {COLORS['card_bg']} !important;
        border-radius: 6px;
        font-weight: 500;
        padding: 0.5rem 1rem;
        transition: all 0.2s;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: {COLORS['header_bg']} !important;
        color: {COLORS['text_primary']} !important;
        border-bottom: 3px solid {COLORS['mlb_red']} !important;
    }}
    
    .stTabs [aria-selected="true"] [data-baseweb="tab-highlight"] {{
        background: transparent !important;
    }}
    
    /* ========================================================================
       Buttons
       ======================================================================== */
    
    .stButton > button {{
        background: {COLORS['mlb_blue']};
        color: {COLORS['text_primary']};
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }}
    
    .stButton > button:hover {{
        background: {COLORS['mlb_red']};
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        transform: translateY(-1px);
    }}
    
    /* ========================================================================
       Selectbox & Dropdowns
       ======================================================================== */
    
    .stSelectbox > div > div {{
        background: {COLORS['card_bg']};
        border: 1px solid {COLORS['border_light']};
        border-radius: 6px;
        color: {COLORS['text_primary']};
    }}
    
    .stSelectbox > div > div:hover {{
        border-color: {COLORS['mlb_blue']};
        box-shadow: 0 0 8px rgba(0, 105, 166, 0.3);
    }}
    
    .stSelectbox label {{
        color: {COLORS['text_secondary']};
    }}
    
    /* ========================================================================
       Multiselect
       ======================================================================== */
    
    [data-baseweb="select"] > div {{
        background: {COLORS['card_bg']};
        border: 1px solid {COLORS['border_light']};
        border-radius: 6px;
        color: {COLORS['text_primary']};
    }}
    
    [data-baseweb="select"] > div:hover {{
        border-color: {COLORS['mlb_blue']};
        box-shadow: 0 0 8px rgba(0, 105, 166, 0.3);
    }}
    
    /* ========================================================================
       Radio Buttons
       ======================================================================== */
    
    .stRadio > label {{
        color: {COLORS['text_secondary']};
    }}
    
    .stRadio [role="radiogroup"] {{
        background: {COLORS['card_bg']};
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        padding: 0.5rem;
    }}
    
    /* ========================================================================
       Sliders
       ======================================================================== */
    
    .stSlider > div {{
        color: {COLORS['text_primary']};
    }}
    
    .stSlider label {{
        color: {COLORS['text_secondary']};
    }}
    
    /* ========================================================================
       Number Input
       ======================================================================== */
    
    .stNumberInput > div > div {{
        background: {COLORS['card_bg']};
        border: 1px solid {COLORS['border_light']};
        border-radius: 6px;
    }}
    
    .stNumberInput > div > div:hover {{
        border-color: {COLORS['mlb_blue']};
    }}
    
    .stNumberInput label {{
        color: {COLORS['text_secondary']};
    }}
    
    /* ========================================================================
       Checkbox
       ======================================================================== */
    
    .stCheckbox > label {{
        color: {COLORS['text_secondary']};
    }}
    
    /* ========================================================================
       Metrics
       ======================================================================== */
    
    [data-testid="stMetricValue"] {{
        color: {COLORS['text_primary']} !important;
        font-weight: 700;
    }}
    
    [data-testid="stMetricLabel"] {{
        color: {COLORS['text_secondary']} !important;
    }}
    
    /* ========================================================================
       DataFrames
       ======================================================================== */
    
    .dataframe {{
        background: {COLORS['card_bg']};
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        color: {COLORS['text_primary']};
    }}
    
    .dataframe thead {{
        background: {COLORS['header_bg']};
        color: {COLORS['text_primary']};
    }}
    
    /* ========================================================================
       Info/Warning/Error Boxes
       ======================================================================== */
    
    .stInfo {{
        background: rgba(0, 105, 166, 0.15);
        border-left: 4px solid {COLORS['mlb_blue']};
        color: {COLORS['text_primary']};
    }}
    
    .stSuccess {{
        background: rgba(76, 175, 80, 0.15);
        border-left: 4px solid #4caf50;
        color: {COLORS['text_primary']};
    }}
    
    .stWarning {{
        background: rgba(255, 152, 0, 0.15);
        border-left: 4px solid #ff9800;
        color: {COLORS['text_primary']};
    }}
    
    .stError {{
        background: rgba(213, 0, 50, 0.15);
        border-left: 4px solid {COLORS['mlb_red']};
        color: {COLORS['text_primary']};
    }}
    
    /* ========================================================================
       Plots Container
       ======================================================================== */
    
    .stPlotlyChart,
    .stPyplot {{
        background: {COLORS['card_bg']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }}
    
    /* ========================================================================
       Form Elements
       ======================================================================== */
    
    .stForm {{
        background: {COLORS['card_bg']};
        border: 1px solid {COLORS['border']};
        border-radius: 12px;
        padding: 1.5rem;
    }}
    
    /* ========================================================================
       Horizontal Rules
       ======================================================================== */
    
    hr {{
        border-color: {COLORS['border']};
        margin: 1.5rem 0;
    }}
    
    /* ========================================================================
       Hide Streamlit default elements
       ======================================================================== */
    
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    header {{ visibility: hidden; }}
    
    /* ========================================================================
       Custom Scrollbar (WebKit browsers)
       ======================================================================== */
    
    ::-webkit-scrollbar {{
        width: 10px;
        height: 10px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {COLORS['background']};
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {COLORS['border_light']};
        border-radius: 5px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: {COLORS['mlb_blue']};
    }}
    
    </style>
    """



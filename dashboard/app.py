"""
Streamlit Dashboard for MLB Pitch Strike Prediction
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from src import config
from src.dashboard_utils import (
    load_model,
    build_features_for_single_pitch,
    get_model_metrics,
    get_feature_names_from_data
)
from src.models import get_feature_importances
from src.plots import plot_strike_probability_heatmap, plot_savant_style_zone
from dashboard.theme import get_gameday_css, get_mlb_logo_path, download_mlb_logo

# Page config
st.set_page_config(
    page_title="MLB Strike Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# INJECT MLB GAMEDAY THEME CSS
# ============================================================================

st.markdown(get_gameday_css(), unsafe_allow_html=True)

# ============================================================================
# HEADER BAR WITH MLB LOGO
# ============================================================================

logo_path = get_mlb_logo_path()
download_mlb_logo(logo_path)

# Create header with logo and title (using HTML for full control)
header_container = st.container()
with header_container:
    header_col1, header_col2 = st.columns([0.8, 11.2], gap="small")
    with header_col1:
        if logo_path.exists():
            try:
                st.image(str(logo_path), width=50, use_container_width=False)
            except:
                st.markdown('<div style="font-size: 1.5rem; font-weight: bold; color: #0069A6;">MLB</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="font-size: 1.5rem; font-weight: bold; color: #0069A6;">MLB</div>', unsafe_allow_html=True)
    with header_col2:
        st.markdown('<h1 style="margin-top: 0; color: #FFFFFF;">MLB Pitch Strike Prediction Dashboard</h1>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.header("Settings")

# Model selection
try:
    available_models = [f.stem for f in config.MODELS_DIR.glob("*.joblib") if f.exists()]
    
    if not available_models:
        st.error("No trained models found! Please run `python scripts/train_models.py` first.")
        st.stop()
    
    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=available_models,
        index=0 if "random_forest" in available_models else 0
    )
    
    # Load model
    try:
        model, model_path = load_model(selected_model)
        st.sidebar.success(f"Loaded model: {selected_model}")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        st.exception(e)
        st.stop()
except Exception as e:
    st.error(f"Error initializing models: {e}")
    st.exception(e)
    st.stop()

# Model metrics
metrics = get_model_metrics(selected_model)
if metrics:
    st.sidebar.markdown("### Model Performance (Test Set)")
    st.sidebar.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
    st.sidebar.metric("F1 Score", f"{metrics.get('f1', 0):.3f}")
    if metrics.get('roc_auc') and not np.isnan(metrics['roc_auc']):
        st.sidebar.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

st.sidebar.markdown("---")

# Data filters
st.sidebar.header("Data Filters")

# ============================================================================
# LOAD DATA
# ============================================================================

@st.cache_data
def load_processed_data():
    """Load and cache processed data."""
    df = pd.read_csv(config.PROCESSED_DATA_FILE)
    return df

# Load data with error handling
if not config.PROCESSED_DATA_FILE.exists():
    st.error(f"Data file not found: {config.PROCESSED_DATA_FILE}")
    st.error("Please run `python scripts/make_dataset.py` to create the processed dataset.")
    st.stop()

try:
    df = load_processed_data()
    if df.empty:
        st.error("Loaded dataset is empty!")
        st.stop()
except Exception as e:
    st.error(f"Failed to load processed data: {e}")
    st.exception(e)
    st.stop()

# Sidebar filters
try:
    pitch_types = sorted(df['pitch_type'].dropna().unique().tolist())
    if not pitch_types:
        st.error("No pitch types found in data!")
        st.stop()
    selected_pitch_types = st.sidebar.multiselect(
        "Pitch Types",
        options=pitch_types,
        default=pitch_types
    )
except Exception as e:
    st.error(f"Error loading pitch types: {e}")
    st.exception(e)
    st.stop()

# Create pitcher_name column if pitcher columns exist (once, at the start)
if 'pitcher_first' in df.columns and 'pitcher_last' in df.columns:
    if 'pitcher_name' not in df.columns:
        df['pitcher_name'] = df['pitcher_first'] + ' ' + df['pitcher_last']

# Pitcher filter (if available)
if 'pitcher_name' in df.columns:
    pitchers = sorted(df['pitcher_name'].dropna().unique().tolist())
    selected_pitchers = st.sidebar.multiselect(
        "Pitchers",
        options=pitchers,
        default=pitchers
    )
else:
    selected_pitchers = None

# Apply filters
df_filtered = df[df['pitch_type'].isin(selected_pitch_types)].copy()

if selected_pitchers and 'pitcher_name' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['pitcher_name'].isin(selected_pitchers)]

st.sidebar.markdown("---")
st.sidebar.info(f"Showing {len(df_filtered):,} of {len(df):,} pitches")

# ============================================================================
# 3×3 ZONE VIEW FILTERS (in sidebar)
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.header("3×3 Zone View Filters")

# Value mode for 3×3 zone view (radio)
savant_value_mode = st.sidebar.radio(
    "Display Mode",
    options=["Count per Zone", "Strike Probability"],
    index=0,
    key='savant_radio'
)

# Convert to function parameter
savant_value_param = "count" if savant_value_mode == "Count per Zone" else "prob"

# Pitcher filter for 3×3 zone view
if 'pitcher_name' in df.columns:
    savant_pitchers = sorted(df['pitcher_name'].dropna().unique().tolist())
    savant_selected_pitcher = st.sidebar.selectbox(
        "Pitcher",
        options=['All'] + savant_pitchers,
        index=0,
        key='savant_pitcher'
    )
else:
    savant_selected_pitcher = 'All'

# Pitch type filter for 3×3 zone view
savant_pitch_types = sorted(df['pitch_type'].dropna().unique().tolist())
savant_selected_pitch_type = st.sidebar.selectbox(
    "Pitch Type",
    options=['All'] + savant_pitch_types,
    index=0,
    key='savant_pitch_type'
)

# Count filters for 3×3 zone view
balls_options = sorted(df['balls'].dropna().unique().tolist())
strikes_options = sorted(df['strikes'].dropna().unique().tolist())

savant_selected_balls = st.sidebar.selectbox(
    "Balls",
    options=['All'] + balls_options,
    index=0,
    key='savant_balls'
)

savant_selected_strikes = st.sidebar.selectbox(
    "Strikes",
    options=['All'] + strikes_options,
    index=0,
    key='savant_strikes'
)

# Handedness filters (optional)
if 'stand' in df.columns:
    savant_batter_handedness = sorted(df['stand'].dropna().unique().tolist())
    savant_selected_stand = st.sidebar.selectbox(
        "Batter Handedness",
        options=['All'] + savant_batter_handedness,
        index=0,
        key='savant_stand'
    )
else:
    savant_selected_stand = 'All'

if 'p_throws' in df.columns:
    savant_pitcher_handedness = sorted(df['p_throws'].dropna().unique().tolist())
    savant_selected_p_throws = st.sidebar.selectbox(
        "Pitcher Handedness",
        options=['All'] + savant_pitcher_handedness,
        index=0,
        key='savant_p_throws'
    )
else:
    savant_selected_p_throws = 'All'

# ============================================================================
# MAIN CONTENT TABS
# ============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Strike Zone Heatmap",
    "Model Insights",
    "What-if Prediction",
    "3×3 Zone Grid"
])

# ============================================================================
# TAB 1: OVERVIEW
# ============================================================================

with tab1:
    st.markdown('<div class="gameday-card">', unsafe_allow_html=True)
    st.header("Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Pitches", f"{len(df_filtered):,}")
    
    with col2:
        strike_rate = df_filtered['is_strike'].mean()
        st.metric("Overall Strike Rate", f"{strike_rate:.1%}")
    
    with col3:
        strikes = df_filtered['is_strike'].sum()
        st.metric("Strikes", f"{strikes:,}")
    
    with col4:
        balls = len(df_filtered) - strikes
        st.metric("Balls", f"{balls:,}")
    
    st.markdown("---")
    
    # Strike rate by pitch type
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Strike Rate by Pitch Type")
        pitch_type_stats = df_filtered.groupby('pitch_type')['is_strike'].agg(['count', 'mean']).reset_index()
        pitch_type_stats.columns = ['Pitch Type', 'Count', 'Strike Rate']
        pitch_type_stats = pitch_type_stats.sort_values('Strike Rate', ascending=False)
        
        fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
        ax.barh(pitch_type_stats['Pitch Type'], pitch_type_stats['Strike Rate'], color='#0069A6')
        ax.set_xlabel('Strike Rate', fontsize=12)
        ax.set_title('Strike Rate by Pitch Type', fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("Strike Rate by Count")
        # Create count string
        df_filtered_copy = df_filtered.copy()
        df_filtered_copy['count_str'] = df_filtered_copy['balls'].astype(str) + '-' + df_filtered_copy['strikes'].astype(str)
        count_stats = df_filtered_copy.groupby('count_str')['is_strike'].agg(['count', 'mean']).reset_index()
        count_stats.columns = ['Count', 'Count_Value', 'Strike Rate']
        count_stats = count_stats[count_stats['Count_Value'] >= 10]  # Only show counts with >= 10 pitches
        # Sort by balls, then strikes for better ordering
        count_stats['balls_sort'] = count_stats['Count'].str.split('-').str[0].astype(int)
        count_stats['strikes_sort'] = count_stats['Count'].str.split('-').str[1].astype(int)
        count_stats = count_stats.sort_values(['balls_sort', 'strikes_sort'], ascending=True)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(count_stats['Count'], count_stats['Strike Rate'], marker='o', linewidth=2, markersize=8)
        ax.set_xlabel('Count (Balls-Strikes)', fontsize=12)
        ax.set_ylabel('Strike Rate', fontsize=12)
        ax.set_title('Strike Rate by Count', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    # Release speed distribution
    st.subheader("Release Speed Distribution")
    fig, ax = plt.subplots(figsize=(10, 5))
    
    strikes = df_filtered[df_filtered['is_strike'] == 1]['release_speed']
    balls = df_filtered[df_filtered['is_strike'] == 0]['release_speed']
    
    ax.hist(
        strikes,
        bins=30,
        alpha=0.6,
        label=f'Strikes (n={len(strikes):,})',
        color='red',
        edgecolor='black'
    )
    ax.hist(
        balls,
        bins=30,
        alpha=0.6,
        label=f'Balls (n={len(balls):,})',
        color='blue',
        edgecolor='black'
    )
    
    ax.set_xlabel('Release Speed (mph)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Release Speed by Strike/Ball', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# TAB 2: STRIKE ZONE HEATMAP
# ============================================================================

with tab2:
    st.markdown('<div class="gameday-card">', unsafe_allow_html=True)
    st.header("Strike Zone Heatmap")
    st.markdown("Visualization of strike probability across different plate locations.")
    
    # Additional filters for heatmap
    col1, col2 = st.columns(2)
    
    with col1:
        heatmap_pitch_type = st.selectbox(
            "Filter by Pitch Type (Heatmap)",
            options=['All'] + pitch_types,
            index=0
        )
    
    with col2:
        bins = st.slider("Heatmap Resolution (bins)", min_value=15, max_value=40, value=25, step=5)
    
    # Filter data for heatmap
    heatmap_df = df_filtered.copy()
    if heatmap_pitch_type != 'All':
        heatmap_df = heatmap_df[heatmap_df['pitch_type'] == heatmap_pitch_type]
    
    if len(heatmap_df) < 10:
        st.warning("Not enough data for heatmap. Adjust filters.")
    else:
        fig = plot_strike_probability_heatmap(heatmap_df, bins=bins)
        st.pyplot(fig)
        plt.close()
        
        # Summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Pitches in Heatmap", f"{len(heatmap_df):,}")
        with col2:
            st.metric("Strike Rate", f"{heatmap_df['is_strike'].mean():.1%}")
        with col3:
            avg_speed = heatmap_df['release_speed'].mean()
            st.metric("Avg Release Speed", f"{avg_speed:.1f} mph")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# TAB 3: MODEL INSIGHTS
# ============================================================================

with tab3:
    st.markdown('<div class="gameday-card">', unsafe_allow_html=True)
    st.header("Model Insights")
    st.markdown("Understanding what features the model considers most important.")
    
    # Check if this is the baseline model
    if selected_model == "baseline":
        st.info("**Note**: The baseline model (DummyClassifier) does not provide feature importances. Please select a different model from the sidebar to view feature importance insights.")
        st.markdown("---")
        st.markdown("**Recommended models for feature insights:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("- **Random Forest**: Shows tree-based feature importances")
        with col2:
            st.markdown("- **Logistic Regression**: Shows coefficient-based importances")
        with col3:
            st.markdown("- **XGBoost**: Shows gradient boosting feature importances")
        st.markdown("")
        st.markdown("**Tip**: Use the 'Select Model' dropdown in the sidebar to switch to a different model.")
    
    # Get feature names
    try:
        feature_names = get_feature_names_from_data(df)
        
        # Get feature importances
        importances = get_feature_importances(model, feature_names)
        
        if len(importances) == 0:
            st.warning(
                f"The '{selected_model}' model does not support feature importance extraction. "
                f"Please select a model that supports feature importances (e.g., Random Forest, "
                f"Logistic Regression, or XGBoost) from the sidebar."
            )
        else:
            # Top features
            top_n = st.slider("Number of Top Features to Display", min_value=5, max_value=30, value=15)
            
            top_features = importances.head(top_n)
            
            # Bar chart
            fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
            ax.barh(range(len(top_features)), top_features.values, color='#0069A6')
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features.index, fontsize=10)
            ax.invert_yaxis()
            ax.set_xlabel('Importance', fontsize=12)
            ax.set_title(f'Top {top_n} Feature Importances ({selected_model})', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Table
            st.subheader("Feature Importance Details")
            importance_df = top_features.reset_index()
            importance_df.columns = ['Feature', 'Importance']
            importance_df['Importance'] = importance_df['Importance'].round(4)
            st.dataframe(importance_df, use_container_width=True)
            
            # Interpretation
            st.markdown("### Interpretation")
            st.info(
                f"The {selected_model} model considers the features above as most important for "
                f"predicting strike probability. Higher importance values indicate features that "
                f"contribute more to the model's predictions."
            )
    
    except Exception as e:
        st.error(f"Error extracting feature importances: {e}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# TAB 4: WHAT-IF PREDICTION
# ============================================================================

with tab4:
    st.markdown('<div class="gameday-card">', unsafe_allow_html=True)
    st.header("What-if Prediction")
    st.markdown("Enter pitch characteristics to predict strike probability.")
    
    with st.form("prediction_form"):
        st.subheader("Pitch Characteristics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Categorical inputs
            pitch_type_input = st.selectbox("Pitch Type", options=pitch_types)
            stand_input = st.selectbox("Batter Stance", options=['L', 'R'])
            p_throws_input = st.selectbox("Pitcher Throws", options=['L', 'R'])
            inning_topbot_input = st.selectbox("Inning Half", options=['Top', 'Bot'])
        
        with col2:
            # Count inputs
            balls_input = st.slider("Balls", min_value=0, max_value=3, value=0)
            strikes_input = st.slider("Strikes", min_value=0, max_value=2, value=0)
            outs_when_up_input = st.slider("Outs", min_value=0, max_value=2, value=0)
        
        st.subheader("Base Runners")
        col1, col2, col3 = st.columns(3)
        with col1:
            on_1b = st.checkbox("Runner on 1B", value=False)
        with col2:
            on_2b = st.checkbox("Runner on 2B", value=False)
        with col3:
            on_3b = st.checkbox("Runner on 3B", value=False)
        
        st.subheader("Pitch Location & Characteristics")
        col1, col2 = st.columns(2)
        
        with col1:
            plate_x_input = st.slider("Plate X (feet)", min_value=-3.0, max_value=3.0, value=0.0, step=0.1)
            plate_z_input = st.slider("Plate Z (feet)", min_value=0.0, max_value=6.0, value=2.5, step=0.1)
            release_speed_input = st.slider("Release Speed (mph)", min_value=60.0, max_value=105.0, value=95.0, step=0.5)
        
        with col2:
            pfx_x_input = st.slider("Horizontal Movement (pfx_x)", min_value=-20.0, max_value=20.0, value=0.0, step=0.5)
            pfx_z_input = st.slider("Vertical Movement (pfx_z)", min_value=-20.0, max_value=20.0, value=0.0, step=0.5)
            
            # Strike zone bounds (use defaults from data)
            sz_top_default = df['sz_top'].median() if 'sz_top' in df.columns else 3.5
            sz_bot_default = df['sz_bot'].median() if 'sz_bot' in df.columns else 1.5
            
            sz_top_input = st.number_input("Strike Zone Top", min_value=1.0, max_value=5.0, value=float(sz_top_default), step=0.1)
            sz_bot_input = st.number_input("Strike Zone Bottom", min_value=0.0, max_value=3.0, value=float(sz_bot_default), step=0.1)
        
        submit_button = st.form_submit_button("Predict Strike Probability", type="primary")
        
        if submit_button:
            # Build pitch data dictionary
            pitch_data = {
                'pitch_type': pitch_type_input,
                'stand': stand_input,
                'p_throws': p_throws_input,
                'inning_topbot': inning_topbot_input,
                'balls': balls_input,
                'strikes': strikes_input,
                'outs_when_up': outs_when_up_input,
                'on_1b': 1 if on_1b else 0,
                'on_2b': 1 if on_2b else 0,
                'on_3b': 1 if on_3b else 0,
                'plate_x': plate_x_input,
                'plate_z': plate_z_input,
                'release_speed': release_speed_input,
                'pfx_x': pfx_x_input,
                'pfx_z': pfx_z_input,
                'sz_top': sz_top_input,
                'sz_bot': sz_bot_input,
            }
            
            try:
                # Preprocess the pitch
                X_single = build_features_for_single_pitch(pitch_data, df)
                
                # Get prediction
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_single)[0, 1]
                    prediction = "Strike" if proba >= 0.5 else "Ball"
                else:
                    pred = model.predict(X_single)[0]
                    proba = 1.0 if pred == 1 else 0.0
                    prediction = "Strike" if pred == 1 else "Ball"
                
                # Display results
                st.markdown("---")
                st.subheader("Prediction Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Predicted Outcome", prediction)
                
                with col2:
                    st.metric("Strike Probability", f"{proba:.1%}")
                
                # Visualization
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Show strike zone and pitch location
                strike_zone_width = 1.5  # feet on each side
                
                # Draw strike zone
                rect = plt.Rectangle(
                    (-strike_zone_width/2, sz_bot_input),
                    strike_zone_width,
                    sz_top_input - sz_bot_input,
                    fill=False,
                    edgecolor='green',
                    linewidth=2,
                    label='Strike Zone'
                )
                ax.add_patch(rect)
                
                # Plot pitch location
                color = 'red' if proba >= 0.5 else 'blue'
                ax.scatter(
                    plate_x_input,
                    plate_z_input,
                    s=300,
                    c=color,
                    marker='o',
                    edgecolors='black',
                    linewidths=2,
                    label=f'Pitch (Prob: {proba:.1%})'
                )
                
                ax.set_xlim([-3, 3])
                ax.set_ylim([0, 6])
                ax.set_xlabel('Plate X (feet)', fontsize=12)
                ax.set_ylabel('Plate Z (feet)', fontsize=12)
                ax.set_title('Pitch Location Relative to Strike Zone', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=10)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Interpretation
                st.markdown("### Interpretation")
                if proba >= 0.7:
                    st.success(
                        f"The model predicts a **high probability** ({proba:.1%}) that this pitch "
                        f"will be a strike. This is likely because the pitch location is in or near "
                        f"the strike zone, combined with favorable count or pitch characteristics."
                    )
                elif proba >= 0.5:
                    st.info(
                        f"The model predicts a **moderate probability** ({proba:.1%}) that this pitch "
                        f"will be a strike. The outcome is somewhat uncertain based on the input features."
                    )
                else:
                    st.warning(
                        f"The model predicts a **low probability** ({proba:.1%}) that this pitch "
                        f"will be a strike. This is likely because the pitch is outside the strike zone "
                        f"or has characteristics that make it less likely to be called a strike."
                    )
            
            except Exception as e:
                st.error(f"Error making prediction: {e}")
                st.exception(e)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# TAB 5: 3×3 ZONE GRID VIEW
# ============================================================================

with tab5:
    st.markdown('<div class="gameday-card">', unsafe_allow_html=True)
    st.header("3×3 Strike Zone Grid Visualization")
    st.markdown("Professional 3×3 strike zone grid visualization with zone-by-zone analysis.")
    st.markdown("*Use the sidebar filters below to customize the visualization.*")
    
    # ============================================================================
    # APPLY FILTERS FROM SIDEBAR
    # ============================================================================
    
    # Start with full dataset
    df_savant = df.copy()
    
    # Apply filters from sidebar
    if savant_selected_pitcher != 'All':
        if 'pitcher_name' in df_savant.columns:
            df_savant = df_savant[df_savant['pitcher_name'] == savant_selected_pitcher]
    
    if savant_selected_pitch_type != 'All':
        df_savant = df_savant[df_savant['pitch_type'] == savant_selected_pitch_type]
    
    if savant_selected_balls != 'All':
        df_savant = df_savant[df_savant['balls'] == savant_selected_balls]
    
    if savant_selected_strikes != 'All':
        df_savant = df_savant[df_savant['strikes'] == savant_selected_strikes]
    
    if savant_selected_stand != 'All' and 'stand' in df_savant.columns:
        df_savant = df_savant[df_savant['stand'] == savant_selected_stand]
    
    if savant_selected_p_throws != 'All' and 'p_throws' in df_savant.columns:
        df_savant = df_savant[df_savant['p_throws'] == savant_selected_p_throws]
    
    st.markdown("---")
    
    # ============================================================================
    # DISPLAY VISUALIZATION
    # ============================================================================
    
    # Check if we have enough data
    if len(df_savant) < 5:
        st.warning(
            f"Not enough data to display visualization. "
            f"Only {len(df_savant)} pitch(es) match the selected filters. "
            f"Please adjust your filters to include at least 5 pitches."
        )
        
        # Show summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Pitches in Filtered Dataset", len(df_savant))
        with col2:
            if len(df_savant) > 0:
                strike_rate = df_savant['is_strike'].mean()
                st.metric("Strike Rate", f"{strike_rate:.1%}")
            else:
                st.metric("Strike Rate", "N/A")
        with col3:
            if len(df_savant) > 0:
                strikes = df_savant['is_strike'].sum()
                st.metric("Strikes", strikes)
            else:
                st.metric("Strikes", 0)
    else:
        # Display summary stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Pitches", len(df_savant))
        
        with col2:
            strike_rate = df_savant['is_strike'].mean()
            st.metric("Strike Rate", f"{strike_rate:.1%}")
        
        with col3:
            strikes = df_savant['is_strike'].sum()
            st.metric("Strikes", strikes)
        
        with col4:
            balls = len(df_savant) - strikes
            st.metric("Balls", balls)
        
        st.markdown("---")
        
        # Create and display visualization
        try:
            fig = plot_savant_style_zone(df_savant, value=savant_value_param, figsize=(7, 8))
            
            # Display the figure
            st.pyplot(fig)
            plt.close()
            
            # Add interpretation
            st.markdown("### Interpretation")
            
            if savant_value_param == "count":
                st.info(
                    "This visualization shows the **pitch count** in each of the 9 strike zone zones. "
                    "Zones are numbered 1-9 from top-left to bottom-right. Darker blue zones indicate "
                    "more pitches thrown in that location."
                )
            else:
                st.info(
                    "This visualization shows the **strike probability** in each of the 9 strike zone zones. "
                    "Zones are numbered 1-9 from top-left to bottom-right. "
                    "Red zones indicate lower strike probability, yellow zones indicate moderate probability, "
                    "and green zones indicate higher strike probability."
                )
            
            # Zone legend
            st.markdown("#### Zone Numbering")
            st.markdown(
                """
                The strike zone is divided into a 3×3 grid:
                - **Zones 1-3**: Top row (highest pitches)
                - **Zones 4-6**: Middle row
                - **Zones 7-9**: Bottom row (lowest pitches)
                - **Zones 1, 4, 7**: Left side (from pitcher's perspective)
                - **Zones 2, 5, 8**: Center
                - **Zones 3, 6, 9**: Right side (from pitcher's perspective)
                """
            )
        
        except Exception as e:
            st.error(f"Error creating visualization: {e}")
            st.exception(e)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "MLB Strike Prediction Dashboard | MSBA 265 Project"
    "</div>",
    unsafe_allow_html=True
)


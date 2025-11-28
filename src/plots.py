"""
Plotting utilities for MLB Strike Prediction Project.

Functions for creating visualizations of model performance and data.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, Polygon, FancyBboxPatch, Circle
from sklearn.metrics import roc_curve, roc_auc_score

from src.utils import get_logger

logger = get_logger(__name__)


def plot_roc_curve(
    y_true: pd.Series,
    y_pred_proba: np.ndarray,
    model_name: str = "Model",
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> plt.Axes:
    """
    Plot ROC curve for a single model.
    
    Parameters
    ----------
    y_true : pd.Series
        True labels
    y_pred_proba : np.ndarray
        Predicted probabilities for positive class
    model_name : str
        Name of the model (for legend)
    ax : matplotlib.Axes, optional
        Axes to plot on (default: create new figure)
    **kwargs
        Additional arguments passed to plt.plot
    
    Returns
    -------
    matplotlib.Axes
        Axes object with plot
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    # Create figure if no axes provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot ROC curve
    ax.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC = {auc:.3f})",
        linewidth=2,
        **kwargs
    )
    
    # Plot diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)', linewidth=1)
    
    # Labels and formatting
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_roc_curves_multiple(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot ROC curves for multiple models.
    
    Parameters
    ----------
    models : dict
        Dictionary mapping model names to trained models
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test labels
    figsize : tuple
        Figure size (width, height)
    
    Returns
    -------
    matplotlib.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for name, model in models.items():
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            plot_roc_curve(y_test, y_pred_proba, model_name=name, ax=ax)
        except AttributeError:
            logger.warning(f"{name} does not support predict_proba, skipping ROC curve")
        except Exception as e:
            logger.error(f"Error plotting ROC curve for {name}: {e}")
    
    plt.tight_layout()
    return fig


def plot_feature_importances(
    importances: pd.Series,
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot feature importances (bar chart).
    
    Parameters
    ----------
    importances : pd.Series
        Series of feature importances (sorted descending)
    top_n : int
        Number of top features to plot (default: 20)
    figsize : tuple
        Figure size (width, height)
    
    Returns
    -------
    matplotlib.Figure
        Figure object
    """
    # Get top N features
    top_features = importances.head(top_n)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot bar chart
    ax.barh(range(len(top_features)), top_features.values)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features.index, fontsize=10)
    ax.invert_yaxis()  # Highest importance at top
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig


def plot_strike_probability_heatmap(
    df: pd.DataFrame,
    bins: int = 20,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot heatmap of strike probability over plate location.
    
    Creates a 2D heatmap showing strike probability at different plate_x vs plate_z positions.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'plate_x', 'plate_z', and 'is_strike' columns
    bins : int
        Number of bins for each dimension (default: 20)
    figsize : tuple
        Figure size (width, height)
    
    Returns
    -------
    matplotlib.Figure
        Figure object
    """
    # Create bins
    x_bins = np.linspace(df['plate_x'].min(), df['plate_x'].max(), bins + 1)
    z_bins = np.linspace(df['plate_z'].min(), df['plate_z'].max(), bins + 1)
    
    # Bin the data
    df_binned = df.copy()
    df_binned['x_bin'] = pd.cut(df['plate_x'], bins=x_bins, labels=False, include_lowest=True)
    df_binned['z_bin'] = pd.cut(df['plate_z'], bins=z_bins, labels=False, include_lowest=True)
    
    # Calculate strike rate for each bin
    heatmap_data = df_binned.groupby(['x_bin', 'z_bin'])['is_strike'].mean().unstack(fill_value=0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = ax.imshow(
        heatmap_data.values,
        extent=[
            df['plate_x'].min(), df['plate_x'].max(),
            df['plate_z'].min(), df['plate_z'].max()
        ],
        origin='lower',
        aspect='auto',
        cmap='RdYlGn',
        interpolation='nearest'
    )
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Strike Probability', fontsize=12)
    
    # Labels
    ax.set_xlabel('Plate X (feet)', fontsize=12)
    ax.set_ylabel('Plate Z (feet)', fontsize=12)
    ax.set_title('Strike Probability by Plate Location', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    plt.tight_layout()
    return fig


def plot_release_speed_distribution(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot histogram of release speed, split by is_strike.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'release_speed' and 'is_strike' columns
    figsize : tuple
        Figure size (width, height)
    
    Returns
    -------
    matplotlib.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histograms for strikes and balls
    strikes = df[df['is_strike'] == 1]['release_speed']
    balls = df[df['is_strike'] == 0]['release_speed']
    
    ax.hist(
        strikes,
        bins=30,
        alpha=0.6,
        label=f'Strikes (n={len(strikes)})',
        color='red',
        edgecolor='black'
    )
    ax.hist(
        balls,
        bins=30,
        alpha=0.6,
        label=f'Balls (n={len(balls)})',
        color='blue',
        edgecolor='black'
    )
    
    ax.set_xlabel('Release Speed (mph)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Release Speed by Strike/Ball', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_savant_style_zone(
    df: pd.DataFrame,
    value: str = "count",
    figsize: Tuple[int, int] = (6, 7)
) -> plt.Figure:
    """
    Plot Baseball Savant-style strike zone visualization with 3x3 grid.
    
    Creates a professional strike zone visualization with:
    - 3x3 grid of zones
    - Zone values (count or strike probability)
    - MLB strike zone frame
    - Home plate polygon
    - Batter silhouettes (left and right)
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'plate_x', 'plate_z', and 'is_strike' columns
    value : str
        Type of value to display: 'count' (pitch counts) or 'prob' (strike probability)
        Default: 'count'
    figsize : tuple
        Figure size (width, height)
        Default: (6, 7)
    
    Returns
    -------
    matplotlib.Figure
        Figure object with strike zone visualization
    
    Examples
    --------
    >>> fig = plot_savant_style_zone(df, value="count")
    >>> fig = plot_savant_style_zone(df, value="prob")
    """
    logger.info(f"Creating Savant-style strike zone plot (value={value})")
    
    # Validate input
    required_cols = ['plate_x', 'plate_z', 'is_strike']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # MLB strike zone bounds (in feet)
    zone_left = -0.83
    zone_right = 0.83
    zone_bottom = 1.5
    zone_top = 3.5
    zone_width = zone_right - zone_left
    zone_height = zone_top - zone_bottom
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # ============================================================================
    # 3x3 GRID BINS
    # ============================================================================
    
    # Create 3x3 grid within strike zone
    n_zones_x = 3
    n_zones_z = 3
    
    # Zone boundaries
    x_bins = np.linspace(zone_left, zone_right, n_zones_x + 1)
    z_bins = np.linspace(zone_bottom, zone_top, n_zones_z + 1)
    
    # Bin the data
    df_copy = df.copy()
    df_copy['x_zone'] = pd.cut(df_copy['plate_x'], bins=x_bins, labels=False, include_lowest=True)
    df_copy['z_zone'] = pd.cut(df_copy['plate_z'], bins=z_bins, labels=False, include_lowest=True)
    
    # Remove NaN zones (outside strike zone)
    df_copy = df_copy.dropna(subset=['x_zone', 'z_zone'])
    
    # Calculate zone values
    zone_values = {}
    zone_counts = {}
    
    for z_idx in range(n_zones_z):
        for x_idx in range(n_zones_x):
            zone_mask = (df_copy['x_zone'] == x_idx) & (df_copy['z_zone'] == z_idx)
            zone_data = df_copy[zone_mask]
            
            if len(zone_data) > 0:
                if value == "count":
                    zone_values[(x_idx, z_idx)] = len(zone_data)
                elif value == "prob":
                    zone_values[(x_idx, z_idx)] = zone_data['is_strike'].mean()
                zone_counts[(x_idx, z_idx)] = len(zone_data)
            else:
                zone_values[(x_idx, z_idx)] = 0
                zone_counts[(x_idx, z_idx)] = 0
    
    # ============================================================================
    # DRAW STRIKE ZONE GRID
    # ============================================================================
    
    # Color scheme based on value type
    if value == "prob":
        cmap = plt.cm.RdYlGn
        vmin, vmax = 0.0, 1.0
    else:
        cmap = plt.cm.Blues
        max_count = max(zone_values.values()) if zone_values else 1
        vmin, vmax = 0, max_count if max_count > 0 else 1
    
    # Draw 3x3 grid cells
    for z_idx in range(n_zones_z):
        for x_idx in range(n_zones_x):
            x_left = x_bins[x_idx]
            x_right = x_bins[x_idx + 1]
            z_bottom = z_bins[z_idx]
            z_top = z_bins[z_idx + 1]
            
            zone_val = zone_values.get((x_idx, z_idx), 0)
            
            # Get color for this zone
            if max(vmax - vmin, 1e-10) > 0:
                normalized_val = (zone_val - vmin) / (vmax - vmin)
            else:
                normalized_val = 0.0
            
            color = cmap(normalized_val)
            
            # Draw zone rectangle
            rect = Rectangle(
                (x_left, z_bottom),
                x_right - x_left,
                z_top - z_bottom,
                facecolor=color,
                edgecolor='black',
                linewidth=1.5,
                alpha=0.7
            )
            ax.add_patch(rect)
            
            # Add zone value text
            center_x = (x_left + x_right) / 2
            center_z = (z_bottom + z_top) / 2
            
            # Format text based on value type
            if value == "count":
                text = f"{int(zone_val)}"
                fontsize = 12
            else:  # prob
                text = f"{zone_val:.2f}"
                fontsize = 11
            
            # Choose text color for contrast
            if normalized_val > 0.5:
                text_color = 'white'
            else:
                text_color = 'black'
            
            ax.text(
                center_x, center_z, text,
                ha='center', va='center',
                fontsize=fontsize,
                fontweight='bold',
                color=text_color
            )
            
            # Add zone number label (1-9, starting from top-left)
            zone_num = (n_zones_z - 1 - z_idx) * n_zones_x + x_idx + 1
            label_x = x_left + 0.1
            label_z = z_top - 0.15
            
            ax.text(
                label_x, label_z, str(zone_num),
                ha='left', va='top',
                fontsize=9,
                color='gray',
                alpha=0.6
            )
    
    # ============================================================================
    # DRAW STRIKE ZONE FRAME
    # ============================================================================
    
    # Outer strike zone rectangle
    zone_frame = Rectangle(
        (zone_left, zone_bottom),
        zone_width,
        zone_height,
        fill=False,
        edgecolor='black',
        linewidth=3,
        linestyle='-'
    )
    ax.add_patch(zone_frame)
    
    # ============================================================================
    # DRAW HOME PLATE
    # ============================================================================
    
    # Home plate polygon (below strike zone)
    plate_bottom = zone_bottom - 0.5
    plate_width = 1.0  # 17 inches = ~1.42 feet, but we'll use 1.0 for visual
    plate_points = [
        (0, plate_bottom),  # Bottom point
        (-plate_width/2, plate_bottom + 0.3),  # Bottom left
        (-plate_width/2 * 0.6, plate_bottom + 0.4),  # Top left
        (plate_width/2 * 0.6, plate_bottom + 0.4),  # Top right
        (plate_width/2, plate_bottom + 0.3),  # Bottom right
        (0, plate_bottom),  # Back to bottom
    ]
    
    plate = Polygon(
        plate_points,
        fill=True,
        facecolor='black',
        edgecolor='black',
        linewidth=2,
        alpha=0.8
    )
    ax.add_patch(plate)
    
    # ============================================================================
    # DRAW BATTER SILHOUETTES
    # ============================================================================
    
    # Left-handed batter (left side)
    batter_left_x = zone_left - 1.2
    batter_left_bottom = plate_bottom + 0.5
    
    # Body rectangle
    body_width = 0.4
    body_height = 1.8
    body_left = FancyBboxPatch(
        (batter_left_x - body_width/2, batter_left_bottom),
        body_width, body_height,
        boxstyle="round,pad=0.05",
        facecolor='#2C3E50',
        edgecolor='black',
        linewidth=1.5,
        alpha=0.8
    )
    ax.add_patch(body_left)
    
    # Head circle
    head_left = Circle(
        (batter_left_x, batter_left_bottom + body_height + 0.15),
        0.15,
        facecolor='#2C3E50',
        edgecolor='black',
        linewidth=1.5,
        alpha=0.8
    )
    ax.add_patch(head_left)
    
    # Bat line (angled)
    bat_length = 0.8
    bat_angle = 30  # degrees
    bat_start_x = batter_left_x + 0.25
    bat_start_z = batter_left_bottom + body_height * 0.7
    bat_end_x = bat_start_x + bat_length * np.cos(np.radians(bat_angle))
    bat_end_z = bat_start_z + bat_length * np.sin(np.radians(bat_angle))
    
    ax.plot(
        [bat_start_x, bat_end_x],
        [bat_start_z, bat_end_z],
        'k-',
        linewidth=3,
        alpha=0.7
    )
    
    # Right-handed batter (right side)
    batter_right_x = zone_right + 1.2
    batter_right_bottom = plate_bottom + 0.5
    
    # Body rectangle
    body_right = FancyBboxPatch(
        (batter_right_x - body_width/2, batter_right_bottom),
        body_width, body_height,
        boxstyle="round,pad=0.05",
        facecolor='#2C3E50',
        edgecolor='black',
        linewidth=1.5,
        alpha=0.8
    )
    ax.add_patch(body_right)
    
    # Head circle
    head_right = Circle(
        (batter_right_x, batter_right_bottom + body_height + 0.15),
        0.15,
        facecolor='#2C3E50',
        edgecolor='black',
        linewidth=1.5,
        alpha=0.8
    )
    ax.add_patch(head_right)
    
    # Bat line (angled, opposite direction)
    bat_start_x = batter_right_x - 0.25
    bat_start_z = batter_right_bottom + body_height * 0.7
    bat_end_x = bat_start_x - bat_length * np.cos(np.radians(bat_angle))
    bat_end_z = bat_start_z + bat_length * np.sin(np.radians(bat_angle))
    
    ax.plot(
        [bat_start_x, bat_end_x],
        [bat_start_z, bat_end_z],
        'k-',
        linewidth=3,
        alpha=0.7
    )
    
    # ============================================================================
    # FORMATTING
    # ============================================================================
    
    # Set axis limits and aspect
    ax.set_xlim(zone_left - 2.0, zone_right + 2.0)
    ax.set_ylim(plate_bottom - 0.3, zone_top + 0.5)
    ax.set_aspect('equal')
    
    # Remove default ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Title
    if value == "count":
        title = "Pitch Count by Zone"
    else:
        title = "Strike Probability by Zone"
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add axis labels
    ax.text(
        zone_left - 1.5, zone_bottom + zone_height/2,
        'Height\n(feet)',
        ha='center', va='center',
        fontsize=10,
        rotation=90
    )
    
    ax.text(
        (zone_left + zone_right) / 2, plate_bottom - 0.8,
        'Width (feet)',
        ha='center', va='center',
        fontsize=10
    )
    
    plt.tight_layout()
    
    logger.info(f"Created Savant-style strike zone plot with {len(df_copy)} pitches")
    
    return fig

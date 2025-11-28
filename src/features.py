"""
Feature engineering module for MLB Strike Prediction Project.

Builds features and target from processed data.
"""

import logging
from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from src.utils import get_logger

logger = get_logger(__name__)


def build_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, list]:
    """
    Build feature matrix and target vector from processed data.
    
    Features:
    - Categorical: pitch_type, stand, p_throws, inning_topbot
    - Numeric (context): balls, strikes, outs_when_up, on_1b, on_2b, on_3b
    - Numeric (pitch): release_speed, pfx_x, pfx_z, plate_x, plate_z, sz_top, sz_bot
    - Engineered: zone_mid, dist_to_zone_center
    
    Parameters
    ----------
    df : pd.DataFrame
        Processed DataFrame with required columns
    
    Returns
    -------
    X : pd.DataFrame
        Feature matrix (numeric, ready for modeling)
    y : pd.Series
        Target vector (is_strike)
    feature_names : list
        List of feature names (after encoding)
    """
    logger.info("Building features and target")
    
    df = df.copy()
    
    # Ensure we have the target
    if 'is_strike' not in df.columns:
        raise ValueError("DataFrame must contain 'is_strike' column")
    
    y = df['is_strike'].copy()
    
    # ============================================================================
    # ENGINEERED FEATURES
    # ============================================================================
    
    # Zone midpoint
    if 'sz_top' in df.columns and 'sz_bot' in df.columns:
        df['zone_mid'] = (df['sz_top'] + df['sz_bot']) / 2
    else:
        logger.warning("sz_top/sz_bot not found, skipping zone_mid calculation")
        df['zone_mid'] = 0.0
    
    # Distance to zone center
    if all(col in df.columns for col in ['plate_x', 'plate_z', 'zone_mid']):
        df['dist_to_zone_center'] = np.sqrt(
            df['plate_x'] ** 2 + (df['plate_z'] - df['zone_mid']) ** 2
        )
    else:
        logger.warning("Missing columns for dist_to_zone_center, setting to 0")
        df['dist_to_zone_center'] = 0.0
    
    # ============================================================================
    # BASE FEATURES (to be encoded/processed)
    # ============================================================================
    
    # Categorical features
    categorical_features = ['pitch_type', 'stand', 'p_throws', 'inning_topbot']
    
    # Numeric context features
    numeric_context = [
        'balls', 'strikes', 'outs_when_up',
        'on_1b', 'on_2b', 'on_3b'
    ]
    
    # Numeric pitch features
    numeric_pitch = [
        'release_speed', 'pfx_x', 'pfx_z',
        'plate_x', 'plate_z', 'sz_top', 'sz_bot'
    ]
    
    # Engineered features
    engineered_features = ['zone_mid', 'dist_to_zone_center']
    
    # Collect all feature columns
    feature_cols = categorical_features + numeric_context + numeric_pitch + engineered_features
    
    # Check which columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"Missing feature columns: {missing_cols}")
    
    available_feature_cols = [col for col in feature_cols if col in df.columns]
    
    # ============================================================================
    # PREPARE NUMERIC FEATURES
    # ============================================================================
    
    numeric_cols = numeric_context + numeric_pitch + engineered_features
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    # Fill missing numeric values with median (for robustness)
    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"Filled missing values in {col} with median: {median_val:.2f}")
    
    # ============================================================================
    # ENCODE CATEGORICAL FEATURES
    # ============================================================================
    
    available_categorical = [col for col in categorical_features if col in df.columns]
    
    if available_categorical:
        # Fill missing categorical values with 'Unknown'
        for col in available_categorical:
            df[col] = df[col].fillna('Unknown').astype(str)
        
        # One-hot encode
        encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        encoded = encoder.fit_transform(df[available_categorical])
        
        # Get feature names
        encoded_feature_names = encoder.get_feature_names_out(available_categorical)
        
        # Create DataFrame
        encoded_df = pd.DataFrame(
            encoded,
            columns=encoded_feature_names,
            index=df.index
        )
        
        logger.info(
            f"Encoded {len(available_categorical)} categorical features "
            f"into {len(encoded_feature_names)} binary features"
        )
    else:
        encoded_df = pd.DataFrame(index=df.index)
        encoded_feature_names = []
        logger.warning("No categorical features available")
    
    # ============================================================================
    # COMBINE FEATURES
    # ============================================================================
    
    # Combine numeric and encoded categorical
    numeric_df = df[numeric_cols].copy()
    
    X = pd.concat([numeric_df, encoded_df], axis=1)
    
    # Convert to numeric (ensure all columns are numeric)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Fill any remaining NaN with 0
    X = X.fillna(0)
    
    feature_names = list(X.columns)
    
    logger.info(
        f"Feature matrix shape: {X.shape} "
        f"({len(numeric_cols)} numeric + {len(encoded_feature_names)} encoded categorical)"
    )
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y, feature_names


"""
Dashboard utilities for MLB Strike Prediction Project.

Helper functions for loading models, preprocessing single pitches, and extracting model info.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder

from src import config
from src.features import build_features_and_target
from src.models import get_feature_importances
from src.utils import get_logger

logger = get_logger(__name__)


def load_model(model_name: str = "random_forest") -> Tuple[Any, str]:
    """
    Load a trained model from the models directory.
    
    Parameters
    ----------
    model_name : str
        Name of the model file (without .joblib extension)
        Default: "random_forest"
    
    Returns
    -------
    model : Trained model
        Loaded sklearn model
    model_path : Path
        Path to the model file
    
    Raises
    ------
    FileNotFoundError
        If model file doesn't exist
    """
    model_path = config.MODELS_DIR / f"{model_name}.joblib"
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            f"Available models: {list(config.MODELS_DIR.glob('*.joblib'))}"
        )
    
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    
    return model, model_path


def build_features_for_single_pitch(
    pitch_data: Dict[str, Any],
    reference_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Build features for a single pitch row, using the reference dataset for encoding and statistics.
    
    This function replicates the feature engineering logic from build_features_and_target
    but for a single row, using the reference dataset to fit the encoder and get medians.
    
    Parameters
    ----------
    pitch_data : dict
        Dictionary with pitch attributes (pitch_type, stand, p_throws, etc.)
    reference_df : pd.DataFrame
        Reference dataset used to fit encoders and compute statistics
    
    Returns
    -------
    pd.DataFrame
        Single-row feature matrix (ready for model prediction)
    """
    # Create a single-row DataFrame from the pitch data
    pitch_df = pd.DataFrame([pitch_data])
    
    # ============================================================================
    # ENGINEERED FEATURES
    # ============================================================================
    
    # Zone midpoint
    if 'sz_top' in pitch_df.columns and 'sz_bot' in pitch_df.columns:
        pitch_df['zone_mid'] = (pitch_df['sz_top'] + pitch_df['sz_bot']) / 2
    elif 'sz_top' not in pitch_df.columns or 'sz_bot' not in pitch_df.columns:
        # Use defaults from reference data if available
        if 'sz_top' in reference_df.columns and 'sz_bot' in reference_df.columns:
            sz_top_default = reference_df['sz_top'].median()
            sz_bot_default = reference_df['sz_bot'].median()
            if 'sz_top' not in pitch_df.columns:
                pitch_df['sz_top'] = sz_top_default
            if 'sz_bot' not in pitch_df.columns:
                pitch_df['sz_bot'] = sz_bot_default
            pitch_df['zone_mid'] = (pitch_df['sz_top'] + pitch_df['sz_bot']) / 2
        else:
            pitch_df['zone_mid'] = 0.0
    
    # Distance to zone center
    if all(col in pitch_df.columns for col in ['plate_x', 'plate_z', 'zone_mid']):
        pitch_df['dist_to_zone_center'] = np.sqrt(
            pitch_df['plate_x'] ** 2 + (pitch_df['plate_z'] - pitch_df['zone_mid']) ** 2
        )
    else:
        pitch_df['dist_to_zone_center'] = 0.0
    
    # ============================================================================
    # FEATURE COLUMNS
    # ============================================================================
    
    categorical_features = ['pitch_type', 'stand', 'p_throws', 'inning_topbot']
    numeric_context = ['balls', 'strikes', 'outs_when_up', 'on_1b', 'on_2b', 'on_3b']
    numeric_pitch = ['release_speed', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'sz_top', 'sz_bot']
    engineered_features = ['zone_mid', 'dist_to_zone_center']
    
    numeric_cols = numeric_context + numeric_pitch + engineered_features
    
    # ============================================================================
    # PREPARE NUMERIC FEATURES
    # ============================================================================
    
    # Fill missing numeric values with medians from reference data
    for col in numeric_cols:
        if col not in pitch_df.columns or pd.isna(pitch_df[col].iloc[0]):
            if col in reference_df.columns:
                median_val = reference_df[col].median()
                pitch_df[col] = median_val
            else:
                pitch_df[col] = 0.0
    
    # Ensure all numeric columns are present
    for col in numeric_cols:
        if col not in pitch_df.columns:
            pitch_df[col] = 0.0
    
    # ============================================================================
    # ENCODE CATEGORICAL FEATURES
    # ============================================================================
    
    available_categorical = [col for col in categorical_features if col in pitch_df.columns]
    
    if available_categorical:
        # Fill missing categorical values
        for col in available_categorical:
            if pd.isna(pitch_df[col].iloc[0]):
                pitch_df[col] = 'Unknown'
            pitch_df[col] = pitch_df[col].astype(str)
        
        # Fit encoder on reference data (same categories)
        encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        
        # Prepare reference categorical data
        ref_categorical = reference_df[available_categorical].copy()
        for col in available_categorical:
            ref_categorical[col] = ref_categorical[col].fillna('Unknown').astype(str)
        
        # Fit encoder
        encoder.fit(ref_categorical)
        
        # Transform single pitch
        pitch_categorical = pitch_df[available_categorical]
        encoded = encoder.transform(pitch_categorical)
        encoded_feature_names = encoder.get_feature_names_out(available_categorical)
        
        # Create DataFrame
        encoded_df = pd.DataFrame(
            encoded,
            columns=encoded_feature_names,
            index=pitch_df.index
        )
    else:
        encoded_df = pd.DataFrame(index=pitch_df.index)
        encoded_feature_names = []
    
    # ============================================================================
    # COMBINE FEATURES
    # ============================================================================
    
    # Get numeric features
    numeric_df = pitch_df[numeric_cols].copy()
    
    # Combine
    X = pd.concat([numeric_df, encoded_df], axis=1)
    
    # Ensure all columns are numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Fill any remaining NaN with 0
    X = X.fillna(0)
    
    return X


def get_model_metrics(model_name: str) -> Optional[Dict[str, float]]:
    """
    Load model metrics from saved JSON file if available.
    
    Parameters
    ----------
    model_name : str
        Name of the model
    
    Returns
    -------
    dict or None
        Dictionary of metrics if available, None otherwise
    """
    metrics_path = config.MODELS_DIR / "metrics_test.json"
    
    if not metrics_path.exists():
        return None
    
    try:
        import json
        with open(metrics_path, 'r') as f:
            metrics_dict = json.load(f)
        
        if model_name in metrics_dict:
            return metrics_dict[model_name]
    except Exception as e:
        logger.warning(f"Could not load metrics: {e}")
    
    return None


def get_feature_names_from_data(df: pd.DataFrame) -> list:
    """
    Get feature names by building features on the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Processed dataset
    
    Returns
    -------
    list
        List of feature names (after encoding)
    """
    _, _, feature_names = build_features_and_target(df)
    return feature_names


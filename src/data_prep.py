"""
Data preparation module for MLB Strike Prediction Project.

Handles loading raw data, creating labels, and cleaning.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from src import config
from src.utils import get_logger, ensure_directories_exist

logger = get_logger(__name__)


def load_raw_data(path: Path) -> pd.DataFrame:
    """
    Load raw data from CSV file.
    
    Parameters
    ----------
    path : Path
        Path to raw data CSV file
    
    Returns
    -------
    pd.DataFrame
        Loaded DataFrame
    
    Raises
    ------
    FileNotFoundError
        If file doesn't exist
    ValueError
        If required columns are missing
    """
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")
    
    logger.info(f"Loading raw data from {path}")
    df = pd.read_csv(path)
    
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    # Check for required columns
    missing_cols = [col for col in config.REQUIRED_COLUMNS if col not in df.columns]
    
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )
    
    logger.info("✓ All required columns present")
    return df


def create_is_strike_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary 'is_strike' label from description column.
    
    Labeling logic:
    - is_strike = 1 (strike) if description in STRIKE_DESCRIPTIONS
    - is_strike = 0 (ball) if description in BALL_DESCRIPTIONS
    - Drops rows where description doesn't match either category
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with 'description' column
    
    Returns
    -------
    pd.DataFrame
        DataFrame with 'is_strike' column added and rows filtered
    """
    logger.info("Creating is_strike label from description column")
    
    # Create binary label
    df = df.copy()
    
    # Create is_strike column
    df['is_strike'] = np.nan
    
    # Mark strikes (label = 1)
    df.loc[df['description'].isin(config.STRIKE_DESCRIPTIONS), 'is_strike'] = 1
    
    # Mark balls (label = 0)
    df.loc[df['description'].isin(config.BALL_DESCRIPTIONS), 'is_strike'] = 0
    
    # Count rows before filtering
    initial_count = len(df)
    
    # Drop rows where is_strike is NaN (i.e., description not in strike or ball lists)
    # This includes in-play outcomes, pickoffs, etc.
    df = df.dropna(subset=['is_strike'])
    
    dropped_count = initial_count - len(df)
    
    logger.info(
        f"Labeled {len(df)} pitches: "
        f"{df['is_strike'].sum():.0f} strikes, "
        f"{(~df['is_strike'].astype(bool)).sum():.0f} balls. "
        f"Dropped {dropped_count} rows with other outcomes."
    )
    
    # Convert to int
    df['is_strike'] = df['is_strike'].astype(int)
    
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and filter the dataset.
    
    Operations:
    - Filter release_speed within reasonable bounds
    - Drop rows with missing critical columns
    - Keep only relevant columns for modeling
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    
    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame
    """
    logger.info("Applying basic cleaning and filtering")
    
    df = df.copy()
    initial_count = len(df)
    
    # Filter by release speed
    if 'release_speed' in df.columns:
        before_speed = len(df)
        df = df[
            (df['release_speed'] >= config.MIN_RELEASE_SPEED) &
            (df['release_speed'] <= config.MAX_RELEASE_SPEED)
        ]
        logger.info(
            f"Filtered release speed: {before_speed} -> {len(df)} "
            f"({before_speed - len(df)} removed)"
        )
    
    # Drop rows with missing critical columns
    critical_cols = ['plate_x', 'plate_z', 'balls', 'strikes', 'is_strike']
    
    for col in critical_cols:
        if col in df.columns:
            before = len(df)
            df = df.dropna(subset=[col])
            dropped = before - len(df)
            if dropped > 0:
                logger.info(f"Dropped {dropped} rows with missing {col}")
    
    logger.info(f"Cleaning complete: {initial_count} -> {len(df)} rows")
    
    return df


def save_processed(df: pd.DataFrame, path: Path) -> None:
    """
    Save processed dataset to CSV.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save
    path : Path
        Output file path
    """
    ensure_directories_exist()
    
    logger.info(f"Saving processed data to {path}")
    df.to_csv(path, index=False)
    logger.info(f"Saved {len(df)} rows to {path}")


def prepare_dataset(
    raw_data_path: Optional[Path] = None,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Complete data preparation pipeline.
    
    Steps:
    1. Load raw data
    2. Create is_strike label
    3. Clean and filter
    4. Save processed data
    
    Parameters
    ----------
    raw_data_path : Path, optional
        Path to raw data file (default: config.RAW_DATA_FILE)
    output_path : Path, optional
        Path to save processed data (default: config.PROCESSED_DATA_FILE)
    
    Returns
    -------
    pd.DataFrame
        Processed DataFrame
    """
    raw_data_path = raw_data_path or config.RAW_DATA_FILE
    output_path = output_path or config.PROCESSED_DATA_FILE
    
    # Load raw data
    df = load_raw_data(raw_data_path)
    
    # Create label
    df = create_is_strike_label(df)
    
    # Clean
    df = basic_cleaning(df)
    
    # Save
    save_processed(df, output_path)
    
    return df


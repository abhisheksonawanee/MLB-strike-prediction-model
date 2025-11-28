"""
Data download module for MLB Strike Prediction Project.

Downloads pitch-level data from Statcast using pybaseball.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
from pybaseball import playerid_lookup, statcast_pitcher

from src import config
from src.utils import ensure_directories_exist, get_logger

logger = get_logger(__name__)


def find_pitcher_id(first_name: str, last_name: str) -> Optional[int]:
    """
    Find MLBAM ID for a pitcher given first and last name.
    
    Parameters
    ----------
    first_name : str
        Pitcher's first name
    last_name : str
        Pitcher's last name
    
    Returns
    -------
    int or None
        MLBAM ID if found, None otherwise
    
    Raises
    ------
    ValueError
        If multiple matches found (ambiguous) or no matches found
    """
    logger.info(f"Looking up pitcher: {first_name} {last_name}")
    
    try:
        # Look up player ID
        player_info = playerid_lookup(last_name, first_name)
        
        if player_info.empty:
            raise ValueError(f"No player found matching {first_name} {last_name}")
        
        # Filter for pitchers if there are multiple matches
        if len(player_info) > 1:
            logger.warning(f"Multiple matches found for {first_name} {last_name}, using first result")
        
        # Get the key_mlbam ID (this is the ID needed for statcast_pitcher)
        mlbam_id = player_info.iloc[0]['key_mlbam']
        
        if pd.isna(mlbam_id):
            raise ValueError(f"No MLBAM ID found for {first_name} {last_name}")
        
        logger.info(f"Found MLBAM ID: {mlbam_id}")
        return int(mlbam_id)
    
    except Exception as e:
        logger.error(f"Error looking up {first_name} {last_name}: {e}")
        raise


def download_pitcher_data(
    first_name: str,
    last_name: str,
    start_date: str,
    end_date: str,
    throws: Optional[str] = None
) -> pd.DataFrame:
    """
    Download pitch-level data for a specific pitcher and date range.
    
    Parameters
    ----------
    first_name : str
        Pitcher's first name
    last_name : str
        Pitcher's last name
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    throws : str, optional
        Pitcher's throwing hand (L or R) from config
    
    Returns
    -------
    pd.DataFrame
        DataFrame with pitch-level data
    """
    # Find pitcher ID
    try:
        pitcher_id = find_pitcher_id(first_name, last_name)
    except (ValueError, Exception) as e:
        logger.error(f"Failed to find pitcher ID for {first_name} {last_name}: {e}")
        return pd.DataFrame()
    
    logger.info(f"Downloading data for {first_name} {last_name} (MLBAM ID: {pitcher_id}) from {start_date} to {end_date}")
    
    try:
        # Download statcast data
        df = statcast_pitcher(start_dt=start_date, end_dt=end_date, player_id=pitcher_id)
        
        if df.empty:
            logger.warning(f"No data found for {first_name} {last_name} in date range {start_date} to {end_date}")
            return pd.DataFrame()
        
        logger.info(f"✓ Downloaded {len(df)} pitches for {first_name} {last_name}")
        
        # Add pitcher tracking columns
        df['pitcher_first'] = first_name
        df['pitcher_last'] = last_name
        df['pitcher_name'] = f"{first_name} {last_name}"
        
        # Add throws column from config if provided
        if throws is not None:
            df['throws_config'] = throws
        
        return df
    
    except Exception as e:
        logger.error(f"Error downloading data for {first_name} {last_name}: {e}")
        logger.exception(e)
        return pd.DataFrame()


def download_all_pitchers(
    pitchers: List[Dict[str, str]],
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Download data for multiple pitchers and concatenate.
    
    Parameters
    ----------
    pitchers : list of dict
        List of pitcher dicts with keys: 'first', 'last', 'throws' (or 'handedness' for backward compatibility)
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    
    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame with all pitcher data
    """
    logger.info(f"Starting download for {len(pitchers)} pitchers")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info("-" * 60)
    
    all_data = []
    successful_downloads = 0
    failed_downloads = 0
    
    for idx, pitcher in enumerate(pitchers, 1):
        first_name = pitcher['first']
        last_name = pitcher['last']
        
        # Support both 'throws' and 'handedness' keys for backward compatibility
        throws = pitcher.get('throws') or pitcher.get('handedness')
        
        logger.info(f"[{idx}/{len(pitchers)}] Processing: {first_name} {last_name} ({throws if throws else 'Unknown'})")
        
        df = download_pitcher_data(first_name, last_name, start_date, end_date, throws=throws)
        
        if not df.empty:
            all_data.append(df)
            successful_downloads += 1
            logger.info(f"  → Success: {len(df)} pitches")
        else:
            failed_downloads += 1
            logger.warning(f"  → Failed: No data retrieved")
        
        logger.info("-" * 60)
    
    if not all_data:
        raise ValueError("No data downloaded for any pitchers. Check logs for errors.")
    
    # Concatenate all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    logger.info("=" * 60)
    logger.info(f"DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total pitchers configured: {len(pitchers)}")
    logger.info(f"Successful downloads: {successful_downloads}")
    logger.info(f"Failed downloads: {failed_downloads}")
    logger.info(f"Total pitches downloaded: {len(combined_df):,}")
    
    # Show breakdown by pitcher
    if 'pitcher_name' in combined_df.columns:
        pitcher_counts = combined_df['pitcher_name'].value_counts().sort_index()
        logger.info(f"\nPitches by pitcher:")
        for name, count in pitcher_counts.items():
            logger.info(f"  {name}: {count:,} pitches")
    
    # Show breakdown by throws if available
    if 'throws_config' in combined_df.columns:
        throws_counts = combined_df['throws_config'].value_counts()
        logger.info(f"\nPitches by throwing hand:")
        for hand, count in throws_counts.items():
            logger.info(f"  {hand}-handed: {count:,} pitches")
    
    logger.info("=" * 60)
    
    return combined_df


def save_raw_data(df: pd.DataFrame, output_path: Path, overwrite: bool = False) -> None:
    """
    Save raw data to CSV file.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save
    output_path : Path
        Output file path
    overwrite : bool
        If True, overwrite existing file; if False, skip if file exists
    """
    if output_path.exists() and not overwrite:
        logger.info(f"File {output_path} already exists. Skipping save. (Use overwrite=True to overwrite)")
        return
    
    ensure_directories_exist()
    
    logger.info(f"Saving raw data to {output_path}")
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} rows to {output_path}")


def download_and_save(
    pitchers: Optional[List[Dict[str, str]]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    output_path: Optional[Path] = None,
    overwrite: bool = False
) -> pd.DataFrame:
    """
    Main function to download data for configured pitchers and save to CSV.
    
    Parameters
    ----------
    pitchers : list of dict, optional
        List of pitcher configurations (default: from config.PITCHERS or config.PITCHER_CONFIG)
    start_date : str, optional
        Start date (default: from config.DEFAULT_START_DATE)
    end_date : str, optional
        End date (default: from config.DEFAULT_END_DATE)
    output_path : Path, optional
        Output file path (default: from config.RAW_DATA_FILE)
    overwrite : bool
        Whether to overwrite existing file
    
    Returns
    -------
    pd.DataFrame
        Downloaded DataFrame
    """
    # Use defaults from config if not provided
    # Try PITCHERS first, then fall back to PITCHER_CONFIG for backward compatibility
    if pitchers is None:
        pitchers = getattr(config, 'PITCHERS', None) or getattr(config, 'PITCHER_CONFIG', None)
    
    start_date = start_date or config.DEFAULT_START_DATE
    end_date = end_date or config.DEFAULT_END_DATE
    output_path = output_path or config.RAW_DATA_FILE
    
    if pitchers is None:
        raise ValueError("No pitchers configured. Please set config.PITCHERS or config.PITCHER_CONFIG")
    
    # Download data
    df = download_all_pitchers(pitchers, start_date, end_date)
    
    # Save to file
    save_raw_data(df, output_path, overwrite=overwrite)
    
    return df


"""
Script to download raw MLB pitch data from Statcast.

Usage:
    python scripts/download_data.py

This will download data for the default pitchers and season specified in src/config.py.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_download import download_and_save
from src.utils import setup_logging, ensure_directories_exist

if __name__ == "__main__":
    setup_logging()
    ensure_directories_exist()
    
    print("=" * 60)
    print("MLB Pitch Data Download")
    print("=" * 60)
    print()
    
    print("Downloading data for configured pitchers...")
    df = download_and_save(overwrite=True)
    
    print()
    print("=" * 60)
    print(f"Download complete! Downloaded {len(df)} total pitches.")
    print(f"Data saved to: data/raw/pitches_raw.csv")
    print("=" * 60)


"""
Script to create processed dataset from raw data.

Usage:
    python scripts/make_dataset.py

This will:
1. Load raw data from data/raw/pitches_raw.csv
2. Create is_strike labels
3. Clean and filter data
4. Save processed data to data/processed/pitches_processed.csv
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_prep import prepare_dataset
from src.utils import setup_logging, ensure_directories_exist

if __name__ == "__main__":
    setup_logging()
    ensure_directories_exist()
    
    print("=" * 60)
    print("MLB Pitch Data Preparation")
    print("=" * 60)
    print()
    
    print("Preparing dataset...")
    df = prepare_dataset()
    
    print()
    print("=" * 60)
    print(f"Dataset preparation complete!")
    print(f"Processed {len(df)} pitches")
    print(f"Strike rate: {df['is_strike'].mean():.2%}")
    print(f"Data saved to: data/processed/pitches_processed.csv")
    print("=" * 60)


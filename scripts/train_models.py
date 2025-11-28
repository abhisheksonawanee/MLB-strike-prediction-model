"""
Script to train all models on processed data.

Usage:
    python scripts/train_models.py

This will:
1. Load processed data from data/processed/pitches_processed.csv
2. Build features and target
3. Split into train/test sets
4. Train all models
5. Evaluate on validation set (train split)
6. Save models and metrics to models/
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.features import build_features_and_target
from src.models import (
    split_data,
    train_all_models,
    evaluate_all_models,
    save_metrics
)
from src import config
from src.utils import setup_logging, ensure_directories_exist

if __name__ == "__main__":
    setup_logging()
    ensure_directories_exist()
    
    print("=" * 60)
    print("MLB Pitch Model Training")
    print("=" * 60)
    print()
    
    # Load processed data
    print(f"Loading processed data from {config.PROCESSED_DATA_FILE}...")
    df = pd.read_csv(config.PROCESSED_DATA_FILE)
    print(f"Loaded {len(df)} pitches")
    print()
    
    # Build features
    print("Building features and target...")
    X, y, feature_names = build_features_and_target(df)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    print()
    
    # Split data
    print("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    print()
    
    # Train models
    print("Training models...")
    trained_models = train_all_models(X_train, y_train)
    print()
    
    # Evaluate on training set (for validation metrics)
    print("Evaluating models on training set...")
    train_metrics = evaluate_all_models(trained_models, X_train, y_train)
    print()
    print("Training Set Metrics:")
    print(train_metrics.round(4))
    print()
    
    # Save training metrics
    train_metrics_path = config.MODELS_DIR / "metrics_train"
    save_metrics(train_metrics, train_metrics_path)
    
    # Also evaluate on test set for reference (though we'll do full eval in evaluate_models.py)
    print("Evaluating models on test set...")
    test_metrics = evaluate_all_models(trained_models, X_test, y_test)
    print()
    print("Test Set Metrics:")
    print(test_metrics.round(4))
    print()
    
    # Save test metrics
    test_metrics_path = config.MODELS_DIR / "metrics_test"
    save_metrics(test_metrics, test_metrics_path)
    
    print("=" * 60)
    print("Training complete!")
    print(f"Models saved to: {config.MODELS_DIR}")
    print(f"Metrics saved to: {config.MODELS_DIR}")
    print("=" * 60)


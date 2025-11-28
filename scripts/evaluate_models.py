"""
Script to evaluate trained models on test data.

Usage:
    python scripts/evaluate_models.py

This will:
1. Load processed data and trained models
2. Build features and target
3. Evaluate all models on test set
4. Print and save metrics
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import joblib
from src.features import build_features_and_target
from src.models import (
    split_data,
    evaluate_all_models,
    save_metrics
)
from src import config
from src.utils import setup_logging, ensure_directories_exist

if __name__ == "__main__":
    setup_logging()
    ensure_directories_exist()
    
    print("=" * 60)
    print("MLB Pitch Model Evaluation")
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
    print()
    
    # Split data (same split as training)
    print("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    print()
    
    # Load trained models
    print("Loading trained models...")
    models = {}
    model_files = list(config.MODELS_DIR.glob("*.joblib"))
    
    if not model_files:
        print("ERROR: No trained models found in models/ directory.")
        print("Please run scripts/train_models.py first.")
        sys.exit(1)
    
    for model_path in model_files:
        model_name = model_path.stem
        models[model_name] = joblib.load(model_path)
        print(f"  Loaded: {model_name}")
    print()
    
    # Evaluate models
    print("Evaluating models on test set...")
    print()
    test_metrics = evaluate_all_models(models, X_test, y_test)
    
    print()
    print("=" * 60)
    print("Test Set Evaluation Results")
    print("=" * 60)
    print()
    print(test_metrics.round(4))
    print()
    
    # Save metrics
    test_metrics_path = config.MODELS_DIR / "metrics_test"
    save_metrics(test_metrics, test_metrics_path)
    
    print("=" * 60)
    print("Evaluation complete!")
    print(f"Metrics saved to: {test_metrics_path}.csv and .json")
    print("=" * 60)


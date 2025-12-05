"""
Modeling utilities for MLB Strike Prediction Project.

Includes model definitions, training, and evaluation functions.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Skipping XGBClassifier.")

from src import config
from src.utils import get_logger, ensure_directories_exist

logger = get_logger(__name__)


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = None,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target vector
    test_size : float
        Proportion of data for test set
    random_state : int, optional
        Random seed for reproducibility
    stratify : bool
        Whether to stratify by target (default: True)
    
    Returns
    -------
    X_train, X_test, y_train, y_test
        Train and test splits
    """
    random_state = random_state or config.RANDOM_STATE
    
    stratify_param = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )
    
    logger.info(
        f"Split data: train={len(X_train)}, test={len(X_test)} "
        f"(train strike rate: {y_train.mean():.2%}, "
        f"test strike rate: {y_test.mean():.2%})"
    )
    
    return X_train, X_test, y_train, y_test


def get_models() -> Dict[str, Any]:
    """
    Get dictionary of model definitions.
    
    Returns
    -------
    dict
        Dictionary mapping model names to model instances
    """
    models = {
        'baseline': DummyClassifier(strategy='most_frequent', random_state=config.RANDOM_STATE),
        'logistic_regression': LogisticRegression(
            solver='lbfgs',
            max_iter=1000,
            random_state=config.RANDOM_STATE,
            n_jobs=-1
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=config.RANDOM_STATE,
            n_jobs=-1
        ),
    }
    
    # Optionally add XGBoost if available
    if XGBOOST_AVAILABLE:
        models['xgboost'] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            eval_metric='logloss'
        )
    
    return models


def train_model(model, X_train: pd.DataFrame, y_train: pd.Series, model_name: str) -> Any:
    """
    Train a single model.
    
    Parameters
    ----------
    model : sklearn-like estimator
        Model instance to train
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    model_name : str
        Name of the model (for logging)
    
    Returns
    -------
    Trained model
    """
    logger.info(f"Training {model_name}...")
    
    model.fit(X_train, y_train)
    
    logger.info(f"✓ Trained {model_name}")
    
    return model


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    models_dir: Path = None
) -> Dict[str, Any]:
    """
    Train all models and save them.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    models_dir : Path, optional
        Directory to save models (default: config.MODELS_DIR)
    
    Returns
    -------
    dict
        Dictionary of trained models
    """
    models_dir = models_dir or config.MODELS_DIR
    ensure_directories_exist()
    
    models = get_models()
    trained_models = {}
    
    for name, model in models.items():
        trained_model = train_model(model, X_train, y_train, name)
        trained_models[name] = trained_model
        
        # Save model (ensure path is a string for compatibility)
        model_path = models_dir / f"{name}.joblib"
        # Ensure the models directory exists
        model_path.parent.mkdir(parents=True, exist_ok=True)
        # Use string path when calling joblib.dump to avoid Windows path issues
        joblib.dump(trained_model, str(model_path))
        logger.info(f"Saved {name} to {model_path}")
    
    return trained_models


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "model"
) -> Dict[str, float]:
    """
    Evaluate a model and return metrics.
    
    Parameters
    ----------
    model : sklearn-like estimator
        Trained model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
    model_name : str
        Name of the model (for logging)
    
    Returns
    -------
    dict
        Dictionary of metrics (accuracy, precision, recall, f1, roc_auc)
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = None
    
    # Try to get probability predictions (for ROC-AUC)
    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        logger.warning(f"{model_name} does not support predict_proba, skipping ROC-AUC")
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
    }
    
    # ROC-AUC (if probabilities available)
    if y_pred_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        except ValueError:
            logger.warning(f"Could not calculate ROC-AUC for {model_name}")
            metrics['roc_auc'] = np.nan
    else:
        metrics['roc_auc'] = np.nan
    
    # Format ROC-AUC value
    if np.isnan(metrics['roc_auc']):
        roc_auc_str = 'N/A'
    else:
        roc_auc_str = f"{metrics['roc_auc']:.3f}"
    
    logger.info(
        f"{model_name} metrics: "
        f"Accuracy={metrics['accuracy']:.3f}, "
        f"F1={metrics['f1']:.3f}, "
        f"ROC-AUC={roc_auc_str}"
    )
    
    return metrics


def evaluate_all_models(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.DataFrame:
    """
    Evaluate all models and return metrics DataFrame.
    
    Parameters
    ----------
    models : dict
        Dictionary of trained models
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
    
    Returns
    -------
    pd.DataFrame
        DataFrame with metrics for each model
    """
    all_metrics = []
    
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, name)
        metrics['model'] = name
        all_metrics.append(metrics)
    
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df = metrics_df.set_index('model')
    
    return metrics_df


def save_metrics(metrics_df: pd.DataFrame, path: Path) -> None:
    """
    Save metrics to CSV and JSON files.
    
    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame with metrics
    path : Path
        Base path for saving (will create .csv and .json versions)
    """
    ensure_directories_exist()
    
    # Save CSV
    csv_path = path.with_suffix('.csv')
    metrics_df.to_csv(csv_path)
    logger.info(f"Saved metrics to {csv_path}")
    
    # Save JSON
    json_path = path.with_suffix('.json')
    metrics_dict = metrics_df.to_dict(orient='index')
    
    # Convert numpy types to native Python types for JSON
    json_dict = {}
    for model, metrics in metrics_dict.items():
        json_dict[model] = {
            k: float(v) if not np.isnan(v) else None
            for k, v in metrics.items()
        }
    
    with open(json_path, 'w') as f:
        json.dump(json_dict, f, indent=2)
    
    logger.info(f"Saved metrics to {json_path}")


def get_feature_importances(model: Any, feature_names: List[str]) -> pd.Series:
    """
    Extract feature importances from a model.
    
    Parameters
    ----------
    model : sklearn-like estimator
        Trained model with feature_importances_ attribute
    feature_names : list
        List of feature names
    
    Returns
    -------
    pd.Series
        Series of feature importances (sorted descending)
    """
    if hasattr(model, 'feature_importances_'):
        importances = pd.Series(model.feature_importances_, index=feature_names)
        return importances.sort_values(ascending=False)
    elif hasattr(model, 'coef_'):
        # For logistic regression, use absolute coefficients
        if model.coef_.ndim > 1:
            coef = np.abs(model.coef_[0])
        else:
            coef = np.abs(model.coef_)
        importances = pd.Series(coef, index=feature_names)
        return importances.sort_values(ascending=False)
    else:
        logger.warning("Model does not have feature_importances_ or coef_ attribute")
        return pd.Series(dtype=float)


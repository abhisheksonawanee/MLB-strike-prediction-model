"""
Utility functions for MLB Strike Prediction Project.
"""

import logging
from pathlib import Path
from typing import Optional

from src import config


def setup_logging(level: int = logging.INFO) -> None:
    """
    Set up basic logging configuration.
    
    Parameters
    ----------
    level : int
        Logging level (default: logging.INFO)
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def ensure_directories_exist() -> None:
    """
    Create necessary directories if they don't exist.
    
    Creates:
    - data/raw
    - data/processed
    - models
    """
    directories = [
        config.RAW_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.MODELS_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ Ensured directory exists: {directory}")


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Parameters
    ----------
    name : str, optional
        Logger name (default: root logger)
    
    Returns
    -------
    logging.Logger
        Logger instance
    """
    return logging.getLogger(name)


"""
Configuration module for MLB Strike Prediction Project.

Defines paths, constants, and default pitcher configurations.
"""

from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# File paths
RAW_DATA_FILE = RAW_DATA_DIR / "pitches_raw.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "pitches_processed.csv"

# ML constants
RANDOM_STATE = 42

# Default season for data download
DEFAULT_SEASON = 2023
DEFAULT_START_DATE = "2023-03-25"
DEFAULT_END_DATE = "2023-10-01"

# Alias for clearer naming
START_DATE = DEFAULT_START_DATE
END_DATE = DEFAULT_END_DATE

# Pitcher configuration - expanded list for comprehensive dataset
# Each dict contains first name, last name, and throws (L or R)
PITCHERS = [
    {"first": "Gerrit",  "last": "Cole",      "throws": "R"},
    {"first": "Corbin",  "last": "Burnes",    "throws": "R"},
    {"first": "Logan",   "last": "Webb",      "throws": "R"},
    {"first": "Spencer", "last": "Strider",   "throws": "R"},
    {"first": "Zack",    "last": "Wheeler",   "throws": "R"},
    {"first": "Luis",    "last": "Castillo",  "throws": "R"},
    {"first": "Sonny",   "last": "Gray",      "throws": "R"},
    {"first": "Tyler",   "last": "Glasnow",   "throws": "R"},
    {"first": "Clayton", "last": "Kershaw",   "throws": "L"},
    {"first": "Blake",   "last": "Snell",     "throws": "L"},
    {"first": "Max",     "last": "Fried",     "throws": "L"},
    {"first": "Framber", "last": "Valdez",    "throws": "L"},
    {"first": "Jordan",  "last": "Montgomery","throws": "L"},
    {"first": "Justin",  "last": "Steele",    "throws": "L"},
]

# Alias for backward compatibility
PITCHER_CONFIG = PITCHERS

# Statcast columns that are required for the pipeline
REQUIRED_COLUMNS = [
    "description",
    "release_speed",
    "plate_x",
    "plate_z",
    "balls",
    "strikes",
    "pitch_type",
    "stand",
    "p_throws",
    "inning_topbot",
    "outs_when_up",
    "on_1b",
    "on_2b",
    "on_3b",
    "sz_top",
    "sz_bot",
    "pfx_x",
    "pfx_z",
]

# Strike descriptions (label = 1)
STRIKE_DESCRIPTIONS = [
    "called_strike",
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "missed_bunt",
]

# Ball descriptions (label = 0)
BALL_DESCRIPTIONS = [
    "ball",
    "blocked_ball",
    "intent_ball",
    "ball_in_dirt",
    "pitchout",
    "hit_by_pitch",
]

# Velocity bounds for filtering
MIN_RELEASE_SPEED = 60.0  # mph
MAX_RELEASE_SPEED = 105.0  # mph


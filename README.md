# MLB Pitch Strike Prediction

**MSBA 265 Course Project**

This project predicts whether an MLB pitch will be called a strike or a ball using pitch-level Statcast data from the `pybaseball` library.

## Project Overview

The goal is to build a binary classification model that predicts `is_strike = 1` (strike) or `is_strike = 0` (ball) for each pitch, using features such as:
- Pitch location (plate_x, plate_z)
- Pitch characteristics (release speed, pitch type, spin rate)
- Game context (count, outs, runners on base)
- Batter and pitcher handedness

## Project Structure

```
.
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── pyproject.toml           # Project metadata
├── data/
│   ├── raw/                 # Raw Statcast CSV files
│   └── processed/           # Cleaned, feature-engineered CSV
├── models/                  # Trained model files (.joblib)
├── notebooks/
│   ├── 01_eda.ipynb        # Exploratory data analysis
│   └── 02_modeling.ipynb   # Model training and evaluation
├── dashboard/               # Streamlit dashboard
│   └── app.py              # Main dashboard application
├── src/                     # Reusable Python modules
│   ├── __init__.py
│   ├── config.py           # Configuration and constants
│   ├── data_download.py    # Data download functions
│   ├── data_prep.py        # Data cleaning and labeling
│   ├── features.py         # Feature engineering
│   ├── models.py           # Model definitions and training
│   ├── plots.py            # Visualization utilities
│   ├── dashboard_utils.py  # Dashboard helper functions
│   └── utils.py            # Helper functions
│   ├── __init__.py
│   ├── config.py           # Configuration and constants
│   ├── data_download.py    # Data download functions
│   ├── data_prep.py        # Data cleaning and labeling
│   ├── features.py         # Feature engineering
│   ├── models.py           # Model definitions and training
│   ├── plots.py            # Visualization utilities
│   └── utils.py            # Helper functions
└── scripts/                 # Command-line scripts
    ├── download_data.py    # Download raw data
    ├── make_dataset.py     # Create processed dataset
    ├── train_models.py     # Train all models
    └── evaluate_models.py  # Evaluate trained models
```

## Setup Instructions

### 1. Create Virtual Environment

```bash
# On Windows
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Data Pipeline

Execute the following scripts in order:

```bash
# Step 1: Download raw data from Statcast
python scripts/download_data.py

# Step 2: Process and clean the data
python scripts/make_dataset.py

# Step 3: Train models
python scripts/train_models.py

# Step 4: Evaluate models (optional, also done in train_models.py)
python scripts/evaluate_models.py
```

### 4. Explore in Notebooks

Open the Jupyter notebooks for interactive exploration:

```bash
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_modeling.ipynb
```

### 5. Launch Dashboard (Optional)

Run the interactive Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

The dashboard provides:
- Dataset overview and statistics
- Strike zone heatmaps
- Model insights and feature importance
- Interactive "what-if" prediction tool

## Data Source

Data is downloaded from **MLB Statcast** via the `pybaseball` library:
- Default pitchers: Gerrit Cole, Corbin Burnes, Logan Webb
- Default season: 2023 regular season (March 25 - October 1)
- All pitch-level data including location, velocity, spin, and outcomes

You can modify the pitchers and date range in `src/config.py`.

## Label Definition: `is_strike`

The binary target `is_strike` is created from the `description` field:

- **Strikes (is_strike = 1)**: called_strike, swinging_strike, swinging_strike_blocked, foul, foul_tip, missed_bunt
- **Balls (is_strike = 0)**: ball, blocked_ball, intent_ball, ball_in_dirt, pitchout, hit_by_pitch
- **Dropped**: All other outcomes (in-play events, pickoffs, etc.)

## Models

The project trains and evaluates multiple models:

1. **Baseline**: DummyClassifier (predicts most frequent class)
2. **Logistic Regression**: Linear model with L2 regularization
3. **Random Forest**: Ensemble of decision trees (200 trees)
4. **XGBoost** (optional): Gradient boosting classifier

Models are saved in `models/` as `.joblib` files. Metrics are saved as CSV and JSON.

## Features

The model uses the following features:

**Categorical:**
- `pitch_type` (FF, SL, CH, etc.)
- `stand` (batter handedness: L/R)
- `p_throws` (pitcher handedness: L/R)
- `inning_topbot` (Top/Bottom)

**Numeric Context:**
- `balls`, `strikes`, `outs_when_up`
- `on_1b`, `on_2b`, `on_3b` (base runners)

**Numeric Pitch:**
- `release_speed` (mph)
- `pfx_x`, `pfx_z` (pitch movement)
- `plate_x`, `plate_z` (pitch location at plate)
- `sz_top`, `sz_bot` (strike zone boundaries)

**Engineered:**
- `zone_mid` = (sz_top + sz_bot) / 2
- `dist_to_zone_center` = distance from pitch location to zone center

Categorical features are one-hot encoded. All features are numeric in the final feature matrix.

## Evaluation Metrics

Models are evaluated on:
- **Accuracy**: Overall correctness
- **Precision**: Of predicted strikes, how many were actually strikes
- **Recall**: Of actual strikes, how many were predicted correctly
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve (handles class imbalance better)

## Configuration

Modify settings in `src/config.py`:
- `PITCHER_CONFIG`: List of pitchers to download
- `DEFAULT_SEASON`: Year for data download
- `DEFAULT_START_DATE`, `DEFAULT_END_DATE`: Date range
- `RANDOM_STATE`: Random seed for reproducibility

## Troubleshooting

**Issue**: `pybaseball` download fails or is slow
- **Solution**: The library may have rate limits. Wait and retry, or download data for fewer pitchers.

**Issue**: Missing columns in raw data
- **Solution**: Statcast columns can vary. Check `src/config.py` `REQUIRED_COLUMNS` and adjust if needed.

**Issue**: Import errors when running scripts
- **Solution**: Ensure you're in the project root directory and have activated the virtual environment.

## Next Steps

- Experiment with different pitchers or seasons
- Add more features (weather, stadium, etc.)
- Try different model hyperparameters
- Perform cross-validation for more robust evaluation
- Add feature importance visualizations
- Explore class imbalance handling (SMOTE, class weights)

## License

This project is for educational purposes as part of MSBA 265.
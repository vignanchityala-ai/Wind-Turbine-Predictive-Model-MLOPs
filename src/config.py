"""
config.py
=========
Central configuration for the Wind Farm A early-fault-detection pipeline.

Dataset background (CARE benchmark, Gück/Roelofs/Faulstich 2024, re-hosted on
Kaggle as "Wind Turbine SCADA Data For Early Fault Detection"):

- 95 CSVs total across 3 wind farms. We target Wind Farm A: 5 turbines,
  22 sub-datasets (11 anomaly, 11 normal), 86 columns, 10-minute resolution.
- Each sub-dataset = one train/predict "episode" for one turbine:
    * ~1 year of training data (train_test == 'train')
    * 4-98 days of prediction data (train_test == 'prediction'), padded
      before/after the actual event window so the event can't be guessed
      from the amount of prediction data alone.
- Every row has a status_type_id (0-5). Only 0 and 2 count as "normal".
  Status-labeled abnormal points are the turbine already flagging itself,
  so they're excluded from anomaly-detection scoring (they're trivial).
- Each sub-dataset has an event_info.csv / event_info.json (varies by
  release) describing: event_id, event_label ('anomaly'/'normal'),
  event_start, event_end. For anomaly datasets, event_start is the
  earliest point anything might already be wrong; the actual fault
  onset is event_end.

Because this dataset can't be pulled programmatically from this environment
(Kaggle needs auth, Zenodo isn't on the allowed egress list), point
RAW_DATA_DIR at wherever you've extracted the Kaggle download.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — EDIT THESE after downloading from Kaggle
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "Wind Farm A"   # <- unzip here
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

for _d in (PROCESSED_DIR, MODEL_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------
ID_COL = "id"                     # row id
TIME_COL = "time_stamp"           # 10-minute timestamp
ASSET_COL = "asset_id"            # turbine identifier
SPLIT_COL = "train_test"          # 'train' / 'prediction'
STATUS_COL = "status_type_id"     # 0-5, see STATUS_MAP

TRAIN_VALUE = "train"
PREDICTION_VALUE = "prediction"

NON_FEATURE_COLS = [ID_COL, TIME_COL, ASSET_COL, SPLIT_COL, STATUS_COL]

# Status-ID semantics from the CARE paper (Table 3.3)
STATUS_MAP = {
    0: ("Normal operation without limitations", True),
    1: ("Derated power generation with a power restriction", False),
    2: ("Asset is idling / waiting to operate", True),
    3: ("Asset is in service mode", False),
    4: ("Asset is down due to a fault or other reason", False),
    5: ("Other operational states (test, setup, icing, emergency power)", False),
}
NORMAL_STATUS_IDS = {sid for sid, (_, normal) in STATUS_MAP.items() if normal}
ABNORMAL_STATUS_IDS = {sid for sid, (_, normal) in STATUS_MAP.items() if not normal}

# Known-safe recognizable feature name fragments (only power / reactive
# power / wind speed retain semantic names post-anonymization)
POWER_HINTS = ["power"]
WIND_SPEED_HINTS = ["wind_speed", "windspeed"]

# ---------------------------------------------------------------------------
# Modeling config
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# Rolling-window feature engineering
ROLLING_WINDOWS = [6, 18, 36]   # in 10-min steps -> 1h, 3h, 6h

# Fraction of the (normal-only) training data held out for validation /
# threshold calibration
VALIDATION_FRACTION = 0.2

# Contamination assumption for unsupervised models (IsolationForest etc.)
# Kept low since training data is expected to be ~normal behavior.
ASSUMED_CONTAMINATION = 0.01

# Reconstruction-error percentile (on validation normal data) used to set
# the anomaly threshold for the autoencoder / NBM approach.
THRESHOLD_PERCENTILE = 99.0

# Minimum consecutive flagged points before we call it an "event" rather
# than an isolated spike (reduces false alarms — reliability sub-score).
MIN_EVENT_LENGTH = 6   # 1 hour of consecutive flags at 10-min resolution

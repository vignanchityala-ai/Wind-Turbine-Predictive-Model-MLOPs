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
# Paths & Multi-Farm Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

FARM_CONFIGS = {
    "A": {
        "raw_dir": PROJECT_ROOT / "data" / "raw" / "Wind Farm A",
        "n_features": 86,
        "n_turbines": 5,
        "n_datasets": 22
    },
    "B": {
        "raw_dir": PROJECT_ROOT / "data" / "raw" / "Wind Farm B",
        "n_features": 257,
        "n_turbines": 16,
        "n_datasets": 38
    },
    "C": {
        "raw_dir": PROJECT_ROOT / "data" / "raw" / "Wind Farm C",
        "n_features": 957,
        "n_turbines": 15,
        "n_datasets": 35
    },
}

FARMS = list(FARM_CONFIGS.keys())

# Default for backward compatibility
RAW_DATA_DIR = FARM_CONFIGS["A"]["raw_dir"]

# Data lake zones
BRONZE_DIR = PROJECT_ROOT / "data" / "bronze"
SILVER_DIR = PROJECT_ROOT / "data" / "silver"
GOLD_DIR   = PROJECT_ROOT / "data" / "gold"

# Other standard paths
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

for _d in (BRONZE_DIR, SILVER_DIR, GOLD_DIR, PROCESSED_DIR, MODEL_DIR, OUTPUT_DIR):
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

# OPEN QUESTION, confirmed from the official Zenodo record (v6,
# https://zenodo.org/records/15846963, "Notes" section) but not yet resolved
# in this pipeline's code:
#
#   "In wind farm A status_type_id labels can be ignored while evaluating
#    prediction time frames of error events with metrics like the CARE-score
#    since the status_type_id of wind farm A is based on the EDP failure
#    logbook and it is intended to be used for filtering of the training
#    data."
#
# i.e. for Farm A specifically, status_type_id is fine to use for filtering
# TRAINING rows (which is all this pipeline uses it for at training time —
# see run_pipeline.process_one), but the note suggests it should NOT be used
# to help build prediction-period ground truth for Coverage/Reliability,
# since for Farm A it's derived from the same failure logbook that also
# defines event_start/event_end, rather than being an independent SCADA
# signal (which it reportedly is for Farms B and C). evaluation.py currently
# also uses status_type_id to restrict WHICH prediction-period points are
# eligible for scoring at all (see normal_mask in evaluate_subdataset) —
# whether that specific use counts as the kind of use the note is warning
# against isn't settled yet. Worth validating empirically against real Farm
# A data (e.g. check whether non-normal status_type_id in the prediction
# window aligns suspiciously exactly with event_start/event_end) before
# trusting Coverage/Reliability numbers on Farm A at face value.
#
# Also per the same source: Farm A's data no longer contains status_type_id
# 5 as of v5+ (all converted to 0/normal, since it wasn't ground-truth-based
# for Farm A) — status 5 remains meaningful for Farms B and C, where it's
# real SCADA-status-code-based. No code change needed for that part; STATUS_MAP
# already treats 5 as not-normal, which is correct for B/C and moot for A.
#
# One more angle on the same open question: evaluation.py's Coverage metric
# (evaluate_subdataset) currently treats every normal-status point AFTER
# fault_onset as "should NOT have been flagged" (gt=0), so a model that
# correctly keeps flagging through and after the fault gets penalized as a
# false positive for doing the right thing. In practice this window is
# probably narrow -- status_type_id typically flips to abnormal (excluded
# from scoring via normal_mask) fairly soon after the real fault begins --
# but "probably narrow" hasn't been checked against real data. Worth
# checking as part of the same empirical validation mentioned above: how
# large is the gap between fault_onset (event_end) and the first
# non-normal status_type_id in the prediction period, across real anomaly
# datasets? A large gap would mean Coverage is currently harsher than it
# should be for models that keep alerting appropriately post-fault.

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
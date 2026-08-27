"""
feature_selection.py
======================
Feature selection for the (large) set of rolling-stat-derived features
engineer_features() produces. Two real problems this addresses:

1. All-NaN / near-constant columns. Confirmed on real Farm A data: sklearn's
   SimpleImputer warns "Skipping features without any observed values" for
   sensor_46/sensor_49 -- meaning these columns are ENTIRELY NaN in the
   training period for some turbines. An entirely-NaN (or near-constant)
   column adds dimensionality with zero discriminative value, and can
   distort distance/tree-based scoring (IsolationForest can partition on a
   feature that carries no real signal). Dropped via a NaN-fraction and
   variance threshold.

2. Redundancy from the rolling-window design. Every sensor gets THREE
   window sizes (1h/3h/6h) x TWO stats (mean/std) = 6 derived columns.
   Adjacent windows on a smooth SCADA signal are often highly correlated
   (roll6_mean and roll18_mean for the same sensor, especially) -- this
   adds dimensionality without much independent signal. Dropped via greedy
   pairwise-correlation pruning.

IMPORTANT -- train/serve consistency: selection must run ONLY on
TRAINING-normal data (the same rows used to fit the model), never on the
full dataset or the prediction period, or it leaks information about
what's "typical" from data the model shouldn't have seen yet. The SELECTED
column list must then be treated as the canonical feature_cols from that
point on -- persisted in the model bundle and reused unchanged at serving
time, the same train/serve-consistency principle as power_curve_reference
and feature_descriptions elsewhere in this pipeline. This module doesn't
handle persistence itself; the caller (run_pipeline.process_one) already
does this correctly by just reassigning feature_cols to the selected list
before it gets saved into the bundle -- no new bundle key needed.

--- Memory: sampling for the selection DECISION, not for training ---
Confirmed on real Farm A data at farm-pooled scale (22 datasets, 859,574
pooled rows, 555 engineered columns): computing df[working].var() over the
full pooled set tried to allocate a 3.55 GiB intermediate array and
crashed with MemoryError. This never showed up in testing because the
synthetic test fixture (a handful of tiny datasets) is roughly 15x smaller
in total data volume than a real 22-dataset farm.

The fix: variance/NaN-fraction/correlation are all used to make a
DECISION about which columns to keep, not to train anything -- a
representative random sample is statistically sufficient to determine
whether a column is near-constant, mostly missing, or redundant with
another column. Above max_rows_for_selection, a fixed-seed random sample
is used for all three checks instead of the full dataframe; the model
itself is still trained on ALL the data afterward -- only this selection
step is sampled.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_MAX_ROWS_FOR_SELECTION = 100_000


def select_features(
    df: pd.DataFrame,
    candidate_cols: list[str],
    max_nan_fraction: float = 0.5,
    min_variance: float = 1e-8,
    max_correlation: float = 0.95,
    protect: list[str] = None,
    max_rows_for_selection: int = DEFAULT_MAX_ROWS_FOR_SELECTION,
    random_state: int = 42,
) -> tuple[list[str], dict]:
    """
    Select a reduced feature set from candidate_cols. Fit ONLY on df --
    pass in training-normal rows specifically (see module docstring for
    why this matters).

    protect: columns to always keep regardless of these checks, e.g. the
    power-curve residual features -- they're physically meaningful even if
    they happen to be highly correlated with something else in a given
    dataset, so correlation alone shouldn't be grounds to drop them.

    max_rows_for_selection: if df has more rows than this, a fixed-seed
    random sample of this size is used for the NaN/variance/correlation
    computations instead of the full dataframe -- see module docstring.
    Lower this (e.g. to 20_000-50_000) if you still hit memory pressure on
    a particularly memory-constrained machine; raise it if you have memory
    to spare and want the selection decision computed from more data.

    Returns (selected_cols, diagnostics). diagnostics has three lists
    (dropped_nan, dropped_variance, dropped_correlation) explaining what
    was cut and why, for logging/transparency -- feature selection that
    silently removes columns is exactly the kind of thing that's confusing
    to debug later if it's not visible.
    """
    protect = set(protect or [])
    diagnostics = {"dropped_nan": [], "dropped_variance": [], "dropped_correlation": []}

    working = [c for c in candidate_cols if c in df.columns]
    missing_from_df = [c for c in candidate_cols if c not in df.columns]
    if missing_from_df:
        log.warning(
            "select_features: %d candidate column(s) not present in the "
            "given dataframe, skipped: %s", len(missing_from_df), missing_from_df[:5],
        )

    stat_df = df
    if working and len(df) > max_rows_for_selection:
        stat_df = df[working].sample(n=max_rows_for_selection, random_state=random_state)
        log.info(
            "select_features: sampling %d/%d rows to compute selection "
            "statistics (variance/NaN-fraction/correlation) -- keeps peak "
            "memory bounded regardless of pooled dataset size. The model "
            "itself still trains on all %d rows afterward; only this "
            "column-selection DECISION is made from a representative sample.",
            max_rows_for_selection, len(df), len(df),
        )
    elif working:
        stat_df = df[working]

    # --- Step 1: drop columns with too much missingness in training data ---
    if working:
        nan_frac = stat_df.isna().mean()
        drop_nan = [c for c in working if c not in protect and nan_frac.get(c, 0) > max_nan_fraction]
        if drop_nan:
            diagnostics["dropped_nan"] = drop_nan
            working = [c for c in working if c not in drop_nan]
            stat_df = stat_df[working]

    # --- Step 2: drop near-constant columns (near-zero variance) ---
    if working:
        variances = stat_df.var()
        drop_var = [c for c in working if c not in protect and variances.get(c, 0) < min_variance]
        if drop_var:
            diagnostics["dropped_variance"] = drop_var
            working = [c for c in working if c not in drop_var]
            stat_df = stat_df[working]

    # --- Step 3: greedy pairwise-correlation pruning ---
    # Iterate in column order; for each column, drop it if it's highly
    # correlated with an EARLIER column already kept. O(n^2) in column
    # count -- fine at Farm A scale (~150-600 engineered cols after steps
    # 1-2), would need chunking for Farm C's much larger feature set.
    if len(working) > 1:
        corr = stat_df.corr().abs()
        kept: list[str] = []
        drop_corr = []
        for c in working:
            if c in protect:
                kept.append(c)
                continue
            is_redundant = False
            for k in kept:
                if k in corr.columns and c in corr.index:
                    val = corr.loc[c, k]
                    if pd.notna(val) and val > max_correlation:
                        is_redundant = True
                        break
            if is_redundant:
                drop_corr.append(c)
            else:
                kept.append(c)
        diagnostics["dropped_correlation"] = drop_corr
        working = kept

    log.info(
        "Feature selection: %d -> %d columns (dropped %d for >%.0f%% NaN, "
        "%d near-constant, %d redundant/correlated > %.2f)",
        len(candidate_cols), len(working),
        len(diagnostics["dropped_nan"]), max_nan_fraction * 100,
        len(diagnostics["dropped_variance"]),
        len(diagnostics["dropped_correlation"]), max_correlation,
    )
    return working, diagnostics


def pre_select_sensors(
    df: pd.DataFrame,
    sensor_cols: list[str],
    max_sensors: int = 100,
    max_nan_fraction: float = 0.5,
    min_variance: float = 1e-8,
    random_state: int = 42,
) -> list[str]:
    """Select top-N RAW sensors BEFORE rolling-stat feature engineering.

    This is specifically for Farm C (957 sensors). Without pre-selection,
    rolling stats on all 957 sensors × 3 windows × 2 stats = 5,742 new
    columns. With ~6 GB available RAM, this causes MemoryError.

    Strategy:
    1. Drop sensors with >max_nan_fraction missingness
    2. Drop near-constant sensors (variance < min_variance)
    3. From the survivors, rank by variance (higher = more informative)
    4. Keep top max_sensors

    This runs on a sample of the raw data (before any rolling), so it's
    fast and memory-light. The selected sensor list should be persisted
    in the model bundle for train/serve consistency.

    Args:
        df: raw SCADA data (before feature engineering)
        sensor_cols: list of sensor column names (excluding power, wind, metadata)
        max_sensors: max number of sensors to keep
        max_nan_fraction: drop sensors with NaN fraction above this
        min_variance: drop sensors with variance below this

    Returns:
        list of selected sensor column names
    """
    if len(sensor_cols) <= max_sensors:
        log.info("pre_select_sensors: %d sensors <= max %d, no pre-selection needed",
                 len(sensor_cols), max_sensors)
        return sensor_cols

    log.info("pre_select_sensors: reducing %d sensors to top %d",
             len(sensor_cols), max_sensors)

    # Sample if needed
    sample_df = df
    if len(df) > 50_000:
        sample_df = df.sample(n=50_000, random_state=random_state)

    # Step 1: Drop high-NaN
    present = [c for c in sensor_cols if c in sample_df.columns]
    nan_frac = sample_df[present].isna().mean()
    surviving = [c for c in present if nan_frac[c] <= max_nan_fraction]
    n_dropped_nan = len(present) - len(surviving)

    # Step 2: Drop near-constant
    variances = sample_df[surviving].var()
    surviving = [c for c in surviving if variances[c] >= min_variance]
    n_dropped_var = len(present) - n_dropped_nan - len(surviving) + n_dropped_nan

    # Step 3: Rank by variance, take top-N
    var_ranked = variances[surviving].sort_values(ascending=False)
    selected = list(var_ranked.head(max_sensors).index)

    log.info("pre_select_sensors: %d -> %d sensors "
             "(dropped %d high-NaN, %d near-constant, kept top-%d by variance)",
             len(sensor_cols), len(selected),
             n_dropped_nan, n_dropped_var, max_sensors)

    return selected
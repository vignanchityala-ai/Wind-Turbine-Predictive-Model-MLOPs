"""
features.py
============
Feature engineering for Wind Farm A SCADA data.

Design choices, and why:

1. We work per-sub-dataset (per turbine-episode), never mixing turbines'
   raw sensor scales together without normalization, since anonymized
   sensors aren't guaranteed comparable across turbines.
2. We only use status_type_id to LABEL points (normal vs abnormal) and to
   FILTER training data — never as a model input feature, since it would
   leak the label (status already encodes "something's wrong").
3. Power and wind-speed columns are the only semantically-known signals
   (per the CARE anonymization scheme) so we build physics-informed
   features from them (power curve residual, power/wind ratio) in
   addition to generic rolling statistics on all numeric sensors.
4. Rolling statistics are computed causally (no look-ahead) so the
   pipeline is valid for real-time / streaming deployment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def identify_columns(df: pd.DataFrame) -> dict:
    """Classify columns into feature / power / wind-speed / non-feature."""
    non_feature = set(config.NON_FEATURE_COLS)
    numeric_cols = [
        c for c in df.columns
        if c not in non_feature and pd.api.types.is_numeric_dtype(df[c])
    ]
    power_cols = [c for c in numeric_cols
                  if any(h in c.lower() for h in config.POWER_HINTS)]
    wind_cols = [c for c in numeric_cols
                 if any(h in c.lower() for h in config.WIND_SPEED_HINTS)]
    sensor_cols = [c for c in numeric_cols if c not in power_cols + wind_cols]
    return {
        "numeric": numeric_cols,
        "power": power_cols,
        "wind_speed": wind_cols,
        "sensor": sensor_cols,
    }


def clean_zeros_as_missing(df: pd.DataFrame, cols: list[str],
                            run_length_threshold: int = 6) -> pd.DataFrame:
    """
    Wind Farm B/C are documented to use 0 for missing values; Wind Farm A is
    generally cleaner, but long runs of exact zeros in a sensor column
    (longer than run_length_threshold consecutive points) are still very
    likely sensor dropout rather than a genuine physical zero, so we mask
    them to NaN and let downstream imputation handle it.
    """
    df = df.copy()
    for c in cols:
        is_zero = df[c] == 0
        # identify runs of consecutive zeros
        grp = (is_zero != is_zero.shift()).cumsum()
        run_lengths = is_zero.groupby(grp).transform("size")
        mask = is_zero & (run_lengths >= run_length_threshold)
        df.loc[mask, c] = np.nan
    return df


def fit_power_curve_reference(df: pd.DataFrame, power_col: str,
                               wind_col: str, bin_width: float = 0.5) -> dict:
    """
    Fit an empirical power-curve lookup (median power per wind-speed bin)
    from NORMAL, TRAINING-period rows only.

    IMPORTANT: fit this ONCE per turbine, at training time, and persist the
    returned reference alongside the trained model. At inference time, call
    apply_power_curve_reference() with this SAME reference — never refit
    from whatever short window of live data happens to arrive at serving
    time. Refitting at serve time is a classic train/serve skew bug: the
    "expected power" baseline would silently drift depending on what's in
    the request instead of staying anchored to the turbine's known-healthy
    baseline.
    """
    bins = np.arange(0, df[wind_col].max() + bin_width, bin_width)
    tmp = df.copy()
    tmp["_wind_bin"] = pd.cut(tmp[wind_col], bins=bins)

    normal_mask = tmp[config.STATUS_COL].isin(config.NORMAL_STATUS_IDS) \
        if config.STATUS_COL in tmp.columns else pd.Series(True, index=tmp.index)
    train_mask = (tmp[config.SPLIT_COL] == config.TRAIN_VALUE) \
        if config.SPLIT_COL in tmp.columns else pd.Series(True, index=tmp.index)
    ref_rows = tmp[normal_mask & train_mask]

    expected = ref_rows.groupby("_wind_bin", observed=True)[power_col].median()
    return {"bins": bins, "expected": expected, "power_col": power_col, "wind_col": wind_col}


def apply_power_curve_reference(df: pd.DataFrame, reference: dict) -> pd.DataFrame:
    """
    Apply a previously-fit power-curve reference (from fit_power_curve_reference)
    to any dataframe — training, evaluation, or a live inference window. Large
    negative residuals (producing much less power than expected for the wind
    speed) are a classic early indicator of degradation (blade fouling, pitch
    faults, gearbox issues, etc.).
    """
    df = df.copy()
    power_col, wind_col = reference["power_col"], reference["wind_col"]
    df["_wind_bin"] = pd.cut(df[wind_col], bins=reference["bins"])
    # .map() on a categorical Series can silently return a categorical-dtype
    # result instead of float (depends on how completely the reference
    # covers the bin categories) — force numeric so the subtraction below
    # doesn't break with "Object with dtype category cannot perform ... subtract".
    df["expected_power"] = df["_wind_bin"].map(reference["expected"]).astype("float64")
    df["power_residual"] = df[power_col] - df["expected_power"]
    df["power_residual_pct"] = df["power_residual"] / df["expected_power"].replace(0, np.nan)
    return df.drop(columns=["_wind_bin"])


def add_rolling_features(df: pd.DataFrame, cols: list[str],
                          windows: list[int] = None) -> pd.DataFrame:
    """Causal rolling mean/std for each numeric column, at multiple horizons."""
    windows = windows or config.ROLLING_WINDOWS
    df = df.copy()
    for w in windows:
        roll = df[cols].rolling(window=w, min_periods=max(2, w // 3))
        means = roll.mean().add_suffix(f"_roll{w}_mean")
        stds = roll.std().add_suffix(f"_roll{w}_std")
        df = pd.concat([df, means, stds], axis=1)
    return df


def _engineer_common(df: pd.DataFrame, power_curve_reference: dict | None) -> tuple[pd.DataFrame, list[str]]:
    """Shared feature-assembly steps used by both training and serving paths."""
    cols = identify_columns(df)
    df = clean_zeros_as_missing(df, cols["numeric"])

    if power_curve_reference is not None:
        df = apply_power_curve_reference(df, power_curve_reference)

    # Keep rolling features to a manageable subset for Wind Farm A (86 cols
    # is small enough to roll all numeric sensors; for B/C you'd want to
    # pre-select top-variance or domain-relevant sensors first).
    df = add_rolling_features(df, cols["numeric"])

    engineered_cols = [c for c in df.columns
                       if c not in config.NON_FEATURE_COLS
                       and c not in ("expected_power",)]
    engineered_cols = [c for c in engineered_cols if pd.api.types.is_numeric_dtype(df[c])]

    return df, engineered_cols


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict | None]:
    """
    Full feature-engineering pass for one sub-dataset AT TRAINING TIME: fits
    a fresh power-curve reference from this data's normal/training rows.

    Returns (augmented_df, list_of_model_feature_columns, power_curve_reference).
    Persist power_curve_reference alongside the trained model — the serving
    path (engineer_features_for_serving) needs the exact same reference to
    avoid train/serve skew, not a freshly-fit one.
    """
    cols = identify_columns(df)
    reference = None
    if cols["power"] and cols["wind_speed"]:
        reference = fit_power_curve_reference(df, cols["power"][0], cols["wind_speed"][0])

    df, engineered_cols = _engineer_common(df, reference)
    return df, engineered_cols, reference


def engineer_features_for_serving(df: pd.DataFrame,
                                   power_curve_reference: dict | None) -> tuple[pd.DataFrame, list[str]]:
    """
    Feature-engineering pass for LIVE INFERENCE: applies a previously-fit
    power_curve_reference (loaded from the persisted model bundle) instead
    of fitting a new one. Does not require status_type_id / train_test
    columns to be present, since live readings won't have them.

    df should be a window of recent readings sorted ascending by time_stamp,
    with at least max(config.ROLLING_WINDOWS) rows so rolling features on
    the most recent row reflect a full lookback.
    """
    return _engineer_common(df, power_curve_reference)


def build_label(df: pd.DataFrame) -> pd.Series:
    """
    Binary ground-truth label per row for scoring purposes:
    1 = abnormal status (per STATUS_MAP), 0 = normal status.
    This is the STATUS-based label, distinct from the model's anomaly score.
    """
    return (~df[config.STATUS_COL].isin(config.NORMAL_STATUS_IDS)).astype(int)

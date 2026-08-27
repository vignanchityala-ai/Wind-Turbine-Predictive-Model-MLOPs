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
   addition to generic rolling statistics on all numeric sensors. Column
   selection prefers a feature_descriptions.FeatureDescriptions (loaded
   from the dataset's own sensor-description file) when supplied, since a
   naive "contains 'power'" substring match wrongly catches reactive_power_*
   columns, and doesn't distinguish "Grid power" (delivered) from "Possible
   grid active power" (a capacity/theoretical figure) — see
   feature_descriptions.py for the reasoning. Falls back to the substring
   heuristic (reactive_power-excluded) when no description file is given,
   e.g. for the synthetic test data.
4. Rolling statistics are computed causally (no look-ahead) so the
   pipeline is valid for real-time / streaming deployment. ANGLE columns
   (wind direction, pitch angle, nacelle direction — flagged is_angle=TRUE
   in the sensor description file) get circular mean/std instead of linear:
   a linear rolling mean across the 359°/1° wrap gives ~180°, which is
   backwards. This is a real, named issue for this dataset — the source
   Zenodo record's "Known Data Issues" section calls out Pitch Angle
   wrapping specifically. Without a description file, angle columns aren't
   known and get treated as regular linear sensors (silently wrong near a
   wrap boundary) — this is a real limitation, not just a hedge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .feature_descriptions import FeatureDescriptions


def identify_columns(df: pd.DataFrame, feature_descriptions: FeatureDescriptions | None = None) -> dict:
    """
    Classify columns into feature / power / wind-speed / non-feature.

    If feature_descriptions is supplied, its heuristically-chosen
    power_base/wind_speed_base (e.g. "power_30" over "power_29" or any
    reactive_power_* column; measured "wind_speed_3" over estimated
    "wind_speed_4") are used to pick the actual matching dataframe columns.
    Otherwise falls back to substring matching — which now explicitly
    excludes "reactive" so reactive_power_27/28 don't get wrongly treated
    as the main power signal just because "power" is a substring of their
    name (this was a real bug once the actual Farm A column names were
    confirmed).
    """
    non_feature = set(config.NON_FEATURE_COLS)
    numeric_cols = [
        c for c in df.columns
        if c not in non_feature and pd.api.types.is_numeric_dtype(df[c])
    ]

    power_cols, wind_cols = [], []
    if feature_descriptions is not None:
        if feature_descriptions.power_base:
            power_cols = feature_descriptions.resolve(feature_descriptions.power_base, numeric_cols)
        if feature_descriptions.wind_speed_base:
            wind_cols = feature_descriptions.resolve(feature_descriptions.wind_speed_base, numeric_cols)

    if not power_cols:
        power_cols = [c for c in numeric_cols
                      if any(h in c.lower() for h in config.POWER_HINTS)
                      and "reactive" not in c.lower()]
    if not wind_cols:
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
                            run_length_threshold: int = 6,
                            exclude: list[str] = None) -> pd.DataFrame:
    """
    Wind Farm B/C are documented to use 0 for missing values; Wind Farm A is
    generally cleaner, but long runs of exact zeros in a sensor column
    (longer than run_length_threshold consecutive points) are still very
    likely sensor dropout rather than a genuine physical zero, so we mask
    them to NaN and let downstream imputation handle it.

    exclude: columns to skip this heuristic for entirely — pass power and
    wind-speed columns here. A sustained zero is completely physically
    plausible for those (a calm period genuinely has zero wind and zero
    power output for hours), so masking it to NaN would erase real "calm
    period" data and distort the power-curve reference with imputed
    (median) values instead of the true zero. This was a real bug: the
    heuristic was originally applied to every numeric column, including
    power and wind speed.
    """
    df = df.copy()
    exclude = set(exclude or [])
    for c in cols:
        if c in exclude:
            continue
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

    # Raise rather than silently default to "treat every row as normal
    # training data" if these are missing -- that would fit the "expected
    # power" baseline on unfiltered data, including anomalous/prediction
    # rows, defeating the whole point of the reference being a healthy
    # baseline. In the normal pipeline flow this can't happen (load_subdataset
    # already requires both columns), but fit_power_curve_reference is a
    # public function another caller could invoke directly with an
    # arbitrary dataframe -- better to fail loudly than fit silently wrong.
    if config.STATUS_COL not in tmp.columns or config.SPLIT_COL not in tmp.columns:
        raise ValueError(
            f"fit_power_curve_reference requires both {config.STATUS_COL!r} "
            f"and {config.SPLIT_COL!r} columns to identify normal training "
            f"rows -- found columns: {list(tmp.columns)[:10]}..."
        )
    normal_mask = tmp[config.STATUS_COL].isin(config.NORMAL_STATUS_IDS)
    train_mask = tmp[config.SPLIT_COL] == config.TRAIN_VALUE
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


def _rolling_circular_mean_std(series_degrees: pd.Series, window: int,
                                min_periods: int) -> tuple[pd.Series, pd.Series]:
    """
    Rolling circular mean/std for an angle column (degrees), via sin/cos
    decomposition — the standard technique for averaging directional data
    (used for wind direction in meteorology; same idea applies to pitch
    angle and nacelle direction here). A linear mean of e.g. 359° and 1°
    gives 180° (exactly backwards); the circular mean correctly gives ~0°.

    circular std uses the standard formula from Mardia's circular
    statistics: std = sqrt(-2 * ln(R)), where R is the mean resultant
    length (1.0 = all angles identical, 0.0 = uniformly spread).
    """
    radians = np.deg2rad(series_degrees.astype(float))
    sin_roll = np.sin(radians).rolling(window=window, min_periods=min_periods).mean()
    cos_roll = np.cos(radians).rolling(window=window, min_periods=min_periods).mean()

    mean_angle = np.rad2deg(np.arctan2(sin_roll, cos_roll)) % 360
    R = np.sqrt(sin_roll ** 2 + cos_roll ** 2).clip(upper=1.0)
    circ_std = np.rad2deg(np.sqrt(-2 * np.log(R.clip(lower=1e-9))))
    return mean_angle, circ_std


def add_rolling_features(df: pd.DataFrame, cols: list[str],
                          windows: list[int] = None,
                          angle_cols: list[str] = None) -> pd.DataFrame:
    """
    Causal rolling mean/std for each numeric column, at multiple horizons.
    Columns listed in angle_cols get circular mean/std (see
    _rolling_circular_mean_std) instead of linear — pass the actual
    resolved column names (e.g. from
    feature_descriptions.angle_columns(df.columns)), not base sensor names.

    Builds all new columns in a dict and concats once at the end, rather
    than once per window -- pd.concat copies the (growing) dataframe on
    every call, so doing it len(windows) times instead of once matters at
    Farm A scale (86 cols x 3 windows x 2 stats ~= 500 new columns) and
    matters more for Farm B/C's larger column counts.
    """
    windows = windows or config.ROLLING_WINDOWS
    angle_cols = set(angle_cols or [])
    linear_cols = [c for c in cols if c not in angle_cols]
    circular_cols = [c for c in cols if c in angle_cols]

    new_cols: dict[str, pd.Series] = {}
    for w in windows:
        min_periods = max(2, w // 3)
        if linear_cols:
            roll = df[linear_cols].rolling(window=w, min_periods=min_periods)
            means = roll.mean()
            stds = roll.std()
            for c in linear_cols:
                new_cols[f"{c}_roll{w}_mean"] = means[c]
                new_cols[f"{c}_roll{w}_std"] = stds[c]
        for c in circular_cols:
            mean_s, std_s = _rolling_circular_mean_std(df[c], w, min_periods)
            new_cols[f"{c}_roll{w}_circmean"] = mean_s
            new_cols[f"{c}_roll{w}_circstd"] = std_s

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def add_temporal_features(df: pd.DataFrame, sensor_cols: list[str],
                          lag_steps: list[int] = None) -> pd.DataFrame:
    """Add temporal features that capture time-dependent patterns.

    Three categories:
    1. Rate-of-change (first-order differences) for key sensors — captures
       sudden changes that rolling mean/std smooth over.
    2. Lag features (t-1, t-6, t-36) — gives the model access to recent
       history without the averaging of rolling windows.
    3. Cyclical time encoding (hour-of-day, day-of-week as sin/cos) —
       captures diurnal and weekly patterns in wind/power without the model
       needing to learn that hour 23 and hour 0 are adjacent.

    Uses at most 10 sensor columns for rate-of-change and lags to keep
    feature count manageable (especially for Farm C). Selects the first 10
    from sensor_cols — caller should pre-sort by importance if desired.
    """
    lag_steps = lag_steps or [1, 6, 36]  # 10min, 1h, 6h at 10-min resolution
    new_cols: dict[str, pd.Series] = {}

    # Limit to top sensors for rate/lag features to avoid explosion
    top_sensors = sensor_cols[:10]

    # 1. Rate-of-change (first-order difference)
    for c in top_sensors:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            new_cols[f"{c}_diff"] = df[c].diff()

    # 2. Lag features
    for lag in lag_steps:
        for c in top_sensors:
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                new_cols[f"{c}_lag{lag}"] = df[c].shift(lag)

    # 3. Cyclical time encoding
    if config.TIME_COL in df.columns:
        ts = pd.to_datetime(df[config.TIME_COL], errors="coerce")
        hour = ts.dt.hour + ts.dt.minute / 60.0
        dow = ts.dt.dayofweek.astype(float)

        new_cols["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        new_cols["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        new_cols["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        new_cols["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    if new_cols:
        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def _engineer_common(df: pd.DataFrame, power_curve_reference: dict | None,
                      feature_descriptions: FeatureDescriptions | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Shared feature-assembly steps used by both training and serving paths."""
    cols = identify_columns(df, feature_descriptions)
    df = clean_zeros_as_missing(df, cols["numeric"], exclude=cols["power"] + cols["wind_speed"])

    if power_curve_reference is not None:
        df = apply_power_curve_reference(df, power_curve_reference)

    angle_cols = feature_descriptions.angle_columns(cols["numeric"]) if feature_descriptions else []

    # Keep rolling features to a manageable subset for Wind Farm A (86 cols
    # is small enough to roll all numeric sensors; for B/C you'd want to
    # pre-select top-variance or domain-relevant sensors first).
    df = add_rolling_features(df, cols["numeric"], angle_cols=angle_cols)

    # Add temporal features (rate-of-change, lags, cyclical time)
    df = add_temporal_features(df, cols["sensor"])

    engineered_cols = [c for c in df.columns
                       if c not in config.NON_FEATURE_COLS
                       and c not in ("expected_power",)]
    engineered_cols = [c for c in engineered_cols if pd.api.types.is_numeric_dtype(df[c])]

    return df, engineered_cols


def engineer_features(df: pd.DataFrame,
                       feature_descriptions: FeatureDescriptions | None = None
                       ) -> tuple[pd.DataFrame, list[str], dict | None]:
    """
    Full feature-engineering pass for one sub-dataset AT TRAINING TIME: fits
    a fresh power-curve reference from this data's normal/training rows.

    feature_descriptions (optional): loaded via
    feature_descriptions.load_feature_descriptions(path) from the dataset's
    own sensor-description file. When supplied, drives (a) which column is
    treated as "the" power/wind-speed signal (avoiding the reactive-power
    mixup and preferring measured over estimated wind speed) and (b)
    circular vs linear rolling statistics for angle columns. Without it,
    the pipeline still runs (e.g. for the synthetic test data, which has no
    description file) but falls back to substring-matching for column
    selection and treats all columns as linear — angle columns, if any,
    will get silently-wrong rolling stats near the wrap boundary.

    Returns (augmented_df, list_of_model_feature_columns, power_curve_reference).
    Persist power_curve_reference alongside the trained model — the serving
    path (engineer_features_for_serving) needs the exact same reference to
    avoid train/serve skew, not a freshly-fit one. Persist feature_descriptions
    too (or at minimum its angle_bases/power_base/wind_speed_base) for the
    same reason — serving must resolve angle/power/wind columns identically
    to how they were resolved at training time.
    """
    cols = identify_columns(df, feature_descriptions)
    reference = None
    if cols["power"] and cols["wind_speed"]:
        if feature_descriptions is not None:
            power_col = feature_descriptions.pick_primary(feature_descriptions.power_base, cols["power"])
            wind_col = feature_descriptions.pick_primary(feature_descriptions.wind_speed_base, cols["wind_speed"])
        else:
            power_col, wind_col = cols["power"][0], cols["wind_speed"][0]
        if power_col and wind_col:
            reference = fit_power_curve_reference(df, power_col, wind_col)

    df, engineered_cols = _engineer_common(df, reference, feature_descriptions)
    return df, engineered_cols, reference


def engineer_features_for_serving(df: pd.DataFrame,
                                   power_curve_reference: dict | None,
                                   feature_descriptions: FeatureDescriptions | None = None
                                   ) -> tuple[pd.DataFrame, list[str]]:
    """
    Feature-engineering pass for LIVE INFERENCE: applies a previously-fit
    power_curve_reference (loaded from the persisted model bundle) instead
    of fitting a new one. Does not require status_type_id / train_test
    columns to be present, since live readings won't have them.

    Pass the SAME feature_descriptions used at training time (from the
    persisted bundle), so angle columns and power/wind-speed selection
    resolve identically at serve time — using a different or absent
    feature_descriptions here than at training time would silently change
    which columns get circular treatment vs. linear, and could pick a
    different power/wind column than the model was actually trained on.

    df should be a window of recent readings sorted ascending by time_stamp,
    with at least max(config.ROLLING_WINDOWS) rows so rolling features on
    the most recent row reflect a full lookback.
    """
    return _engineer_common(df, power_curve_reference, feature_descriptions)


def build_label(df: pd.DataFrame) -> pd.Series:
    """
    Binary ground-truth label per row for scoring purposes:
    1 = abnormal status (per STATUS_MAP), 0 = normal status.
    This is the STATUS-based label, distinct from the model's anomaly score.
    """
    return (~df[config.STATUS_COL].isin(config.NORMAL_STATUS_IDS)).astype(int)
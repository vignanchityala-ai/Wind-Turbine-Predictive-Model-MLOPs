"""
farm_pipeline.py
==================
"One model per farm" training strategy: pools the TRAINING-normal portion
of every sub-dataset in a farm into a single training set, fits one shared
model, then evaluates that same model separately against each
sub-dataset's own prediction period (so per-dataset Coverage/Reliability/
Earliness/Accuracy are still available, same as the per-turbine strategy).

This is a deliberate architectural choice, not the default: the CARE
benchmark's own design evaluates each sub-dataset as an independent
episode (see run_pipeline.process_one), and results from this farm-level
strategy are NOT comparable to published CARE-score numbers. Use this when
the goal is a maintainable production system (one model to retrain/monitor
per farm) rather than benchmark reproduction.

--- Turbine-awareness WITHOUT turbine identity as an input ---
Earlier versions of this module one-hot encoded asset_id as a model
feature. Reconsidered: that lets the model condition on WHO the turbine
is ("when asset_WT03=1, sensor_5 in range X is normal") rather than on
the underlying PHYSICAL behavior -- which both weakens generalization
(the model can lean on identity as a shortcut instead of learning
transferable physical relationships) and breaks down completely for a
turbine not seen during training (all-zero one-hot pattern, no learned
behavior for it at all).

The fix used here instead: z-score every plain sensor column relative to
THAT TURBINE'S OWN training-period baseline, before pooling across
turbines. The model then never sees a raw value or an identity label --
only "how many standard deviations is this reading from what's normal for
wherever it came from," a physically meaningful, turbine-agnostic
quantity every turbine's data is expressed in the same units of. This is
exactly the same principle already used for the power-curve residual (fit
per turbine, but the model only ever sees the residual, never which
turbine produced it) -- just extended to every other sensor instead of
stopping at power. A turbine unseen at training time simply needs *some*
baseline period to compute its own mean/std from at serve time -- no
retraining required to incorporate it, unlike one-hot columns which don't
exist for an unseen ID at all.

Angle columns (wind direction, pitch angle, nacelle direction) and
power/wind-speed are excluded from this normalization -- angles aren't
meaningfully z-scored, and power/wind-speed are already handled by the
power-curve mechanism specifically.

Structured as two explicit phases, kept separate on purpose:

  prepare_pooled_training_data()  -- DATA PREPARATION. Fits per-asset power
                                      curves and per-asset sensor
                                      normalization stats, checks for
                                      cross-dataset time-range overlaps,
                                      engineers features per dataset (on
                                      already-normalized sensor values),
                                      pools training-normal rows, runs
                                      feature selection. Produces a
                                      DataPreparationReport you can inspect
                                      BEFORE any model gets fit.

  train_farm_model()              -- MODEL TRAINING. Calls the above, then
                                      does the fit/validation split, fits
                                      the model, calibrates the threshold,
                                      and evaluates per-dataset.

Other key design decisions (unchanged from before):

1. Rolling-window statistics are still computed PER SUB-DATASET, never
   across the boundary between two datasets -- pooling happens AFTER
   feature engineering, only for the already-engineered training-normal
   rows. Computing a rolling window across a dataset boundary would mix
   unrelated time periods (possibly different turbines, possibly
   non-contiguous years) into a single window, which is never valid.

2. Overlapping training time ranges for the SAME turbine across different
   datasets are DETECTED and warned about, not silently deduplicated.

3. Feature selection runs on the POOLED training set (after concatenation),
   not per-dataset -- selection has to agree on ONE feature set for the
   ONE shared model.

4. The train/validation split (for threshold calibration) is done
   PER-DATASET first (tail-split, preserving "no future leakage" and
   avoiding autocorrelated rolling-window rows leaking across the split),
   THEN concatenated -- not a single random shuffle-split on the pooled
   data.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import pandas as pd

from . import config, data_loader, evaluation, feature_selection, features
from .feature_descriptions import FeatureDescriptions

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def check_for_overlapping_time_ranges(subs: list[data_loader.SubDataset]) -> list[str]:
    """
    For each asset_id, check whether any two sub-datasets' TRAINING periods
    overlap in time. Pooling overlapping windows would include duplicate
    (or near-duplicate) rows for that turbine during that period, over-
    weighting it relative to the rest of the pooled training set. Returns
    human-readable warning strings; empty list means no overlaps found.
    """
    by_asset: dict[str, list[tuple[str, pd.Timestamp, pd.Timestamp]]] = {}
    for sub in subs:
        if sub.asset_id is None:
            continue
        train = sub.train
        if len(train) == 0:
            continue
        start, end = train[config.TIME_COL].min(), train[config.TIME_COL].max()
        by_asset.setdefault(sub.asset_id, []).append((sub.name, start, end))

    warnings = []
    for asset, ranges in by_asset.items():
        ranges_sorted = sorted(ranges, key=lambda r: r[1])
        for i in range(len(ranges_sorted) - 1):
            name1, start1, end1 = ranges_sorted[i]
            name2, start2, end2 = ranges_sorted[i + 1]
            if end1 > start2:
                warnings.append(
                    f"asset_id={asset}: {name1} ({start1} to {end1}) and "
                    f"{name2} ({start2} to {end2}) have overlapping training "
                    f"periods from {max(start1, start2)} to {min(end1, end2)} "
                    f"-- pooling both will duplicate rows from this window."
                )
    return warnings


def fit_power_curve_references_per_asset(
    subs: list[data_loader.SubDataset], power_col: str, wind_col: str
) -> dict[str, dict]:
    """
    Fit one power-curve reference per asset_id (physical turbine), pooling
    that turbine's normal-training rows across every sub-dataset it
    appears in. A turbine's power curve is a property of the turbine
    itself (siting, physical characteristics) -- not of any one ~1-year
    episode -- so this pools across episodes for the same turbine, unlike
    the rolling-window features which must stay within one episode.
    """
    by_asset: dict[str, list[pd.DataFrame]] = {}
    for sub in subs:
        asset = sub.asset_id
        if asset is None:
            log.warning("%s has no asset_id -- excluded from power curve fitting.", sub.name)
            continue
        train = sub.train
        normal_train = train[train[config.STATUS_COL].isin(config.NORMAL_STATUS_IDS)]
        by_asset.setdefault(asset, []).append(normal_train)

    references = {}
    for asset, frames in by_asset.items():
        pooled = pd.concat(frames, ignore_index=True)
        if len(pooled) < 20:
            log.warning(
                "asset_id=%s has only %d pooled normal-training rows -- "
                "power curve reference may be unreliable.", asset, len(pooled),
            )
        references[asset] = features.fit_power_curve_reference(pooled, power_col, wind_col)
    return references


def fit_sensor_normalization_per_asset(
    subs: list[data_loader.SubDataset], sensor_cols: list[str]
) -> dict[str, dict[str, tuple[float, float]]]:
    """
    Fit (mean, std) per raw sensor column, per asset_id, from that asset's
    pooled NORMAL-training rows -- fit only on normal rows for the same
    reason everything else in this pipeline is: the baseline should
    reflect healthy behavior, not whatever's in the data indiscriminately.

    Used to z-score each turbine's sensor readings relative to ITS OWN
    baseline before pooling across turbines -- see module docstring for
    the full reasoning on why this replaces one-hot asset_id.
    """
    by_asset: dict[str, list[pd.DataFrame]] = {}
    for sub in subs:
        if sub.asset_id is None:
            continue
        train = sub.train
        normal_train = train[train[config.STATUS_COL].isin(config.NORMAL_STATUS_IDS)]
        by_asset.setdefault(sub.asset_id, []).append(normal_train)

    stats: dict[str, dict[str, tuple[float, float]]] = {}
    for asset, frames in by_asset.items():
        pooled = pd.concat(frames, ignore_index=True)
        if len(pooled) < 20:
            log.warning(
                "asset_id=%s has only %d pooled normal-training rows -- "
                "sensor normalization stats may be unreliable.", asset, len(pooled),
            )
        asset_stats = {}
        for col in sensor_cols:
            if col not in pooled.columns:
                continue
            mean = pooled[col].mean()
            std = pooled[col].std()
            # A near-zero std (constant sensor) would blow up the z-score;
            # fall back to std=1 so the column just becomes "value minus
            # its mean" instead of dividing by ~0. feature_selection's
            # variance check will likely drop a genuinely-constant column
            # anyway -- this just keeps the normalization step itself safe.
            asset_stats[col] = (float(mean), float(std) if pd.notna(std) and std > 1e-9 else 1.0)
        stats[asset] = asset_stats
    return stats


def apply_sensor_normalization(
    df: pd.DataFrame, asset_stats: dict[str, tuple[float, float]], sensor_cols: list[str]
) -> pd.DataFrame:
    """
    Z-score sensor_cols in df using this asset's fitted (mean, std) per
    column. If asset_stats is empty (asset unseen at training time) or a
    given column has no entry, that column is left unchanged -- there's no
    baseline to normalize against yet for a genuinely new turbine, the
    same fundamental limitation any turbine-aware approach has for a truly
    cold start (though unlike one-hot, once SOME baseline period exists
    for a new turbine, no retraining is needed to use it).
    """
    if not asset_stats:
        return df
    df = df.copy()
    for col in sensor_cols:
        if col in df.columns and col in asset_stats:
            mean, std = asset_stats[col]
            df[col] = (df[col] - mean) / std
    return df


# ---------------------------------------------------------------------------
# Phase 1: Data preparation
# ---------------------------------------------------------------------------
@dataclass
class DataPreparationReport:
    n_datasets: int
    known_assets: list[str]
    datasets_per_asset: dict[str, list[str]]
    overlap_warnings: list[str]
    pooled_row_count: int
    pooled_date_range: tuple
    per_asset_row_counts: dict[str, int]
    top_missingness: list[tuple[str, float]]
    n_sensors_normalized: int
    n_features_before_selection: int
    n_features_selected: int
    dropped_nan: list[str] = field(default_factory=list)
    dropped_variance: list[str] = field(default_factory=list)
    dropped_correlation: list[str] = field(default_factory=list)

    def log_summary(self) -> None:
        log.info("=== Data preparation summary ===")
        log.info("  Datasets: %d | Known turbines: %d | Pooled training-normal rows: %d",
                  self.n_datasets, len(self.known_assets), self.pooled_row_count)
        log.info("  Training date range (pooled, across all datasets): %s to %s",
                  *self.pooled_date_range)
        log.info("  Rows per turbine: %s", self.per_asset_row_counts)
        log.info("  Sensor columns normalized per-turbine (z-scored, no asset_id feature): %d",
                  self.n_sensors_normalized)
        if self.overlap_warnings:
            log.warning("  %d cross-dataset training-range overlap(s) found:", len(self.overlap_warnings))
            for w in self.overlap_warnings:
                log.warning("    %s", w)
        else:
            log.info("  No cross-dataset training-range overlaps detected.")
        if self.top_missingness:
            log.info("  Highest-missingness columns before selection: %s",
                      [(c, round(v, 3)) for c, v in self.top_missingness[:5]])
        log.info(
            "  Features: %d engineered -> %d after selection "
            "(dropped %d for NaN, %d for variance, %d for correlation)",
            self.n_features_before_selection, self.n_features_selected,
            len(self.dropped_nan), len(self.dropped_variance), len(self.dropped_correlation),
        )


def prepare_pooled_training_data(
    subs: list[data_loader.SubDataset],
    power_col: str | None,
    wind_col: str | None,
    feature_descriptions: FeatureDescriptions | None = None,
    do_feature_selection: bool = True,
) -> tuple[pd.DataFrame, dict[str, tuple[pd.DataFrame, list[str]]], list[str],
           dict[str, dict], dict[str, dict], list[str], DataPreparationReport]:
    """
    Explicit DATA PREPARATION phase, kept separate from model fitting so
    it's independently inspectable and testable (see module docstring for
    the full rationale). Returns:

      pooled_train_full   -- concatenated, selected-feature training-normal
                              rows across the whole farm
      engineered            -- {dataset_name: (full_engineered_df, feature_cols)}
                              for every sub-dataset, reused later for the
                              fit/val split and for prediction-period scoring
      known_assets           -- sorted list of every asset_id seen
      power_curve_refs       -- {asset_id: power curve reference}
      sensor_norm_stats      -- {asset_id: {sensor_col: (mean, std)}}
      feature_cols            -- final selected feature list (no asset_id
                              columns -- see module docstring)
      report                  -- DataPreparationReport
    """
    known_assets = sorted({s.asset_id for s in subs if s.asset_id is not None})
    if not known_assets:
        raise ValueError(
            "prepare_pooled_training_data: no sub-dataset has an asset_id. "
            "Check event_info was found correctly (see 'No event metadata "
            "found' warnings if any) -- asset_id can also come from the "
            "data CSV itself if missing from metadata, so this usually "
            "means something upstream is more broadly broken."
        )

    overlap_warnings = check_for_overlapping_time_ranges(subs)

    power_curve_refs = {}
    if power_col and wind_col:
        power_curve_refs = fit_power_curve_references_per_asset(subs, power_col, wind_col)
    else:
        log.warning("No power/wind-speed column identified -- skipping power-curve features entirely.")

    # Determine which columns get per-turbine normalized: every "plain
    # sensor" column (i.e. not power, not wind-speed, not an angle).
    # Schema is identical across every file in a farm, so this only needs
    # computing once, from any one sub-dataset.
    cols = features.identify_columns(subs[0].df, feature_descriptions)
    angle_cols = set(feature_descriptions.angle_columns(subs[0].df.columns)) if feature_descriptions else set()
    sensor_cols_to_normalize = [c for c in cols["sensor"] if c not in angle_cols]

    sensor_norm_stats = fit_sensor_normalization_per_asset(subs, sensor_cols_to_normalize)

    # Engineer features per sub-dataset (rolling stats never cross a
    # dataset boundary). Sensor normalization is applied to the RAW data
    # BEFORE feature engineering, so rolling stats are computed on
    # already-turbine-normalized values -- e.g. a rolling mean of z-scored
    # readings is itself a turbine-agnostic quantity, not a raw value that
    # would need normalizing again after the fact.
    engineered: dict[str, tuple[pd.DataFrame, list[str]]] = {}
    datasets_per_asset: dict[str, list[str]] = {}
    base_feature_cols: list[str] | None = None
    for sub in subs:
        datasets_per_asset.setdefault(sub.asset_id, []).append(sub.name)

        raw_df = sub.df
        if sensor_cols_to_normalize:
            raw_df = apply_sensor_normalization(
                raw_df, sensor_norm_stats.get(sub.asset_id, {}), sensor_cols_to_normalize)

        ref = power_curve_refs.get(sub.asset_id)
        df_feat, f_cols = features.engineer_features_for_serving(raw_df, ref, feature_descriptions)
        label = features.build_label(df_feat)
        df_feat = df_feat.assign(_status_label=label)

        # Downcast engineered feature columns to float32: roughly halves
        # the memory footprint of `engineered`, which holds every
        # dataset's FULL dataframe simultaneously (needed later for the
        # fit/val split and prediction-period scoring). At real farm
        # scale (22 datasets) this dict alone is estimated larger than the
        # feature_selection MemoryError that was confirmed on real data --
        # float32 has ~7 significant digits, comfortably more precision
        # than this feature engineering needs, and sklearn/IsolationForest
        # handle float32 input natively.
        for c in f_cols:
            if pd.api.types.is_float_dtype(df_feat[c]):
                df_feat[c] = df_feat[c].astype("float32")

        engineered[sub.name] = (df_feat, f_cols)
        if base_feature_cols is None:
            base_feature_cols = f_cols
        elif set(f_cols) != set(base_feature_cols):
            # Shouldn't happen within one farm (same schema everywhere),
            # but if it does, fail loudly rather than silently misalign
            # columns across datasets in the pooled set.
            raise ValueError(
                f"{sub.name} produced a different engineered-column set than "
                f"earlier datasets -- schema mismatch within the farm. "
                f"Diff: {set(f_cols) ^ set(base_feature_cols)}"
            )

    # Pool all training-normal rows.
    pooled_frames = []
    per_asset_row_counts: dict[str, int] = {}
    for sub in subs:
        df_feat, f_cols = engineered[sub.name]
        train_mask = df_feat[config.SPLIT_COL] == config.TRAIN_VALUE
        normal_mask = train_mask & (df_feat["_status_label"] == 0)
        subset = df_feat.loc[normal_mask, f_cols]
        pooled_frames.append(subset)
        per_asset_row_counts[sub.asset_id] = per_asset_row_counts.get(sub.asset_id, 0) + len(subset)
    pooled_train_full = pd.concat(pooled_frames, ignore_index=True)

    if len(pooled_train_full) < 50:
        raise ValueError(
            f"Only {len(pooled_train_full)} pooled normal-training rows "
            f"across the whole farm -- too few to train on."
        )

    # Missingness snapshot BEFORE selection, purely for visibility in the
    # report -- select_features() will actually act on this.
    nan_frac = pooled_train_full.isna().mean().sort_values(ascending=False)
    top_missingness = [(c, float(v)) for c, v in nan_frac.head(10).items() if v > 0]

    feature_cols = base_feature_cols
    n_before = len(feature_cols)
    selection_diag = {"dropped_nan": [], "dropped_variance": [], "dropped_correlation": []}
    if do_feature_selection:
        protect = [c for c in ("power_residual", "power_residual_pct") if c in feature_cols]
        feature_cols, selection_diag = feature_selection.select_features(
            pooled_train_full, feature_cols, protect=protect)
        pooled_train_full = pooled_train_full[feature_cols]

    all_starts = [sub.train[config.TIME_COL].min() for sub in subs if len(sub.train)]
    all_ends = [sub.train[config.TIME_COL].max() for sub in subs if len(sub.train)]
    date_range = (min(all_starts), max(all_ends)) if all_starts else (None, None)

    report = DataPreparationReport(
        n_datasets=len(subs),
        known_assets=known_assets,
        datasets_per_asset=datasets_per_asset,
        overlap_warnings=overlap_warnings,
        pooled_row_count=len(pooled_train_full),
        pooled_date_range=date_range,
        per_asset_row_counts=per_asset_row_counts,
        top_missingness=top_missingness,
        n_sensors_normalized=len(sensor_cols_to_normalize),
        n_features_before_selection=n_before,
        n_features_selected=len(feature_cols),
        dropped_nan=selection_diag["dropped_nan"],
        dropped_variance=selection_diag["dropped_variance"],
        dropped_correlation=selection_diag["dropped_correlation"],
    )
    report.log_summary()

    return pooled_train_full, engineered, known_assets, power_curve_refs, sensor_norm_stats, feature_cols, report


# ---------------------------------------------------------------------------
# Phase 2: Model training
# ---------------------------------------------------------------------------
@dataclass
class FarmModelResult:
    detector: object
    threshold: float
    feature_cols: list[str]
    power_curve_references: dict[str, dict]
    sensor_normalization: dict[str, dict]
    known_assets: list[str]
    per_dataset_results: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    prep_report: DataPreparationReport | None = None


def train_farm_model(
    subs: list[data_loader.SubDataset],
    model_kind: str,
    feature_descriptions: FeatureDescriptions | None = None,
    do_feature_selection: bool = True,
) -> FarmModelResult:
    """Train ONE model on pooled training-normal data across all subs,
    then evaluate it separately against each sub's own prediction period."""
    from . import model as model_module
    t0 = time.time()

    if not subs:
        raise ValueError("train_farm_model: no sub-datasets given")

    # power_col/wind_col are determined once -- every Farm A file shares
    # the same raw schema, so this is safe to compute from just the first.
    cols = features.identify_columns(subs[0].df, feature_descriptions)
    if feature_descriptions is not None and cols["power"] and cols["wind_speed"]:
        power_col = feature_descriptions.pick_primary(feature_descriptions.power_base, cols["power"])
        wind_col = feature_descriptions.pick_primary(feature_descriptions.wind_speed_base, cols["wind_speed"])
    else:
        power_col = cols["power"][0] if cols["power"] else None
        wind_col = cols["wind_speed"][0] if cols["wind_speed"] else None

    pooled_train_full, engineered, known_assets, power_curve_refs, sensor_norm_stats, feature_cols, prep_report = \
        prepare_pooled_training_data(subs, power_col, wind_col, feature_descriptions, do_feature_selection)

    # Fit/validation split, PER DATASET first (tail-split, same anti-
    # leakage principle as the per-turbine strategy), THEN concatenated --
    # keeps validation representative across all turbines instead of
    # dominated by whichever dataset sorts last. Uses the final selected
    # feature_cols from preparation.
    train_fit_frames, train_val_frames = [], []
    for sub in subs:
        df_feat, _ = engineered[sub.name]
        train_mask = df_feat[config.SPLIT_COL] == config.TRAIN_VALUE
        normal_mask = train_mask & (df_feat["_status_label"] == 0)
        sub_train = df_feat.loc[normal_mask, feature_cols]
        if len(sub_train) < 2:
            continue
        n_val = max(1, int(len(sub_train) * config.VALIDATION_FRACTION))
        train_fit_frames.append(sub_train.iloc[:-n_val])
        train_val_frames.append(sub_train.iloc[-n_val:])

    train_fit = pd.concat(train_fit_frames, ignore_index=True)
    train_val = pd.concat(train_val_frames, ignore_index=True)

    if model_kind == "isolation_forest":
        clf = model_module.IsolationForestDetector()
        clf.fit(train_fit)
    elif model_kind == "autoencoder":
        clf = model_module.AutoencoderDetector(input_dim=len(feature_cols))
        clf.fit(train_fit, X_val=train_val, epochs=40, verbose=0)
    else:
        raise ValueError(f"Unknown model_kind: {model_kind}")

    val_scores = clf.score(train_val)
    threshold = model_module.calibrate_threshold(val_scores)

    # Evaluate the ONE shared model separately against each dataset's own
    # prediction period -- reuses evaluation.py unchanged, so
    # Coverage/Reliability/Earliness/Accuracy mean exactly the same thing
    # they do in the per-turbine strategy; only how the model was TRAINED
    # differs.
    results = []
    for sub in subs:
        df_feat, _ = engineered[sub.name]
        pred_df = df_feat.loc[df_feat[config.SPLIT_COL] == config.PREDICTION_VALUE]
        if len(pred_df) == 0:
            log.warning("%s: no prediction-period rows, skipping evaluation.", sub.name)
            continue
        pred_scores = clf.score(pred_df[feature_cols])
        pred_flags = model_module.scores_to_events(pred_scores, threshold)
        result = evaluation.evaluate_subdataset(
            name=sub.name,
            is_anomaly=sub.is_anomaly,
            status_ids=pred_df[config.STATUS_COL],
            predicted_flags=pred_flags,
            timestamps=pred_df[config.TIME_COL],
            fault_onset=sub.event_end,
        )
        results.append(result)

    diagnostics = {
        "n_datasets_pooled": len(subs),
        "n_known_assets": len(known_assets),
        "n_features_before_selection": prep_report.n_features_before_selection,
        "n_features": len(feature_cols),
        "n_dropped_nan": len(prep_report.dropped_nan),
        "n_dropped_variance": len(prep_report.dropped_variance),
        "n_dropped_correlation": len(prep_report.dropped_correlation),
        "n_train_fit_rows": len(train_fit),
        "n_train_val_rows": len(train_val),
        "n_overlap_warnings": len(prep_report.overlap_warnings),
        "threshold": threshold,
        "runtime_sec": round(time.time() - t0, 1),
    }
    log.info(
        "Farm model trained: %d datasets pooled, %d known assets, %d "
        "features, %d fit rows, %d val rows (%.1fs total incl. data prep)",
        diagnostics["n_datasets_pooled"], diagnostics["n_known_assets"],
        len(feature_cols), diagnostics["n_train_fit_rows"],
        diagnostics["n_train_val_rows"], diagnostics["runtime_sec"],
    )

    return FarmModelResult(
        detector=clf,
        threshold=threshold,
        feature_cols=feature_cols,
        power_curve_references=power_curve_refs,
        sensor_normalization=sensor_norm_stats,
        known_assets=known_assets,
        per_dataset_results=results,
        diagnostics=diagnostics,
        prep_report=prep_report,
    )
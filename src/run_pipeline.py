"""
run_pipeline.py
================
End-to-end script: for each Wind Farm A sub-dataset,
  1. load + engineer features
  2. train a per-turbine Normal Behavior Model on that turbine's training
     year (normal-status points only)
  3. score the prediction period
  4. calibrate a threshold from held-out normal validation data
  5. collapse scores into sustained "events"
  6. evaluate against the documented anomaly/normal label
Then aggregates results across all sub-datasets and writes a report.

Usage:
    python -m src.run_pipeline --model autoencoder
    python -m src.run_pipeline --model isolation_forest --raw-dir "/path/to/Wind Farm A"

Prerequisite: download the dataset from
https://www.kaggle.com/datasets/azizkasimov/wind-turbine-scada-data-for-early-fault-detection
and extract "Wind Farm A" so that config.RAW_DATA_DIR (or --raw-dir) points
at a folder containing the per-turbine CSVs (+ event_info files if present).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, data_loader, evaluation, features, feature_selection, model as model_module

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pipeline")


def process_one(sub: data_loader.SubDataset, model_kind: str,
                 save_model: bool = False,
                 model_dir: Path = config.MODEL_DIR,
                 feature_descriptions=None,
                 do_feature_selection: bool = True) -> tuple[evaluation.SubDatasetResult, dict]:
    """Run the full fit/score/evaluate cycle for a single sub-dataset."""
    t0 = time.time()
    df, feature_cols, power_curve_reference = features.engineer_features(
        sub.df, feature_descriptions=feature_descriptions)
    label = features.build_label(df)  # 1 = abnormal status
    df = df.assign(_status_label=label)

    train_mask = df[config.SPLIT_COL] == config.TRAIN_VALUE
    normal_train_mask = train_mask & (df["_status_label"] == 0)

    train_full = df.loc[normal_train_mask, feature_cols]
    if len(train_full) < 50:
        raise ValueError(f"{sub.name}: not enough normal training rows ({len(train_full)})")

    n_features_before_selection = len(feature_cols)
    selection_diag = None
    if do_feature_selection:
        # Fit selection ONLY on train_full (training-normal rows) -- never
        # on the full df or the prediction period, or this leaks what
        # "typical" looks like from data the model shouldn't have seen yet.
        # Physically-meaningful power-curve features are protected from the
        # correlation prune even if they happen to correlate with something
        # else in a given dataset.
        protect = [c for c in ("power_residual", "power_residual_pct") if c in feature_cols]
        feature_cols, selection_diag = feature_selection.select_features(
            train_full, feature_cols, protect=protect)
        # Reassigning feature_cols here is what makes the selected subset
        # flow through everywhere downstream automatically: train_fit,
        # train_val, pred_df[feature_cols], and the saved model bundle all
        # use this same variable -- no separate persistence needed, exactly
        # like power_curve_reference and feature_descriptions.
        train_full = train_full[feature_cols]

    # Split training-normal data into fit / threshold-calibration slices
    n_val = max(1, int(len(train_full) * config.VALIDATION_FRACTION))
    train_fit = train_full.iloc[:-n_val]
    train_val = train_full.iloc[-n_val:]

    if model_kind == "autoencoder":
        clf = model_module.AutoencoderDetector(input_dim=len(feature_cols))
        clf.fit(train_fit, X_val=train_val, epochs=40, verbose=0)
    elif model_kind == "isolation_forest":
        clf = model_module.IsolationForestDetector()
        clf.fit(train_fit)
    else:
        raise ValueError(f"Unknown model_kind: {model_kind}")

    val_scores = clf.score(train_val)
    threshold = model_module.calibrate_threshold(val_scores)

    pred_df = df.loc[df[config.SPLIT_COL] == config.PREDICTION_VALUE]
    if len(pred_df) == 0:
        raise ValueError(f"{sub.name}: no prediction-period rows")

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

    if save_model:
        if model_kind != "isolation_forest":
            log.warning(
                "  Skipping model save for %s: persistence is currently only "
                "wired up for --model isolation_forest (sklearn objects are "
                "joblib-safe; the Keras autoencoder needs its own model.save() "
                "path, not yet implemented here).", sub.name,
            )
        else:
            import joblib
            model_dir.mkdir(parents=True, exist_ok=True)
            bundle = {
                "detector": clf,
                "feature_cols": feature_cols,
                "threshold": threshold,
                "power_curve_reference": power_curve_reference,
                # Persisted so serve.py resolves angle/power/wind columns
                # identically to how training resolved them -- using a
                # different (or absent) feature_descriptions at serve time
                # than at train time would silently change which columns get
                # circular treatment and which column counts as "the" power
                # signal, the same train/serve-skew risk as power_curve_reference.
                "feature_descriptions": feature_descriptions,
                "asset_id": sub.asset_id,
                "dataset_name": sub.name,
                "model_kind": model_kind,
                "min_event_length": config.MIN_EVENT_LENGTH,
            }
            joblib.dump(bundle, model_dir / f"{sub.name}.joblib")

    diagnostics = {
        "n_features": len(feature_cols),
        "n_features_before_selection": n_features_before_selection,
        "n_dropped_nan": len(selection_diag["dropped_nan"]) if selection_diag else 0,
        "n_dropped_variance": len(selection_diag["dropped_variance"]) if selection_diag else 0,
        "n_dropped_correlation": len(selection_diag["dropped_correlation"]) if selection_diag else 0,
        "n_train_rows": len(train_fit),
        "n_val_rows": len(train_val),
        "n_pred_rows": len(pred_df),
        "threshold": threshold,
        "runtime_sec": round(time.time() - t0, 1),
    }
    return result, diagnostics


def main():
    parser = argparse.ArgumentParser(description="Wind Farm A early fault detection pipeline")
    parser.add_argument("--raw-dir", type=str, default=str(config.RAW_DATA_DIR),
                        help="Folder containing Wind Farm A sub-dataset CSVs")
    parser.add_argument("--model", type=str, default="isolation_forest",
                        choices=["isolation_forest", "autoencoder"],
                        help="isolation_forest needs only scikit-learn; "
                             "autoencoder needs tensorflow")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N sub-datasets (debugging)")
    parser.add_argument("--output", type=str,
                        default=str(config.OUTPUT_DIR / "evaluation_report.csv"))
    parser.add_argument("--save-models", action="store_true", default=False,
                        help="Persist each turbine's fitted model bundle (joblib) "
                             "to --model-dir, for use by src/serve.py. Currently "
                             "only supported for --model isolation_forest.")
    parser.add_argument("--model-dir", type=str, default=str(config.MODEL_DIR))
    parser.add_argument("--feature-description-path", type=str, default=None,
                        help="Path to the dataset's sensor/feature description "
                             "file (columns: sensor_name, statistics_type, "
                             "description, unit, is_angle, is_counter). When "
                             "given, drives angle-aware circular rolling stats "
                             "and correct power/wind-speed column selection "
                             "(avoiding the reactive-power mixup). Without it, "
                             "the pipeline still runs but falls back to substring "
                             "matching and treats all columns as linear.")
    parser.add_argument("--event-info-path", type=str, default=None,
                        help="Explicit path to the event_info file (e.g. "
                             "comma_event_info.csv), read directly instead of "
                             "searched for. Use this if the automatic search "
                             "(same folder as the data CSVs, or one level up) "
                             "doesn't find it -- real downloads have been "
                             "observed to place it somewhere unexpected, e.g. "
                             "a completely different extracted folder than "
                             "--raw-dir. Without this, every dataset will "
                             "silently get event_label=None if the search "
                             "fails, which breaks is_anomaly and evaluation "
                             "for ALL datasets, not just some -- check the log "
                             "for 'No event metadata found' warnings.")
    parser.add_argument("--no-feature-selection", action="store_true", default=False,
                        help="Disable feature selection (drops near-constant, "
                             "mostly-missing, and highly-correlated engineered "
                             "columns by default). Selection is fit only on "
                             "training-normal rows and the reduced column set "
                             "is what actually gets used everywhere downstream "
                             "-- pass this flag to keep the full engineered "
                             "feature set instead.")
    parser.add_argument("--training-strategy", type=str, default="turbine",
                        choices=["turbine", "farm"],
                        help="'turbine' (default): one model per sub-dataset, "
                             "matching the CARE benchmark's own evaluation "
                             "design -- use this if CARE-score comparability "
                             "matters. 'farm': one shared model trained on "
                             "pooled training-normal data across every "
                             "sub-dataset in the farm, with asset_id as a "
                             "one-hot feature and a per-turbine power-curve "
                             "reference so turbine identity isn't averaged "
                             "away -- use this for a single maintainable "
                             "production model per farm. Either way, "
                             "per-dataset Coverage/Reliability/Earliness are "
                             "still reported the same way in the output.")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    log.info("Discovering sub-datasets under %s", raw_dir)
    paths = data_loader.discover_subdatasets(raw_dir)
    if args.limit:
        paths = paths[: args.limit]
    log.info("Found %d sub-datasets", len(paths))

    feature_descriptions = None
    if args.feature_description_path:
        from . import feature_descriptions as fd_module
        feature_descriptions = fd_module.load_feature_descriptions(args.feature_description_path)
        log.info(
            "Loaded feature descriptions: %d angle column(s), %d counter column(s), "
            "power_base=%s, wind_speed_base=%s",
            len(feature_descriptions.angle_bases), len(feature_descriptions.counter_bases),
            feature_descriptions.power_base, feature_descriptions.wind_speed_base,
        )
    else:
        log.warning(
            "No --feature-description-path given: angle columns (wind "
            "direction, pitch angle, nacelle direction) will get linear "
            "rolling stats instead of circular, which is wrong near the "
            "0/360 wrap. Power/wind-speed column selection also falls back "
            "to substring matching. Pass the dataset's sensor description "
            "file to fix both.",
        )

    if args.training_strategy == "farm":
        results, diag_rows = run_farm_strategy(paths, args, feature_descriptions)
    else:
        results, diag_rows = run_turbine_strategy(paths, args, feature_descriptions)

    if not results:
        log.error("No sub-datasets processed successfully. Exiting.")
        sys.exit(1)

    report_df = evaluation.aggregate_results(results)
    summary = evaluation.summarize(report_df)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(out_path, index=False)

    diag_df = pd.DataFrame(diag_rows)
    diag_df.to_csv(out_path.with_name(out_path.stem + "_diagnostics.csv"), index=False)

    log.info("=" * 60)
    log.info("SUMMARY (%s model, %s strategy, %d sub-datasets)",
              args.model, args.training_strategy, len(results))
    for k, v in summary.items():
        log.info("  %-28s %s", k, _fmt(v))
    log.info("Per-dataset report written to: %s", out_path)
    if args.save_models:
        log.info("Model bundles saved to: %s", args.model_dir)
    log.info("=" * 60)


def run_turbine_strategy(paths, args, feature_descriptions):
    """One model per sub-dataset (the original/default behavior, matching
    the CARE benchmark's own per-episode evaluation design)."""
    results, diag_rows = [], []
    n_no_label = 0
    for i, p in enumerate(paths, 1):
        log.info("[%d/%d] Processing %s", i, len(paths), p.name)
        try:
            sub = data_loader.load_subdataset(p, event_info_path=args.event_info_path)
            if sub.event_label is None:
                n_no_label += 1

            if i == 1 and feature_descriptions is not None:
                mismatches = feature_descriptions.validate_against_columns(sub.df.columns)
                if mismatches:
                    log.warning(
                        "Feature description validation found %d mismatch(es) "
                        "against %s's actual columns -- the column-naming "
                        "convention assumption may be wrong for some sensors:",
                        len(mismatches), p.name,
                    )
                    for m in mismatches:
                        log.warning("  %s", m)
                else:
                    log.info("Feature description validation: all sensor column counts match.")

            res, diag = process_one(sub, args.model, save_model=args.save_models,
                                     model_dir=Path(args.model_dir),
                                     feature_descriptions=feature_descriptions,
                                     do_feature_selection=not args.no_feature_selection)
            results.append(res)
            diag["dataset"] = sub.name
            diag_rows.append(diag)
            log.info(
                "  -> anomaly=%s coverage=%s reliability=%s earliness_h=%s accuracy=%s (%.1fs)",
                res.is_anomaly, _fmt(res.coverage), _fmt(res.reliability),
                _fmt(res.earliness_hours), _fmt(res.accuracy), diag["runtime_sec"],
            )
        except Exception as e:
            log.error("  FAILED on %s: %s", p.name, e)

    if len(paths) > 1 and n_no_label == len(paths):
        log.error(
            "*** %d/%d datasets had NO event metadata found at all -- every "
            "single is_anomaly result below is meaningless (defaults to "
            "False when event_label is unknown). This is almost certainly a "
            "path problem, not a real finding: the event_info file wasn't "
            "found by search, or --event-info-path points at the wrong "
            "file. Check the 'No event metadata found' warnings above for "
            "exactly which paths were checked, locate the real file, and "
            "re-run with --event-info-path pointing at it directly. ***",
            n_no_label, len(paths),
        )
    return results, diag_rows


def run_farm_strategy(paths, args, feature_descriptions):
    """One shared model for the whole farm -- see farm_pipeline.py for the
    full design rationale (per-asset power curves, asset_id one-hot,
    pooled-then-selected features, per-dataset tail-split before pooling)."""
    from . import farm_pipeline

    log.info("Loading all %d sub-datasets for pooled farm-level training...", len(paths))
    subs = []
    n_no_label = 0
    for p in paths:
        try:
            sub = data_loader.load_subdataset(p, event_info_path=args.event_info_path)
            if sub.event_label is None:
                n_no_label += 1
            subs.append(sub)
        except Exception as e:
            log.error("  FAILED loading %s: %s", p.name, e)

    if len(subs) > 1 and n_no_label == len(subs):
        log.error(
            "*** %d/%d datasets had NO event metadata found at all -- every "
            "is_anomaly result will be meaningless. Same diagnosis as the "
            "turbine strategy: check --event-info-path. ***", n_no_label, len(subs),
        )

    if not subs:
        return [], []

    farm_result = farm_pipeline.train_farm_model(
        subs, args.model,
        feature_descriptions=feature_descriptions,
        do_feature_selection=not args.no_feature_selection,
    )

    prep_report_path = Path(args.output).parent / "data_preparation_report.txt"
    prep_report_path.parent.mkdir(parents=True, exist_ok=True)
    r = farm_result.prep_report
    with open(prep_report_path, "w") as f:
        f.write("Data Preparation Report -- farm training strategy\n")
        f.write("=" * 60 + "\n")
        f.write(f"Datasets pooled: {r.n_datasets}\n")
        f.write(f"Known turbines ({len(r.known_assets)}): {r.known_assets}\n")
        f.write(f"Datasets per turbine: {r.datasets_per_asset}\n")
        f.write(f"Training date range (pooled): {r.pooled_date_range[0]} to {r.pooled_date_range[1]}\n")
        f.write(f"Pooled training-normal rows: {r.pooled_row_count}\n")
        f.write(f"Rows per turbine: {r.per_asset_row_counts}\n\n")
        f.write(f"Cross-dataset training-range overlaps: {len(r.overlap_warnings)}\n")
        for w in r.overlap_warnings:
            f.write(f"  - {w}\n")
        f.write("\n")
        f.write(f"Highest-missingness columns (before selection): {r.top_missingness}\n\n")
        f.write(f"Features: {r.n_features_before_selection} engineered -> {r.n_features_selected} after selection\n")
        f.write(f"  Dropped for >50% missingness ({len(r.dropped_nan)}): {r.dropped_nan}\n")
        f.write(f"  Dropped for near-zero variance ({len(r.dropped_variance)}): {r.dropped_variance}\n")
        f.write(f"  Dropped for correlation > 0.95 ({len(r.dropped_correlation)}): {r.dropped_correlation}\n")
    log.info("Data preparation report written to: %s", prep_report_path)

    if args.save_models:
        if args.model != "isolation_forest":
            log.warning(
                "Skipping farm model save: persistence is currently only "
                "wired up for --model isolation_forest.",
            )
        else:
            import joblib
            model_dir = Path(args.model_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            farm_name = Path(args.raw_dir).name or "farm"
            bundle = {
                "detector": farm_result.detector,
                "feature_cols": farm_result.feature_cols,
                "threshold": farm_result.threshold,
                "power_curve_references": farm_result.power_curve_references,
                "sensor_normalization": farm_result.sensor_normalization,
                "known_assets": farm_result.known_assets,
                "feature_descriptions": feature_descriptions,
                "farm_name": farm_name,
                "model_kind": args.model,
                "min_event_length": config.MIN_EVENT_LENGTH,
                "training_strategy": "farm",
            }
            bundle_path = model_dir / f"{farm_name.replace(' ', '_')}_farm_model.joblib"
            joblib.dump(bundle, bundle_path)
            log.info("Farm model bundle saved to: %s", bundle_path)
            log.info(
                "NOTE: src/serve.py does not yet load farm-strategy bundles "
                "(it expects one bundle per dataset_name) -- serving needs "
                "a follow-up update to handle this bundle shape "
                "(asset_id-scoped /predict, per-asset power curve lookup, "
                "per-asset sensor normalization applied at request time) "
                "before this can be deployed.",
            )

    # Build diag_rows in the SAME shape as the turbine strategy, one row
    # per dataset, even though most values are identical across rows here
    # (they all came from the one shared training run) -- keeps the
    # diagnostics CSV schema uniform regardless of which strategy produced it.
    diag_rows = []
    for r in farm_result.per_dataset_results:
        diag_rows.append({
            "dataset": r.name,
            "n_features": farm_result.diagnostics["n_features"],
            "n_features_before_selection": farm_result.diagnostics["n_features_before_selection"],
            "n_dropped_nan": farm_result.diagnostics["n_dropped_nan"],
            "n_dropped_variance": farm_result.diagnostics["n_dropped_variance"],
            "n_dropped_correlation": farm_result.diagnostics["n_dropped_correlation"],
            "n_train_rows": farm_result.diagnostics["n_train_fit_rows"],
            "n_val_rows": farm_result.diagnostics["n_train_val_rows"],
            "threshold": farm_result.threshold,
            "runtime_sec": farm_result.diagnostics["runtime_sec"],
        })

    log.info(
        "Farm model summary: %d datasets pooled, %d known turbines, "
        "%d features selected, threshold=%.4f",
        farm_result.diagnostics["n_datasets_pooled"], len(farm_result.known_assets),
        farm_result.diagnostics["n_features"], farm_result.threshold,
    )
    return farm_result.per_dataset_results, diag_rows


def _fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


if __name__ == "__main__":
    main()
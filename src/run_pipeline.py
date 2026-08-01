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

from . import config, data_loader, evaluation, features, model as model_module

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pipeline")


def process_one(sub: data_loader.SubDataset, model_kind: str,
                 save_model: bool = False,
                 model_dir: Path = config.MODEL_DIR) -> tuple[evaluation.SubDatasetResult, dict]:
    """Run the full fit/score/evaluate cycle for a single sub-dataset."""
    t0 = time.time()
    df, feature_cols, power_curve_reference = features.engineer_features(sub.df)
    label = features.build_label(df)  # 1 = abnormal status
    df = df.assign(_status_label=label)

    train_mask = df[config.SPLIT_COL] == config.TRAIN_VALUE
    normal_train_mask = train_mask & (df["_status_label"] == 0)

    train_full = df.loc[normal_train_mask, feature_cols]
    if len(train_full) < 50:
        raise ValueError(f"{sub.name}: not enough normal training rows ({len(train_full)})")

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
                "asset_id": sub.asset_id,
                "dataset_name": sub.name,
                "model_kind": model_kind,
                "min_event_length": config.MIN_EVENT_LENGTH,
            }
            joblib.dump(bundle, model_dir / f"{sub.name}.joblib")

    diagnostics = {
        "n_features": len(feature_cols),
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
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    log.info("Discovering sub-datasets under %s", raw_dir)
    paths = data_loader.discover_subdatasets(raw_dir)
    if args.limit:
        paths = paths[: args.limit]
    log.info("Found %d sub-datasets", len(paths))

    results, diag_rows = [], []
    for i, p in enumerate(paths, 1):
        log.info("[%d/%d] Processing %s", i, len(paths), p.name)
        try:
            sub = data_loader.load_subdataset(p)
            res, diag = process_one(sub, args.model, save_model=args.save_models,
                                     model_dir=Path(args.model_dir))
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
    log.info("SUMMARY (%s model, %d sub-datasets)", args.model, len(results))
    for k, v in summary.items():
        log.info("  %-28s %s", k, _fmt(v))
    log.info("Per-dataset report written to: %s", out_path)
    if args.save_models:
        log.info("Model bundles saved to: %s", args.model_dir)
    log.info("=" * 60)


def _fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


if __name__ == "__main__":
    main()

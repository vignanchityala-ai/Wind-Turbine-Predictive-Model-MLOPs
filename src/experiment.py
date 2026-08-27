"""
experiment.py
=============
Phase 3: Fair model comparison on Gold data.
Trains each model type on the same data, evaluates with same CARE metrics.
"""

import logging
import time
from pathlib import Path
import json

import pandas as pd
import numpy as np

from src import config, evaluation, feature_selection
from src.model import IsolationForestDetector, AutoencoderDetector, LSTMAutoencoderDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("experiment")

def load_gold_dataset(farm_id: str, dataset_id: str):
    gold_dir = config.GOLD_DIR / f"farm={farm_id}"
    parquet_path = gold_dir / f"dataset_{dataset_id}.parquet"
    meta_path = gold_dir / f"dataset_{dataset_id}_meta.json"
    
    if not parquet_path.exists():
        raise FileNotFoundError(f"Gold Parquet not found: {parquet_path}")
        
    df = pd.read_parquet(parquet_path)
    with open(meta_path) as f:
        meta = json.load(f)
        
    return df, meta["feature_columns"]

def get_event_end(farm_id: str, dataset_id: str):
    from src.data.schema import find_event_info_file, parse_event_info
    raw_dir = config.FARM_CONFIGS[farm_id]["raw_dir"]
    event_info_path = find_event_info_file(raw_dir)
    farm_metadata = parse_event_info(farm_id, event_info_path)
    event = farm_metadata.get_event(str(dataset_id))
    return event.event_end if event else None

def run_experiment_on_dataset(farm_id: str, dataset_id: str):
    log.info("Starting experiment on %s dataset %s", farm_id, dataset_id)
    df, feature_cols = load_gold_dataset(farm_id, dataset_id)
    
    if "_status_label" not in df.columns:
        from src.features import build_label
        df["_status_label"] = build_label(df)
        
    train_mask = df[config.SPLIT_COL] == config.TRAIN_VALUE
    normal_train_mask = train_mask & (df["_status_label"] == 0)
    
    train_full = df.loc[normal_train_mask, feature_cols]
    
    # Feature selection
    protect = [c for c in ("power_residual", "power_residual_pct") if c in feature_cols]
    selected_cols, _ = feature_selection.select_features(train_full, feature_cols, protect=protect)
    train_full = train_full[selected_cols]
    
    # Split training-normal data into fit / threshold-calibration slices
    n_val = max(1, int(len(train_full) * config.VALIDATION_FRACTION))
    train_fit = train_full.iloc[:-n_val]
    train_val = train_full.iloc[-n_val:]
    
    pred_df = df.loc[df[config.SPLIT_COL] == config.PREDICTION_VALUE]
    is_anomaly = (pred_df["_status_label"] == 1).any()
    fault_onset = get_event_end(farm_id, dataset_id) if is_anomaly else None
    
    models = {
        "IsolationForest": IsolationForestDetector(),
        "Autoencoder": AutoencoderDetector(input_dim=len(selected_cols)),
        "LSTM-AE": LSTMAutoencoderDetector(input_dim=len(selected_cols), seq_len=6)
    }
    
    results = []
    
    for name, clf in models.items():
        log.info("Training %s...", name)
        t0 = time.time()
        
        if name == "IsolationForest":
            clf.fit(train_fit)
        else:
            clf.fit(train_fit, X_val=train_val, epochs=20, verbose=0)
            
        train_time = time.time() - t0
        
        val_scores = clf.score(train_val)
        from src.model import calibrate_threshold, scores_to_events
        threshold = calibrate_threshold(val_scores)
        
        pred_scores = clf.score(pred_df[selected_cols])
        pred_flags = scores_to_events(pred_scores, threshold)
        
        res = evaluation.evaluate_subdataset(
            name=f"dataset_{dataset_id}",
            is_anomaly=is_anomaly,
            status_ids=pred_df[config.STATUS_COL],
            predicted_flags=pred_flags,
            timestamps=pred_df[config.TIME_COL],
            fault_onset=fault_onset,
        )
        
        norm_earliness = min((res.earliness_hours or 0) / 48.0, 1.0)
        parts = [res.coverage, res.accuracy, res.reliability, norm_earliness]
        parts = [p for p in parts if p is not None and not np.isnan(p)]
        composite = float(np.mean(parts)) if parts else np.nan
        
        results.append({
            "Dataset": dataset_id,
            "Model": name,
            "Coverage": res.coverage,
            "Reliability": res.reliability,
            "Earliness": res.earliness_hours,
            "Accuracy": res.accuracy,
            "Composite": composite,
            "TrainTime": round(train_time, 1)
        })
        
    return results

def main():
    dev_subset = ["0", "3", "68", "72"]
    farm_id = "A"
    
    all_results = []
    for dataset_id in dev_subset:
        res = run_experiment_on_dataset(farm_id, dataset_id)
        all_results.extend(res)
        
    df_res = pd.DataFrame(all_results)
    print("\n=== Model Comparison ===")
    print(df_res.to_string(index=False))
    
    out_path = config.OUTPUT_DIR / "model_comparison.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(out_path, index=False)
    log.info("Saved comparison to %s", out_path)
    
if __name__ == "__main__":
    import os
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main()

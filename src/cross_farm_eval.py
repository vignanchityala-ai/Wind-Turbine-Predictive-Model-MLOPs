import argparse
import pandas as pd
from pathlib import Path
import logging

from src import config, data_loader, farm_pipeline, evaluation
from src.model import scores_to_events

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("cross_farm_eval")

def load_farm(farm_id, limit=None):
    f_conf = config.FARM_CONFIGS[farm_id]
    raw_dir = Path(f_conf["raw_dir"])
    paths = data_loader.discover_subdatasets(raw_dir)
    if limit:
        paths = paths[:limit]
    subs = []
    for p in paths:
        try:
            subs.append(data_loader.load_subdataset(p, None))
        except Exception as e:
            log.warning("Skipping %s: %s", p, e)
    return subs

def run_cross_eval(train_farms, eval_farms, model_kind="isolation_forest"):
    log.info(f"Training on Farms {train_farms}, Evaluating on Farms {eval_farms}")
    
    train_subs = []
    for f in train_farms:
        train_subs.extend(load_farm(f, limit=4))
        
    eval_subs = []
    eval_engineered = {}
    for f in eval_farms:
        f_subs = load_farm(f, limit=4)
        if not f_subs: continue
        eval_subs.extend(f_subs)
        
        log.info(f"Preparing evaluation data for Farm {f} (fitting baselines)...")
        try:
            _, f_engineered, _, _, _, _, _ = farm_pipeline.prepare_pooled_training_data(
                subs=f_subs,
                power_col=None, wind_col=None,
                feature_descriptions=None,
                do_feature_selection=False
            )
            eval_engineered.update(f_engineered)
        except Exception as e:
            log.warning(f"Failed to prepare evaluation data for Farm {f}: {e}")
            
    log.info("Training model...")
    model_result = farm_pipeline.train_farm_model(
        subs=train_subs,
        model_kind=model_kind,
        feature_descriptions=None,
        do_feature_selection=True
    )
    
    clf = model_result.detector
    threshold = model_result.threshold
    feature_cols = model_result.feature_cols
    
    results = []
    for sub in eval_subs:
        df_feat, _ = eval_engineered[sub.name]
        pred_df = df_feat.loc[df_feat[config.SPLIT_COL] == config.PREDICTION_VALUE]
        if len(pred_df) == 0:
            continue
            
        pred_X = pred_df.reindex(columns=feature_cols, fill_value=0.0)
        pred_scores = clf.score(pred_X)
        pred_flags = scores_to_events(pred_scores, threshold)
        
        result = evaluation.evaluate_subdataset(
            name=sub.name,
            is_anomaly=sub.is_anomaly,
            status_ids=pred_df[config.STATUS_COL],
            predicted_flags=pred_flags,
            timestamps=pred_df[config.TIME_COL],
            fault_onset=sub.event_end,
        )
        results.append(result)
        
    return results

def main():
    out_path = Path("outputs/cross_farm_comparison.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    # Experiment 1: Train A -> Eval B, C
    res_A = run_cross_eval(["A"], ["B", "C"])
    df_A = evaluation.aggregate_results(res_A)
    df_A["Train_Farms"] = "A"
    all_results.append(df_A)
    
    # Experiment 2: Train A, B -> Eval C
    res_AB = run_cross_eval(["A", "B"], ["C"])
    df_AB = evaluation.aggregate_results(res_AB)
    df_AB["Train_Farms"] = "A+B"
    all_results.append(df_AB)
    
    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(out_path, index=False)
    log.info(f"Saved cross-farm results to {out_path}")

if __name__ == "__main__":
    main()

"""Builds notebooks/01_eda_and_modeling.ipynb via nbformat."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

md("""# Wind Turbine Early Fault Detection — EDA & Modeling
### Wind Farm A (CARE benchmark, Kaggle mirror)

This notebook walks through:
1. Loading a Wind Farm A sub-dataset
2. Understanding the schema (status IDs, train/prediction split, event labels)
3. Exploratory data analysis (power curve, sensor drift, missingness)
4. Feature engineering
5. Training a per-turbine Normal Behavior Model (Isolation Forest baseline,
   then optionally an autoencoder)
6. Scoring the prediction period and visualizing the anomaly score vs the
   documented fault onset
7. Evaluating with CARE-style Coverage / Accuracy / Reliability / Earliness

**Before running:** download the dataset from
[Kaggle](https://www.kaggle.com/datasets/azizkasimov/wind-turbine-scada-data-for-early-fault-detection),
extract the `Wind Farm A` folder, and point `RAW_DIR` below at it. If you
just want to see the pipeline run, leave `USE_SYNTHETIC = True` to use the
bundled synthetic-but-schema-accurate sample data.
""")

code("""import sys
sys.path.append('..')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import config, data_loader, features, model as model_module, evaluation

pd.set_option('display.max_columns', 30)
plt.rcParams['figure.figsize'] = (11, 4)
""")

code("""# ---- Configuration ----
USE_SYNTHETIC = True   # set False once you've downloaded the real Kaggle data

if USE_SYNTHETIC:
    from tests.make_synthetic_data import build_synthetic_farm
    RAW_DIR = config.PROJECT_ROOT / "data" / "raw" / "Wind Farm A"
    if not RAW_DIR.exists():
        build_synthetic_farm(RAW_DIR)
else:
    RAW_DIR = config.RAW_DATA_DIR  # edit in src/config.py, or override here directly

paths = data_loader.discover_subdatasets(RAW_DIR)
print(f"Found {len(paths)} sub-datasets:")
for p in paths:
    print(" -", p.name)
""")

md("## 1. Load one anomaly sub-dataset and inspect the schema")

code("""anomaly_paths = [p for p in paths if 'anomaly' in p.stem.lower()]
normal_paths = [p for p in paths if 'anomaly' not in p.stem.lower()]

sub = data_loader.load_subdataset(anomaly_paths[0])
print("Sub-dataset:", sub.name)
print("Asset:", sub.asset_id, "| event_label:", sub.event_label)
print("event_start:", sub.event_start, "| event_end (fault onset):", sub.event_end)
print("Shape:", sub.df.shape)
sub.df.head()
""")

code("""print("Rows by split:")
print(sub.df[config.SPLIT_COL].value_counts())
print()
print("Rows by status_type_id (0/2 = normal, else abnormal):")
print(sub.df[config.STATUS_COL].value_counts().sort_index())
""")

md("""## 2. Power curve sanity check

Power should rise roughly as wind-speed cubed up to rated power, then flatten.
Points that fall well below the expected curve for their wind speed are a
classic early symptom of degradation (blade fouling, pitch misalignment,
gearbox friction, etc.).""")

code("""cols = features.identify_columns(sub.df)
power_col, wind_col = cols['power'][0], cols['wind_speed'][0]
print("Power column:", power_col, "| Wind speed column:", wind_col)

fig, ax = plt.subplots()
normal_train = sub.train[sub.train[config.STATUS_COL].isin(config.NORMAL_STATUS_IDS)]
ax.scatter(normal_train[wind_col], normal_train[power_col], s=2, alpha=0.3, label='train (normal)')
pred = sub.prediction
ax.scatter(pred[wind_col], pred[power_col], s=4, alpha=0.5, color='red', label='prediction period')
ax.set_xlabel('Wind speed'); ax.set_ylabel('Power'); ax.legend(); ax.set_title(f'{sub.name}: Power curve')
plt.show()
""")

md("## 3. Feature engineering")

code("""df_feat, feature_cols = features.engineer_features(sub.df)
print(f"{len(feature_cols)} model-ready numeric features engineered (rolling stats + power-curve residual).")
df_feat[feature_cols].describe().T.head(10)
""")

md("""## 4. Train a per-turbine Normal Behavior Model

We fit on the **training-period, normal-status-only** rows — this is
important: including abnormal-status training rows (derating, service mode)
would teach the model that degraded behavior is normal.""")

code("""label = features.build_label(df_feat)
df_feat = df_feat.assign(_status_label=label)

train_mask = df_feat[config.SPLIT_COL] == config.TRAIN_VALUE
normal_train_mask = train_mask & (df_feat['_status_label'] == 0)
train_full = df_feat.loc[normal_train_mask, feature_cols]

n_val = max(1, int(len(train_full) * config.VALIDATION_FRACTION))
train_fit, train_val = train_full.iloc[:-n_val], train_full.iloc[-n_val:]
print(f"Fitting on {len(train_fit)} rows, calibrating threshold on {len(train_val)} held-out normal rows.")

clf = model_module.IsolationForestDetector()
clf.fit(train_fit)

val_scores = clf.score(train_val)
threshold = model_module.calibrate_threshold(val_scores)
print(f"Calibrated anomaly threshold (99th pct of held-out normal scores): {threshold:.4f}")
""")

md("""> **Want the autoencoder instead?** Uncomment the block below (requires
> `pip install tensorflow --break-system-packages`). It's the approach used
> in the original CARE paper's own benchmark and tends to model nonlinear
> sensor interactions better than Isolation Forest, at the cost of needing
> more data and a GPU/CPU-minutes budget per turbine.
>
> ```python
> clf = model_module.AutoencoderDetector(input_dim=len(feature_cols))
> clf.fit(train_fit, X_val=train_val, epochs=40, verbose=1)
> val_scores = clf.score(train_val)
> threshold = model_module.calibrate_threshold(val_scores)
> ```
""")

md("## 5. Score the prediction period and visualize against the fault onset")

code("""pred_df = df_feat.loc[df_feat[config.SPLIT_COL] == config.PREDICTION_VALUE]
pred_scores = clf.score(pred_df[feature_cols])
pred_flags = model_module.scores_to_events(pred_scores, threshold)

fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(pred_df[config.TIME_COL], pred_scores, lw=0.8, label='anomaly score')
ax.axhline(threshold, color='orange', ls='--', label='threshold')
flagged_times = pred_df[config.TIME_COL][pred_flags == 1]
ax.scatter(flagged_times, pred_scores[pred_flags == 1], color='red', s=8, label='flagged (sustained)', zorder=5)
if sub.event_end is not None:
    ax.axvline(sub.event_end, color='black', ls=':', label='documented fault onset')
ax.set_title(f'{sub.name}: prediction-period anomaly score')
ax.legend(loc='upper left')
plt.show()

if len(flagged_times) > 0 and sub.event_end is not None:
    lead_hours = (sub.event_end - flagged_times.min()).total_seconds() / 3600
    print(f"First sustained alarm: {flagged_times.min()}  ->  {lead_hours:.1f} hours before documented fault onset")
""")

md("## 6. Evaluate with CARE-style metrics for this one sub-dataset")

code("""result = evaluation.evaluate_subdataset(
    name=sub.name,
    is_anomaly=sub.is_anomaly,
    status_ids=pred_df[config.STATUS_COL],
    predicted_flags=pred_flags,
    timestamps=pred_df[config.TIME_COL],
    fault_onset=sub.event_end,
)
result
""")

md("""## 7. Run across ALL sub-datasets

This mirrors `src/run_pipeline.py` — same logic, but interactive so you can
inspect intermediate results. For a full production run, use the script:

```bash
python -m src.run_pipeline --model isolation_forest
```
""")

code("""from src.run_pipeline import process_one

all_results, all_diag = [], []
for p in paths:
    s = data_loader.load_subdataset(p)
    try:
        res, diag = process_one(s, model_kind='isolation_forest')
        all_results.append(res)
        diag['dataset'] = s.name
        all_diag.append(diag)
    except Exception as e:
        print(f"FAILED on {p.name}: {e}")

report_df = evaluation.aggregate_results(all_results)
report_df
""")

code("""summary = evaluation.summarize(report_df)
for k, v in summary.items():
    print(f"{k:28s} {v}")
""")

md("""## Notes for productionizing

- **Per-turbine models**: this notebook and pipeline fit one model per
  turbine/sub-dataset, matching how the CARE dataset (and most real SCADA
  deployments) are structured. If your fleet has many turbines of the same
  type, consider a shared model with turbine ID as a feature instead —
  fewer models to maintain, but you lose some turbine-specific nuance.
- **Retraining cadence**: seasonal effects matter (this data spans full
  years for a reason). Retrain training-window models at least seasonally,
  or use a rolling window.
- **Alert routing**: the `min_event_length` in `config.py` controls how many
  consecutive flagged points are required before an alert fires — tune this
  against your team's tolerance for false alarms vs. detection lag.
- **Next model upgrades**: LSTM/Transformer autoencoders, or a proper
  survival/RUL (remaining-useful-life) model, tend to outperform the dense
  autoencoder baseline here on this exact dataset in published work —
  worth trying once the pipeline skeleton above is proven out.
""")

nb['cells'] = cells
nb['metadata'] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

import os
os.makedirs('/home/claude/wind_turbine_pipeline/notebooks', exist_ok=True)
with open('/home/claude/wind_turbine_pipeline/notebooks/01_eda_and_modeling.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook written.")

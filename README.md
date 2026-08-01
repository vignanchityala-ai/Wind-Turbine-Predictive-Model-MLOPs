# Wind Turbine Early Fault Detection — Pipeline

Predictive-maintenance pipeline for the **CARE benchmark** dataset
("Wind Turbine SCADA Data For Early Fault Detection", Kaggle mirror of
Gück, Roelofs & Faulstich 2024). Targets **Wind Farm A** (5 turbines,
22 sub-datasets, 86 features) as the initial scope — smallest and fastest
to iterate on; the same code generalizes to Farms B/C by pointing
`--raw-dir` at those folders (expect to need feature-selection before
rolling all 257/957 columns, see Notes below).

## What this is

An **early anomaly-detection** pipeline, not a plain classifier: the model
learns what "normal" looks like for each turbine from ~1 year of healthy
operating data, then flags sustained deviations during the prediction
window — with the goal of catching degradation *before* the turbine
actually reports a fault (status_type_id 4).

## 1. Get the data

Kaggle requires auth, so this couldn't be downloaded automatically. You'll
need to:

1. Download from [Kaggle](https://www.kaggle.com/datasets/azizkasimov/wind-turbine-scada-data-for-early-fault-detection)
   (or the original [Zenodo release](https://zenodo.org/records/10958775) —
   check for the latest version, some labels were corrected between releases).
2. Extract the **Wind Farm A** folder.
3. Point the pipeline at it, either by editing `RAW_DATA_DIR` in
   `src/config.py`, or passing `--raw-dir "/path/to/Wind Farm A"` on the
   command line / setting `RAW_DIR` in the notebook.

The loader auto-detects per-turbine CSVs and any accompanying
`event_info.csv`/`.json` metadata, and tolerates minor schema drift (column
name variants, comma vs semicolon separators) across dataset releases.

## 2. Install

```bash
pip install -r requirements.txt
# Only if you want the autoencoder model:
pip install tensorflow
```

## 3. Run

**Notebook** (exploration, one turbine at a time, visualizations):
```bash
jupyter notebook notebooks/01_eda_and_modeling.ipynb
```
Set `USE_SYNTHETIC = False` and point `RAW_DIR` at your real data once
you've downloaded it. It ships pre-executed against bundled synthetic
schema-accurate sample data so you can see what the output looks like
without downloading anything first.

**Full pipeline** (all sub-datasets, batch, produces the evaluation report):
```bash
python -m src.run_pipeline --model isolation_forest --raw-dir "data/raw/Wind Farm A"
# or, if you've installed tensorflow:
python -m src.run_pipeline --model autoencoder --raw-dir "data/raw/Wind Farm A"
```

Outputs land in `outputs/`:
- `evaluation_report.csv` — per-sub-dataset Coverage/Accuracy/Reliability/Earliness
- `evaluation_report_diagnostics.csv` — feature counts, row counts, thresholds, runtime

## 4. Serve it as an API

Add `--save-models` to persist a joblib bundle per turbine (detector +
threshold + feature order + power-curve reference), then run the FastAPI
service that loads them:

```bash
python -m src.run_pipeline --model isolation_forest --save-models
uvicorn src.serve:app --port 8000
# add --reload while developing
```

Interactive API docs (Swagger UI, generated for free by FastAPI) at
`http://localhost:8000/docs`.

```bash
curl -X POST http://localhost:8000/predict/WT100_anomaly \
  -H "Content-Type: application/json" \
  -d '{"readings": [{"time_stamp": "2021-02-08T23:40:00", "power": 812.3, "wind_speed": 9.1, ...}, ...]}'
```

Send at least 36 readings (6h at 10-min resolution, ascending time order) —
the model's rolling-window features need that lookback to be meaningful.
See the docstring at the top of `src/serve.py` for the full API contract,
including two things worth reading before you wire anything real up to
this: the API is deliberately **stateless** (callers supply the window,
not the server), and it does single-point scoring, not the sustained-event
smoothing the offline evaluation uses — both are explained there with the
reasoning, not just asserted.

**Auth**: set the `API_KEY` env var in any environment other than your own
laptop — `uvicorn` with no `API_KEY` set runs with auth disabled.

**Test it**: `pip install -r requirements.txt`, then
`python tests/smoke_test_api.py` against a running server — same check CI
runs on every push.

**Docker**:
```bash
docker build -t wind-turbine-api .
docker run -p 8000:8000 -e API_KEY=your-secret-here wind-turbine-api
```
*(Written carefully but not build-tested — no Docker daemon in the sandbox
this was built in. Validate locally before deploying.)*

**Where to actually run it**, since this is for internal/work use: check
first whether your team already has an AWS/Azure/GCP account this should
live in — that answer determines both the specific hosting service
(ECS Fargate / Cloud Run / Container Apps all fit a single small container
well) and, more importantly, whatever auth and network-access pattern your
company already standardizes on. If there's no existing cloud footprint
yet, a single small VM running this container behind a reverse proxy
(Caddy or nginx for TLS) is a reasonable minimal-ops starting point for a
handful of internal callers — no Kubernetes needed for this scale. Either
way, loop in whoever handles infra/security before this is reachable
outside your own machine, especially once it's scoring real turbine data
rather than the public Kaggle set.

## How it works

```
src/config.py          Schema constants, status-ID map, tunable thresholds
src/data_loader.py      Robust CSV loading + event metadata parsing
src/features.py         Zero-run cleaning, power-curve residual (fit/apply split
                         to avoid train/serve skew), rolling stats
src/model.py             IsolationForestDetector (baseline) + AutoencoderDetector
src/evaluation.py       CARE-style Coverage/Accuracy/Reliability/Earliness scoring
src/run_pipeline.py      Orchestrates load -> features -> fit -> score -> evaluate
                         -> (optionally) persist model bundles
src/serve.py             FastAPI app: loads persisted bundles, exposes /predict
Dockerfile              Packages src/serve.py + models/ into a container
tests/make_synthetic_data.py   Generates schema-accurate fake data for smoke-testing
tests/smoke_test_api.py        End-to-end check: train -> save -> serve -> predict
tests/build_notebook.py        Rebuilds the notebook from source (edit this, not the .ipynb, to change it)
.github/workflows/ci.yml       Runs the above smoke tests on every push
```

**Per-turbine models, not one global model.** Each sub-dataset gets its own
Normal Behavior Model, trained only on that turbine's normal-status training
rows. This matches both the dataset's design and standard SCADA
predictive-maintenance practice — different turbines/sensors have different
baselines, and training on abnormal-status rows would teach the model that
degraded behavior is fine.

**Why not plain accuracy/F1?** A model evaluated only on point-wise
accuracy can get 99%+ by never flagging anything, since anomalies are rare
even within anomalous sub-datasets. We use four complementary metrics
(mirroring the dataset's own CARE-score philosophy):

| Metric | Answers |
|---|---|
| Coverage | Did we detect the pre-fault degradation window? |
| Accuracy | Do we avoid false alarms on entirely healthy turbines? |
| Reliability | Did we raise one confident alarm, or a flood of flickering ones? |
| Earliness | How many hours before the actual fault did we first sound alarm? |

## Notes / next steps

- **Wind Farms B & C**: same code works, but with 257/957 raw sensor
  columns, rolling every column (as this pipeline does for Farm A) will be
  slow and likely overfit small per-turbine training sets. Add a
  feature-selection step (variance threshold, or domain-relevant subsets)
  before scaling up.
- **Known data-quality caveats** (from the dataset's own documentation):
  Wind Farms B/C use 0 to encode missing values in some sensors, and status
  logging can lag behind real state changes. `clean_zeros_as_missing()` in
  `features.py` handles long zero-runs defensively, but validate against
  each farm's specific sensor list before trusting Min/Max/Std columns.
- **Model upgrade path**: the autoencoder here is a simple dense
  architecture. Published work on this exact dataset gets better results
  with LSTM/Transformer autoencoders or ensemble scoring — worth trying
  once this skeleton is validated against real data and your manager has
  seen a first pass.
- **Retraining cadence**: seasonality matters (that's why each sub-dataset
  spans a full year). Plan to retrain at least seasonally in production.

# Wind Turbine MLOps — Agent Handoff Instructions (Phases 3–10)

> **PURPOSE**: This file contains everything an AI agent needs to implement the
> remaining phases of this project. Read this file + `PROJECT_CONTEXT.md` before
> doing ANY work. Do NOT skip sections.

---

## CRITICAL CONTEXT — READ FIRST

### Project Location
```
d:\OneDrive - SoftDEL Systems Pvt. Ltd\Data_of_C\wind_turbine_pipeline\wind_turbine_pipeline
```

### Hardware Constraints
- **RAM**: 16 GB total, ~6-7 GB available
- **Farm A**: 1.47 GB raw CSV → 262 MB Bronze Parquet (DONE)
- **Farm B**: 2.53 GB raw CSV → NOT YET INGESTED (user must copy to data/raw/Wind Farm B/)
- **Farm C**: 34 GB raw CSV → NOT YET INGESTED (user must copy to data/raw/Wind Farm C/)
- **RULE**: NEVER use `pd.read_csv()` on full Farm B/C files. Always use DuckDB + Parquet.

### What Is Already Complete
- **Phase 0**: config.py has `FARM_CONFIGS`, `BRONZE_DIR`, `SILVER_DIR`, `GOLD_DIR`. YAML configs exist.
- **Phase 1**: `src/data/ingestion.py` (streaming CSV→Parquet), `schema.py`, `validation.py`. Farm A Bronze done.
- **Phase 2**: `src/data/silver.py`, `src/data/feature_pipeline.py`. Farm A Silver (22 datasets) and Gold (4 dev subset) done. `features.py` has temporal features. `feature_selection.py` has `pre_select_sensors()`.
- **Original pipeline**: `run_pipeline.py`, `farm_pipeline.py`, `model.py`, `evaluation.py`, `serve.py` ALL STILL WORK. Do NOT break them.

### Key Files To Read Before Each Phase
| File | Why |
|---|---|
| `PROJECT_CONTEXT.md` | Full project state, conversation history, design decisions |
| `src/config.py` | All paths, farm configs, schema constants, model params |
| `src/model.py` | Existing IsolationForest + Autoencoder implementations |
| `src/evaluation.py` | CARE-score evaluation (Coverage, Reliability, Earliness) |
| `src/run_pipeline.py` | Existing pipeline orchestration — your Phase 3 code must be compatible |
| `src/serve.py` | Existing FastAPI endpoints — Phase 6 extends this |
| `src/features.py` | Feature engineering — has `add_temporal_features()` and `add_rolling_features()` |
| `src/data/feature_pipeline.py` | Gold layer pipeline — use `get_dev_subset("A")` for quick iteration |

### Existing Model Interface (in src/model.py)
Every model MUST implement this interface:
```python
class SomeDetector:
    def fit(self, X_train, X_val=None, **kwargs): ...
    def score(self, X) -> np.ndarray:  # returns anomaly scores (higher = more anomalous)
        ...

# After scoring:
threshold = calibrate_threshold(val_scores)  # uses THRESHOLD_PERCENTILE from config
flags = scores_to_events(scores, threshold)  # collapses to sustained events using MIN_EVENT_LENGTH
```

### Existing Evaluation Interface (in src/evaluation.py)
```python
result = evaluate_subdataset(
    name="dataset_0",
    is_anomaly=True,
    status_ids=pred_df["status_type_id"],
    predicted_flags=pred_flags,
    timestamps=pred_df["time_stamp"],
    fault_onset=event_end,  # NOTE: event_end is the fault onset, not event_start
)
# result has: coverage, reliability, earliness, composite
```

---

## PHASE 3 — Baseline + Model Comparison

### Goal
Compare Isolation Forest vs Dense Autoencoder vs LSTM Autoencoder on the SAME Gold data with the SAME evaluation. Pick the winner.

### Prerequisites
- Gold Parquet exists for dev subset: `data/gold/farm=A/dataset_0.parquet`, etc.
- If not, run: `from src.data.feature_pipeline import process_farm_to_gold; process_farm_to_gold("A", dataset_ids=["0","3","68","72"])`

### Step 3.1 — Create experiment runner
**File**: `src/experiment.py` (NEW)

```python
"""
experiment.py — Fair model comparison on Gold data.
Trains each model type on the same data, evaluates with same CARE metrics.
"""
```

This file should:
1. Load Gold Parquet for a dataset via DuckDB (use `src/data/feature_pipeline` pattern)
2. Split into train_normal / val / prediction (use existing temporal tail-split logic from `run_pipeline.py` lines 53-82)
3. Apply feature selection (use existing `feature_selection.select_features()`)
4. Train each model type
5. Evaluate each with `evaluation.evaluate_subdataset()`
6. Return comparison table

### Step 3.2 — Fix Autoencoder persistence
**File**: `src/model.py` (MODIFY)

The existing `AutoencoderDetector` uses Keras but can't be saved via joblib properly. Fix:
```python
def save(self, path):
    # Save Keras model separately, save preprocessor via joblib
    
def load(cls, path):
    # Load Keras model + preprocessor
```

### Step 3.3 — Add LSTM Autoencoder
**File**: `src/model.py` (MODIFY — add new class)

```python
class LSTMAutoencoderDetector:
    """LSTM-based autoencoder for temporal anomaly detection.
    
    Reshapes input into sequences of length `seq_len` (default 6 = 1 hour 
    at 10-min resolution). Encoder-decoder architecture with LSTM layers.
    Anomaly score = reconstruction error (MSE per sample).
    
    IMPORTANT: Must implement same interface as IsolationForestDetector:
    - fit(X_train, X_val=None, epochs=40, verbose=0)
    - score(X) -> np.ndarray
    """
    def __init__(self, input_dim, seq_len=6, latent_dim=32):
        ...
```

**CRITICAL**: The `score()` method must return a 1D array of shape `(n_samples,)` — same as IF and AE. The existing `calibrate_threshold()` and `scores_to_events()` expect this shape.

**MEMORY WARNING**: For LSTM, you need to reshape data into sequences. With 613 features and seq_len=6, each sequence is 6×613 = 3,678 values. With 50K training rows, that's ~700 MB. This is tight but feasible. If memory errors occur, reduce features via `select_features()` first (it typically reduces from 613 to ~200).

### Step 3.4 — Run comparison on dev subset
Create a script that runs all 3 models on datasets [0, 3, 68, 72] and produces a comparison table:

```
| Dataset | Model          | Coverage | Reliability | Earliness | Composite |
|---------|----------------|----------|-------------|-----------|-----------|
| 0       | IsolationForest| ...      | ...         | ...       | ...       |
| 0       | Autoencoder    | ...      | ...         | ...       | ...       |
| 0       | LSTM-AE        | ...      | ...         | ...       | ...       |
```

### Step 3.5 — Select winner
- Best model = highest average CARE composite across anomaly datasets
- Save comparison results to `outputs/model_comparison.csv`
- Document the winner in `PROJECT_CONTEXT.md`

### Verification
- All 3 models train without errors on dev subset
- CARE scores are computed for all models on same datasets
- Comparison table is saved

### Common Pitfalls
- Do NOT use random train/test split. The split is TEMPORAL (tail-split). Use `train_test` column.
- `fault_onset` is `event_end` in the event_info, NOT `event_start`. This is a CARE benchmark convention.
- Feature selection must run ONLY on training-normal data (status 0 or 2, train_test='train').
- LSTM input must be float32, not float64 (saves 50% memory).

---

## PHASE 4 — Multi-Farm Generalization

### Goal
Extend pipeline to Farms B and C. Cross-farm evaluation.

### Prerequisites
- User must copy Farm B/C data to `data/raw/Wind Farm B/` and `data/raw/Wind Farm C/`
- Phase 3 winner model determined

### Step 4.1 — Ingest Farm B and C
```python
from src.data.ingestion import ingest_farm
report_b = ingest_farm("B")
report_c = ingest_farm("C")  # This will take a while (34 GB)
```

For Farm C, ingestion uses PyArrow streaming (50K-row batches). Peak memory ~80 MB. Safe.

### Step 4.2 — Silver + Gold for B and C
```python
from src.data.silver import process_farm_to_silver
from src.data.feature_pipeline import process_farm_to_gold

process_farm_to_silver("B")
process_farm_to_silver("C")

# For Farm C, use dev subset first (pre-selection will reduce 957 to 100 sensors)
process_farm_to_gold("B", dataset_ids=["first_4_ids"])
process_farm_to_gold("C", dataset_ids=["first_4_ids"])
```

**CRITICAL for Farm C**: `feature_pipeline.py` already has `pre_select_sensors()` wired in. It triggers automatically when `n_features > 200`. It will reduce 957 sensors to top 100 by variance before rolling stats. This prevents the 5,742-column explosion.

### Step 4.3 — Cross-farm evaluation
Train on Farm A → evaluate on B, C. Then train on A+B → evaluate on C.

### Step 4.4 — Add `global` training strategy
**File**: `src/run_pipeline.py` (MODIFY)

Add `--training-strategy global` option that pools ALL farms:
```python
parser.add_argument("--training-strategy", choices=["turbine", "farm", "global"])
```

### Verification
- Pipeline runs on all 3 farms without memory errors
- Cross-farm CARE scores documented
- Farm C runs with pre-selected sensors (check logs for "pre_select_sensors: reducing 957")

---

## PHASE 5 — MLflow + Model Registry

### Goal
Track experiments, register models with promotion workflow.

### Prerequisites
- `pip install mlflow`

### Step 5.1 — MLflow tracker wrapper
**File**: `src/tracking/mlflow_tracker.py` (NEW)

```python
"""Wraps training runs in MLflow context."""
import mlflow

class ExperimentTracker:
    def __init__(self, experiment_name, tracking_uri="http://localhost:5000"):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
    
    def log_training_run(self, params, metrics, artifacts, model_name):
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)  # coverage, reliability, earliness, composite
            for name, path in artifacts.items():
                mlflow.log_artifact(path)
            # Register model
            mlflow.register_model(...)
```

### Step 5.2 — Quality gates
```python
def check_quality_gate(new_metrics, production_metrics):
    """New model must beat production on composite score."""
    return new_metrics["composite"] >= production_metrics["composite"]
```

### Step 5.3 — Add model bundle metadata
When saving model bundles (in `run_pipeline.py` line 123-139), add:
```python
bundle["model_version"] = "1.0.0"
bundle["training_date"] = datetime.now().isoformat()
bundle["training_farms"] = ["A"]
bundle["care_composite"] = result.composite
bundle["feature_schema_hash"] = hashlib.sha256(str(sorted(feature_cols)).encode()).hexdigest()
```

### Verification
- `mlflow ui` shows experiments with metrics
- Model registry has entries
- Quality gate blocks inferior models

---

## PHASE 6 — FastAPI Enhancement

### Goal
Extend the existing `src/serve.py` with batch predict, farm serving, bounded cache.

### Prerequisites
- Phase 3 winner model trained and saved

### What Already Exists (DO NOT REWRITE)
`src/serve.py` already has:
- `POST /predict/{dataset_name}` — single dataset prediction
- `GET /models` — list available models
- `GET /health` — health check
- API key auth via `X-API-Key` header
- `_MODEL_CACHE` (unbounded dict)
- `PredictRequest` / `PredictResponse` schemas

### Step 6.1 — Bounded model cache
Replace `_MODEL_CACHE: dict[str, dict] = {}` with:
```python
from functools import lru_cache
# Or use cachetools.LRUCache(maxsize=10)
```

### Step 6.2 — Add new endpoints
Add to `serve.py`:
```python
@app.post("/batch_predict")  # CSV upload, returns per-row scores + aggregated events
@app.post("/predict/farm/{farm_name}")  # farm-level model serving
@app.get("/model/info/{model_name}")  # metadata (training date, features, metrics)
@app.get("/farms")  # list farms and turbine counts
```

### Step 6.3 — Event-based response format
For batch predictions, return events (not just raw row scores):
```python
class BatchPredictResponse(BaseModel):
    events: list[dict]  # [{event_id, start, end, duration_hours, peak_score, mean_score}]
    per_row_scores: list[float]
    threshold: float
    n_readings: int
```

### Step 6.4 — Data freshness header
```python
# In predict endpoint, check how old the latest timestamp is
data_age = (datetime.now() - latest_timestamp).total_seconds() / 60
if data_age > 60:
    response.warning = f"Data is {data_age:.0f} minutes old"
```

### Step 6.5 — Install Prometheus metrics
```
pip install prometheus-fastapi-instrumentator
```
```python
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)
```

### Verification
- `python -m src.serve` starts without errors
- All new endpoints return correct responses
- `tests/smoke_test_api.py` passes
- Prometheus metrics available at `/metrics`

---

## PHASE 7 — Dashboard (Streamlit)

### Goal
Interactive dashboard that calls FastAPI (NEVER imports model code directly).

### Prerequisites
- `pip install streamlit requests plotly`
- Phase 6 FastAPI running on localhost:8000

### Architecture Rule
```
Streamlit (dashboard/) → HTTP requests → FastAPI (src/serve.py) → Model
```
The dashboard directory must NOT import anything from `src/`. It calls the API via `requests.post()`.

### Step 7.1 — API client
**File**: `dashboard/api_client.py` (NEW)
```python
import requests

class WindTurbineAPIClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def predict(self, dataset_name, readings):
        return requests.post(f"{self.base_url}/predict/{dataset_name}", json={"readings": readings})
    
    def batch_predict(self, farm_name, csv_file):
        return requests.post(f"{self.base_url}/batch_predict", files={"file": csv_file})
    
    def get_farms(self):
        return requests.get(f"{self.base_url}/farms").json()
    
    def get_model_info(self, name):
        return requests.get(f"{self.base_url}/model/info/{name}").json()
```

### Step 7.2 — Multi-page Streamlit app
**File**: `dashboard/app.py` (NEW)

Pages:
1. **Upload & Predict**: CSV upload → API call → show decision (🔴/🟢) with raw score + threshold (NOT fake confidence %)
2. **Farm Overview**: Grid of turbines with status indicators
3. **Turbine Deep-Dive**: Power curve, sensor time series, anomaly score chart
4. **Model Performance**: CARE score breakdown, model comparison table
5. **Monitoring**: API latency, feature drift, data freshness

### Display Rules
- Show: `Anomaly Score: 0.87 | Threshold: 0.65 | Decision: 🔴 ANOMALY`
- Do NOT show: `Confidence: 87%` (this is meaningless for anomaly scores)
- For Autoencoder: show per-feature reconstruction error as bar chart
- For Isolation Forest: omit per-feature attribution (not trustworthy)

### Verification
- `streamlit run dashboard/app.py` starts
- All pages render
- Upload CSV → see results (requires FastAPI running)
- `grep -r "from src" dashboard/` returns NOTHING (no direct imports)

---

## PHASE 8 — DVC + Docker + CI/CD

### Step 8.1 — DVC
```bash
pip install dvc
dvc init
dvc add data/bronze data/silver data/gold
# Creates .dvc files (metadata only, tracked in Git)
# Actual data stored in DVC remote
```

**File**: `dvc.yaml` (NEW)
```yaml
stages:
  ingest:
    cmd: python -c "from src.data.ingestion import ingest_farm; ingest_farm('A')"
    deps: [data/raw/Wind Farm A]
    outs: [data/bronze/farm=A]
  
  silver:
    cmd: python -c "from src.data.silver import process_farm_to_silver; process_farm_to_silver('A')"
    deps: [data/bronze/farm=A]
    outs: [data/silver/farm=A]
  
  gold:
    cmd: python -c "from src.data.feature_pipeline import process_farm_to_gold; process_farm_to_gold('A')"
    deps: [data/silver/farm=A]
    outs: [data/gold/farm=A]
  
  train:
    cmd: python -m src.run_pipeline --training-strategy farm
    deps: [data/gold/farm=A, src/model.py, src/features.py]
    outs: [models/]
    params: [params.yaml]
```

**File**: `params.yaml` (NEW)
```yaml
model_kind: isolation_forest
training_strategy: farm
contamination: 0.01
threshold_percentile: 99.0
rolling_windows: [6, 18, 36]
validation_fraction: 0.2
min_event_length: 6
```

### Step 8.2 — Docker
**File**: `docker/Dockerfile.api` (NEW)
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY models/ models/
COPY configs/ configs/
EXPOSE 8000
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
```

**File**: `docker/Dockerfile.dashboard` (NEW)
```dockerfile
FROM python:3.10-slim
WORKDIR /app
RUN pip install streamlit requests plotly
COPY dashboard/ dashboard/
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501"]
```

**File**: `docker-compose.yml` (NEW)
```yaml
services:
  api:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    ports: ["8000:8000"]
    environment:
      - API_KEY=${API_KEY}
    volumes:
      - ./models:/app/models
  
  dashboard:
    build:
      context: .
      dockerfile: docker/Dockerfile.dashboard
    ports: ["8501:8501"]
    depends_on: [api]
    environment:
      - API_URL=http://api:8000
  
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports: ["5000:5000"]
    command: mlflow server --host 0.0.0.0
```

### Step 8.3 — CI/CD
**File**: `.github/workflows/ci.yml` (MODIFY existing)
```yaml
jobs:
  test:
    steps:
      - run: pip install -r requirements.txt
      - run: python -m pytest tests/ -v
      - run: python tests/smoke_test_api.py
      - run: docker build -f docker/Dockerfile.api -t wt-api .
```

### Verification
- `dvc repro` runs full pipeline
- `docker-compose up` starts all services
- CI passes on push

---

## PHASE 9 — Monitoring + Retraining

### Step 9.1 — ML monitoring
**File**: `src/monitoring/drift.py` (NEW)
```python
"""Feature drift detection using PSI (Population Stability Index)."""

def compute_psi(reference, current, bins=10):
    """PSI > 0.2 = significant drift."""
    ...

def check_drift(reference_features, current_features, threshold=0.2):
    """Check all features for drift. Returns list of drifted feature names."""
    ...
```

### Step 9.2 — Data freshness check
**File**: `src/monitoring/freshness.py` (NEW)
```python
def check_data_freshness(latest_timestamp, max_age_minutes=60):
    """Returns True if data is fresh enough for reliable predictions."""
    ...
```

### Step 9.3 — Automated retraining trigger
```python
def should_retrain(drift_results, model_age_days, max_age=30):
    """Trigger retraining if drift detected or model is too old."""
    return any(drift_results.values()) or model_age_days > max_age
```

### Verification
- Drift detection runs on test data
- Freshness check works
- Retraining trigger fires correctly

---

## PHASE 10 — Cloud Deployment (Azure)

### Step 10.1 — Azure Blob for DVC remote
```bash
dvc remote add azure azure://wind-turbine-data
dvc remote modify azure account_name <storage_account>
dvc push
```

### Step 10.2 — Azure Container Registry
```bash
az acr create --name windturbineacr --sku Basic
docker tag wt-api windturbineacr.azurecr.io/wt-api:latest
docker push windturbineacr.azurecr.io/wt-api:latest
```

### Step 10.3 — Azure Container Apps
Deploy API and Dashboard as Container Apps.

### Verification
- API accessible via Azure URL
- Dashboard connects to API
- DVC can push/pull from Azure Blob

---

## GENERAL RULES FOR ALL PHASES

1. **After EVERY phase**: Update `PROJECT_CONTEXT.md` with what was done
2. **After EVERY phase**: Run the existing tests to verify nothing broke: `python tests/smoke_test_api.py`
3. **NEVER** load full CSVs with `pd.read_csv()` for Farm B/C. Use DuckDB.
4. **NEVER** put ML code in the dashboard. Dashboard calls API only.
5. **ALWAYS** use temporal splits (tail-split), never random splits.
6. **ALWAYS** fit feature selection on training-normal data only.
7. **ALWAYS** persist power_curve_reference and feature_cols in model bundles.
8. **Test each phase** before moving to the next. Don't batch phases.
9. **Memory budget**: keep peak usage under 5 GB (leave headroom for OS/org tasks).
10. **Backward compatibility**: the original `python -m src.run_pipeline` must still work.

---

## DEPENDENCIES TO INSTALL (across all remaining phases)

```bash
pip install mlflow streamlit requests plotly prometheus-fastapi-instrumentator dvc evidently cachetools
```

Install these ONLY when the phase that needs them begins, not all at once.

---

## QUICK REFERENCE — FILE LOCATIONS

```
src/config.py               → FARM_CONFIGS, paths, schema constants
src/model.py                → IsolationForestDetector, AutoencoderDetector (add LSTM here)
src/evaluation.py           → evaluate_subdataset(), CARE scores
src/features.py             → engineer_features(), add_temporal_features()
src/feature_selection.py    → select_features(), pre_select_sensors()
src/run_pipeline.py         → process_one(), main pipeline orchestrator
src/farm_pipeline.py        → farm-level pooled training with z-score normalization
src/serve.py                → FastAPI app, predict endpoint, model cache
src/data/ingestion.py       → ingest_farm(), query_bronze()
src/data/silver.py          → process_farm_to_silver(), query_silver()
src/data/feature_pipeline.py → process_farm_to_gold(), get_dev_subset()
src/data/validation.py      → validate_bronze_dataset()
src/data/schema.py          → InternalEvent, FarmMetadata, parse_event_info()
configs/development.yaml    → Dev environment settings
data/bronze/farm=A/         → 22 Parquet files (raw data, columnar)
data/silver/farm=A/         → 22 cleaned Parquet files
data/gold/farm=A/           → 4 feature-engineered Parquet files (dev subset)
PROJECT_CONTEXT.md          → Living documentation (UPDATE THIS AFTER EVERY PHASE)
```

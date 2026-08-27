# Wind Turbine MLOps — Revised Implementation Plan v2
## Incorporating ChatGPT Review Feedback

> [!NOTE]
> This replaces the original implementation plan. Changes from v1 are marked with 🆕.

---

## Revised Phase Order (Dependency-Driven, Not Calendar-Driven)

```
Phase 0  → Architecture Foundation (project structure, configs, logging)
Phase 1  → Scalable Data Pipeline (CSV → Parquet → DuckDB → Bronze/Silver)
Phase 2  → Feature Pipeline (Silver → Gold, leakage-safe, select-before-engineer for C)
Phase 3  → Baseline + Model Comparison (IF vs AE vs LSTM-AE, proper temporal eval)
Phase 4  → Multi-Farm Generalization (B, C ingestion + cross-farm evaluation)
Phase 5  → MLflow + Model Registry + Quality Gates
Phase 6  → FastAPI Enhancement (batch predict, farm serving, bounded cache)
Phase 7  → Dashboard (Streamlit as API client only)
Phase 8  → DVC + Docker + CI/CD
Phase 9  → Monitoring + Retraining
Phase 10 → Cloud Deployment (Azure)
```

🆕 **Key ordering changes from v1**:
- Data pipeline (Phases 0-2) is fully complete before ANY model work
- Models are compared experimentally (Phase 3), not all built simultaneously
- FastAPI comes BEFORE dashboard (Phase 6 before Phase 7)
- Dashboard calls API only — never imports model code directly

---

## Phase 0 — Architecture Foundation

> **Goal**: Establish project structure, configuration system, and logging before any code changes.

### [MODIFY] `src/config.py`
- Add `FARM_CONFIGS` dict for A/B/C with raw_dir, feature_count, turbine_count
- Add `BRONZE_DIR`, `SILVER_DIR`, `GOLD_DIR` paths
- Add DuckDB configuration
- Keep backward-compatible (existing Farm A flows still work unchanged)

### [NEW] `configs/development.yaml` / `configs/staging.yaml` / `configs/production.yaml`
- Farm selection, data paths, model params, MLflow URI, log level
- Loaded via a simple `load_config(env)` function

### [NEW] `src/utils/logging_config.py`
- Structured logging with JSON output option
- Request ID correlation for API calls
- Separate loggers for data pipeline, training, serving

### Verification
- Existing `python -m src.run_pipeline` still works unchanged
- New YAML configs load without error

---

## Phase 1 — Scalable Data Pipeline

> **Goal**: Solve the 39 GB memory problem. Never load full CSVs into pandas.

### [NEW] `src/data/ingestion.py`

🆕 **Explicit DuckDB + PyArrow architecture**:
```
Raw CSV → PyArrow streaming reader → Parquet (Bronze)
                                         ↓
                                    DuckDB queries
                                         ↓
                                 Only needed columns/rows → Pandas
```

**Key functions**:
- `ingest_farm_to_bronze(farm_name)` — streaming CSV → partitioned Parquet
- `query_bronze(farm, dataset_id, columns, time_range)` — DuckDB → filtered DataFrame
- Partitioning: `data/bronze/farm=A/dataset_id=3/data.parquet`

### [NEW] `src/data/validation.py`

Schema + quality validation using Pandera:
- Required columns present and correctly typed
- Timestamp continuity (no unexplained gaps)
- Range checks (power ≥ 0, wind_speed ≥ 0, status ∈ {0..5})
- Missingness thresholds per column
- Duplicate detection
- Event metadata completeness
- Generates PASS/FAIL validation report

### [NEW] `src/data/schema.py`

Canonical internal schema (adapter pattern):
```python
@dataclass
class InternalEvent:
    farm_id: str
    asset_id: str
    dataset_id: str
    event_id: str
    event_start: pd.Timestamp
    event_end: pd.Timestamp
    event_type: str  # 'anomaly' | 'normal'
```

### Bronze → Silver transformation
- Apply validation
- Standardize column names to internal schema
- Handle Farm B/C 0-as-missing conversion
- Merge event metadata into dataset
- Output: `data/silver/farm=A/dataset_id=3/data.parquet`

### Verification
- Ingest all 22 Farm A CSVs → Parquet
- Verify Parquet files are queryable via DuckDB without loading full dataset
- Validation report shows all PASS for Farm A
- Peak memory during ingestion stays under 2 GB

---

## Phase 2 — Feature Pipeline

> **Goal**: Silver → Gold ML-ready features. Leakage-safe. Scalable to Farm C.

### [MODIFY] `src/features.py`

🆕 **Add temporal features** (rate-of-change, lag features, cyclical time encoding):
```python
def add_temporal_features(df):
    # First-order differences for key sensors
    # Lag features: t-1, t-6, t-36
    # Cyclical hour-of-day, day-of-week encoding
```

### [MODIFY] `src/feature_selection.py`

🆕 **Add pre-selection for Farm C** (select-before-engineer):
```python
def pre_select_sensors(farm_name, feature_description_path, max_sensors=100):
    """For Farm C (957 sensors): use domain knowledge + variance + 
    correlation on RAW sensor values to select top-N sensors BEFORE
    engineering rolling stats. Prevents 5,700-column explosion."""
```

This is a new step that runs BEFORE `add_rolling_features()`, not after.

### Gold output
- `data/gold/farm=A/dataset_id=3/features.parquet`
- Contains only ML-ready numeric features + metadata columns
- Features are leakage-safe (causal rolling windows only)

### 🆕 Representative Development Subset
Define explicitly (not random sampling):
```
Farm A development subset:
  - 2 normal datasets (different turbines)
  - 2 anomaly datasets (different fault types)
  - Covers seasonal variation
  - Total: 4 of 22 datasets
```
Use this for rapid iteration. Full farm validation only at phase gates.

### Verification
- Gold features for dev subset are correct (spot-check rolling stats)
- No future information leakage (rolling windows are causal)
- Farm C pre-selection reduces to ≤100 sensors before engineering

---

## Phase 3 — Baseline + Model Comparison

> **Goal**: Establish which model architecture actually works best. No premature ensemble.

🆕 **Experimental design** (compare fairly, then select winner):

```
Experiment 1: Isolation Forest (current baseline)
Experiment 2: Dense Autoencoder (existing, needs persistence fix)
Experiment 3: LSTM Autoencoder (new, captures temporal patterns)
                    ↓
            Evaluate ALL on same temporal validation split
                    ↓
            Select best performer on CARE composite score
                    ↓
            Only then consider ensemble (if empirically justified)
```

### [MODIFY] `src/model.py`
- Fix autoencoder persistence (save Keras model + preprocessor)
- Add LSTM Autoencoder architecture
- Each model must implement: `fit()`, `score()`, `save()`, `load()`

### 🆕 Proper evaluation protocol
- Temporal validation split (already in code — tail-split, not random)
- Cross-validation within Farm A: leave-one-turbine-out
- Report CARE scores per model on the SAME datasets
- **Winner criterion**: highest CARE composite on held-out anomaly datasets

### 🆕 Anomaly score display (dashboard spec)
Do NOT show fake confidence. Show:
```
Decision: 🔴 ANOMALY
Anomaly score: 0.87
Threshold: 0.65
Score above threshold: +0.22
```

For Autoencoder only: show per-feature reconstruction error as attribution.
For Isolation Forest: omit per-feature attribution (not trustworthy).

### Verification
- All 3 models train and evaluate on Farm A dev subset
- CARE scores are comparable (same evaluation function, same datasets)
- Best model is selected with documented justification

---

## Phase 4 — Multi-Farm Generalization

> **Goal**: Extend to Farms B and C. Cross-farm evaluation as formal acceptance criterion.

### Prerequisites
- Download Farm B and Farm C datasets from Kaggle

### Steps
1. Ingest Farm B → Bronze → Silver → Gold
2. Ingest Farm C → Bronze → Silver → Gold (with pre-selection)
3. Train best model on Farm A → evaluate on A, B, C
4. Train on A+B → evaluate on C
5. Train on A+B+C → evaluate on all

### 🆕 Formal acceptance criterion
Cross-farm evaluation is **not optional** — it's a model acceptance gate:
```
Train: Farm A
Test:  Farm B, Farm C
Requirement: CARE composite > 0.3 on unseen farm
```

### Add `global` training strategy
```python
parser.add_argument("--training-strategy", choices=["turbine", "farm", "global"])
```
`global` = pooled across ALL farms with per-asset normalization.

### Verification
- Pipeline runs on all 3 farms without memory errors
- Cross-farm CARE scores documented
- Farm C runs with pre-selected sensors (not all 957)

---

## Phase 5 — MLflow + Model Registry + Quality Gates

> **Goal**: Full experiment tracking with formal promotion workflow.

### [NEW] `src/tracking/mlflow_tracker.py`
- Wraps training runs in MLflow context
- Logs: parameters, CARE metrics, artifacts (model, feature schema, plots)
- Tags: farm_name, git_commit, training_strategy

### 🆕 Model Registry with promotion workflow
```
Training Run → MLflow Experiment
                    ↓
              Candidate Model
                    ↓
              Quality Gate Check:
                coverage ≥ current_production
                earliness ≥ current_production  
                false_alarm_rate ≤ threshold
                    ↓
              PASS → Staging → (manual approval) → Production
              FAIL → Rejected (logged with reason)
```

### 🆕 Model bundle metadata enhancement
Add to every saved model bundle:
```python
{
    "model_version": "1.3.0",
    "data_version": "dvc-abc123",  
    "feature_schema_hash": "sha256:...",
    "training_date": "2026-08-26T11:00:00",
    "training_farms": ["A"],
    "training_strategy": "farm",
    "care_composite": 0.45,
    ...existing fields...
}
```

### Verification
- MLflow UI shows experiments with all metrics
- Model registry has Candidate/Staging/Production stages
- Quality gate blocks inferior models from promotion

---

## Phase 6 — FastAPI Enhancement

> **Goal**: Complete API before dashboard. Dashboard will be a client of this API.

### [MODIFY] `src/serve.py`

1. **`POST /batch_predict`** — accepts CSV upload, returns per-row scores + aggregated events
2. **`POST /predict/farm/{farm_name}`** — farm-model serving
3. **`GET /model/info/{model_name}`** — metadata (training date, features, metrics)
4. **`GET /farms`** — list farms and their turbine counts
5. 🆕 **Bounded model cache** — `functools.lru_cache` or `cachetools.LRUCache(maxsize=10)`
6. 🆕 **Schema compatibility check** — verify incoming features match model's expected schema
7. 🆕 **Data freshness header** — response includes `data_age_minutes` if timestamps are old
8. **Prometheus metrics** via `prometheus-fastapi-instrumentator`

### 🆕 Event-based response format for batch predictions
```json
{
    "events": [
        {
            "event_id": 1,
            "start": "2026-08-20T14:00:00",
            "end": "2026-08-20T19:00:00",  
            "duration_hours": 5.0,
            "peak_score": 0.91,
            "mean_score": 0.78
        }
    ],
    "per_row_scores": [...]
}
```

### Verification
- All endpoints return correct responses
- Batch predict handles Farm A CSV upload
- Schema mismatch returns 422 with clear error
- Smoke test passes: `python tests/smoke_test_api.py`

---

## Phase 7 — Dashboard (Streamlit)

> **Goal**: Interactive predictive maintenance dashboard. Calls FastAPI only — no ML code.

🆕 **Architecture**: `Streamlit → HTTP requests → FastAPI → Model`

### [NEW] `dashboard/app.py`
Multi-page Streamlit app.

### Page 1: Upload & Predict
- CSV file upload widget + Farm selector
- Calls `POST /batch_predict` via `requests.post()`
- Shows decision: 🔴 ANOMALY or 🟢 NORMAL with raw score + threshold (NOT fake confidence %)
- Anomaly score time series plot with threshold line
- 🆕 For Autoencoder: per-feature reconstruction error bar chart
- 🆕 Event aggregation view (not raw row-level flags)

### Page 2: Farm Overview  
- Grid of turbines per farm with status indicators
- Summary stats (total anomalies, coverage, false alarm rate)
- Calls `GET /farms` and `GET /models`

### Page 3: Turbine Deep-Dive
- Power curve visualization (actual vs expected)
- Sensor time series with anomaly regions shaded
- Rolling anomaly score chart
- 🆕 Event timeline (start, end, duration, peak score)

### Page 4: Model Performance
- CARE score breakdown per dataset
- Model comparison table (if multiple trained)
- Calls MLflow API for experiment history

### Page 5: Monitoring
- 🆕 **System monitoring**: API latency, error rate, request count
- 🆕 **ML monitoring**: feature drift, prediction distribution, anomaly rate
- 🆕 **Data freshness**: staleness indicator per turbine

### [NEW] `dashboard/api_client.py`
Thin wrapper around FastAPI endpoints:
```python
class WindTurbineAPIClient:
    def predict(self, farm, readings) -> PredictResponse
    def batch_predict(self, farm, csv_file) -> BatchResponse
    def get_farms(self) -> list[Farm]
    def get_model_info(self, name) -> ModelInfo
```

### Verification
- Dashboard loads and all pages render
- File upload → API call → results display works end-to-end
- All visualizations render with real Farm A data
- No `import model` or `import features` in dashboard code

---

## Phase 8 — DVC + Docker + CI/CD

### DVC
- `dvc init` + track `data/bronze/`, `data/silver/`, `data/gold/`
- `dvc.yaml` pipeline: ingest → validate → featurize → train → evaluate
- `params.yaml` for all tunable parameters
- Remote: local initially, Azure Blob later

### Docker
- `docker/Dockerfile.api` — FastAPI service
- `docker/Dockerfile.dashboard` — Streamlit service  
- `docker-compose.yml` — API + Dashboard + MLflow + PostgreSQL

### CI/CD
- `.github/workflows/ci.yml`: lint (ruff) → type check (mypy) → unit tests → integration tests → smoke test → Docker build
- `.github/workflows/cd.yml`: build → push → deploy staging → smoke test → approval → production
- 🆕 **Model quality gate in CI**: small regression test on synthetic data

### Verification
- `docker-compose up` starts all services
- CI pipeline passes on GitHub Actions
- DVC can reproduce the full pipeline from scratch

---

## Phase 9 — Monitoring + Retraining

### 🆕 Two-layer monitoring

**System monitoring** (Prometheus + Grafana):
- API request count, latency p50/p95/p99, error rate
- CPU, memory usage
- Container health

**ML monitoring** (Evidently or custom):
- Feature distribution drift (KS test / PSI)
- Prediction distribution shift
- Anomaly rate tracking
- 🆕 Input schema drift detection
- 🆕 Data freshness per turbine
- Model age / staleness

### Automated retraining
```
Drift detected OR scheduled trigger
        ↓
Retrain pipeline (DVC repro)
        ↓
MLflow logs new candidate
        ↓
Quality gate: new_model.care > production_model.care
        ↓
PASS → promote to Staging → manual approval → Production
FAIL → reject, log reason, alert
```

---

## Phase 10 — Cloud Deployment (Azure)

- Azure Blob Storage for DVC remote + raw data
- Azure Container Registry for Docker images
- Azure Container Apps for FastAPI + Streamlit
- Azure ML or self-hosted MLflow
- Azure Monitor for system metrics

---

## Open Questions

> [!IMPORTANT]
> ### 1. Do you have Farm B and C data downloaded?
> Phase 4 requires them. Farm A is present (~1.6 GB). If not downloaded, this blocks multi-farm work.

> [!IMPORTANT]
> ### 2. Machine RAM?
> Farm C's 957 features determine how aggressive pre-selection needs to be. With 16 GB RAM, we need ≤100 sensors. With 32+ GB, we can be less aggressive.

> [!IMPORTANT]
> ### 3. Start Phase 0 now?
> Shall I begin implementing Phase 0 (config refactor, project structure, YAML configs) immediately?

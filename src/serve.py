"""
serve.py
=========
FastAPI service exposing trained per-turbine Normal Behavior Models for
inference. Models come from:

    python -m src.run_pipeline --model isolation_forest --save-models

which writes one joblib bundle per turbine/sub-dataset to models/.

Run locally:
    uvicorn src.serve:app --reload --port 8000

Endpoints:
    GET  /health                       liveness check, no auth required
    GET  /models                       list turbines with a trained model available
    POST /predict/{dataset_name}       score a window of recent readings

--- Statelessness, and what that means for callers ---
This API is intentionally stateless: it does not remember readings between
requests. Each /predict call must include a WINDOW of recent readings (at
least max(config.ROLLING_WINDOWS) = 36 points / 6 hours at 10-min
resolution), because the model's features include rolling statistics that
need that lookback. The score returned is for the LAST reading in the
window you send.

Why stateless instead of the server keeping a rolling buffer per turbine:
statelessness means you can run multiple replicas behind a load balancer
with no shared state to synchronize, and a restart doesn't lose in-flight
history. The tradeoff is that the CALLER (your SCADA integration layer) is
responsible for keeping a rolling window and re-sending it each call. If
you'd rather the server hold that state, that's a reasonable evolution —
just know it adds a datastore dependency (e.g. Redis) and complicates
horizontal scaling.

--- Auth ---
Every non-health request must include header `X-API-Key: <value>` matching
the API_KEY environment variable. This is a single shared secret — a
reasonable starting point for a small internal tool, but swap it for
per-caller keys or your company's SSO/OAuth gateway before this is used
more broadly or touches real turbine data. If API_KEY is unset, auth is
DISABLED — that's meant for local dev only; always set it in any deployed
environment.

--- Known gap: single-point scoring vs. the sustained-event evaluation ---
The offline evaluation (src/evaluation.py, outputs/evaluation_report.csv)
only counts a detection after MIN_EVENT_LENGTH (default 6) *consecutive*
flagged points — that's what keeps the Reliability metric meaningful
instead of firing on every noisy single-point spike. This API does NOT do
that smoothing: each /predict call returns a raw score-vs-threshold
comparison for one point, with no memory of previous calls. A caller that
pages someone on every single `flagged: true` will see more false alarms
than the Reliability score in the evaluation report implies.
`min_event_length` is included in the response so a consuming system can
apply the same "N of the last N calls flagged" rule itself before
alerting — this API deliberately stays stateless (see above) rather than
tracking that history server-side.
"""

from __future__ import annotations

import hmac
import os
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from . import config, features

app = FastAPI(
    title="Wind Turbine Early Fault Detection API",
    description="Scores a recent window of SCADA readings for early anomaly "
                "signs, using a per-turbine Normal Behavior Model.",
    version="0.1.0",
)

API_KEY = os.environ.get("API_KEY")  # unset = auth disabled (local dev only)
MAX_READINGS = 1000  # ~1 week at 10-min resolution; see _check_readings_size

_MODEL_CACHE: dict[str, dict] = {}


def _check_auth(x_api_key: Optional[str]) -> None:
    # hmac.compare_digest instead of != : plain string comparison
    # short-circuits on the first differing character, which leaks the
    # key's length and content byte-by-byte via response timing. This
    # matters more once the API is reachable outside your own machine.
    if API_KEY and not hmac.compare_digest(x_api_key or "", API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")


def _list_available_models() -> list[str]:
    if not config.MODEL_DIR.exists():
        return []
    return sorted(p.stem for p in config.MODEL_DIR.glob("*.joblib"))


def _load_bundle(dataset_name: str) -> dict:
    if dataset_name in _MODEL_CACHE:
        return _MODEL_CACHE[dataset_name]
    path = config.MODEL_DIR / f"{dataset_name}.joblib"
    if not path.exists():
        available = _list_available_models()
        raise HTTPException(
            status_code=404,
            detail=f"No trained model found for '{dataset_name}'. Available: {available}",
        )
    bundle = joblib.load(path)
    _MODEL_CACHE[dataset_name] = bundle
    return bundle


class PredictRequest(BaseModel):
    readings: list[dict]  # each dict: time_stamp + the turbine's raw sensor columns


class PredictResponse(BaseModel):
    dataset_name: str
    asset_id: Optional[str] = None
    latest_timestamp: str
    anomaly_score: float
    threshold: float
    flagged: bool
    min_event_length: int
    n_readings_used: int
    warning: Optional[str] = None


@app.get("/health")
def health():
    """No auth required — for load balancer / orchestrator liveness checks."""
    return {"status": "ok", "models_available": len(_list_available_models())}


@app.get("/models")
def list_models(x_api_key: Optional[str] = Header(default=None)):
    _check_auth(x_api_key)
    return {"models": _list_available_models()}


@app.post("/predict/{dataset_name}", response_model=PredictResponse)
def predict(dataset_name: str, request: PredictRequest,
            x_api_key: Optional[str] = Header(default=None)):
    _check_auth(x_api_key)

    if not request.readings:
        raise HTTPException(status_code=422, detail="'readings' cannot be empty")
    if len(request.readings) > MAX_READINGS:
        raise HTTPException(
            status_code=422,
            detail=f"Too many readings ({len(request.readings)}); max {MAX_READINGS} "
                   f"per request. Feature engineering is O(n x windows x columns), "
                   f"so an unbounded request risks CPU/memory exhaustion.",
        )

    bundle = _load_bundle(dataset_name)

    df = pd.DataFrame(request.readings)
    if config.TIME_COL not in df.columns:
        raise HTTPException(status_code=422, detail=f"Each reading needs a '{config.TIME_COL}' field")
    df[config.TIME_COL] = pd.to_datetime(df[config.TIME_COL], errors="coerce")
    if df[config.TIME_COL].isna().any():
        raise HTTPException(status_code=422, detail=f"One or more '{config.TIME_COL}' values failed to parse")
    df = df.sort_values(config.TIME_COL).reset_index(drop=True)

    warning = None
    min_recommended = max(config.ROLLING_WINDOWS)
    if len(df) < min_recommended:
        warning = (
            f"Only {len(df)} readings supplied; recommend >= {min_recommended} "
            f"(6h of 10-min data) for reliable rolling features. The score "
            f"below used a shorter effective lookback."
        )

    df_feat, _ = features.engineer_features_for_serving(
        df, bundle["power_curve_reference"],
        feature_descriptions=bundle.get("feature_descriptions"),
    )

    missing = [c for c in bundle["feature_cols"] if c not in df_feat.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Input readings are missing columns this model expects "
                   f"(showing up to 5): {missing[:5]}",
        )

    X_latest = df_feat.iloc[[-1]][bundle["feature_cols"]]
    score = float(bundle["detector"].score(X_latest)[0])
    flagged = bool(score >= bundle["threshold"])

    return PredictResponse(
        dataset_name=dataset_name,
        asset_id=bundle.get("asset_id"),
        latest_timestamp=str(df[config.TIME_COL].iloc[-1]),
        anomaly_score=score,
        threshold=bundle["threshold"],
        flagged=flagged,
        min_event_length=bundle.get("min_event_length", 1),
        n_readings_used=len(df),
        warning=warning,
    )
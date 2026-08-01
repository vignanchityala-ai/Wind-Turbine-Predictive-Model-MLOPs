"""
smoke_test_api.py
==================
End-to-end smoke test for the serving API: builds a real request payload
from a synthetic dataset, hits a running server, and asserts the response
looks right. Uses only the standard library (no `requests` dependency).

Usage:
    python tests/make_synthetic_data.py
    python -m src.run_pipeline --model isolation_forest --save-models
    uvicorn src.serve:app --port 8000 &
    python tests/smoke_test_api.py
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import config, data_loader  # noqa: E402

BASE_URL = "http://localhost:8000"


def _get(path: str):
    with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=5) as resp:
        return resp.status, json.loads(resp.read())


def _post(path: str, payload: dict):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE_URL}{path}", data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def main():
    raw_dir = config.PROJECT_ROOT / "data" / "raw" / "Wind Farm A"
    csvs = [p for p in raw_dir.rglob("*.csv") if "event_info" not in p.stem.lower()]
    anomaly_paths = [p for p in csvs if "anomaly" in p.stem.lower()]
    if not anomaly_paths:
        print("No synthetic anomaly dataset found under", raw_dir,
              "— run tests/make_synthetic_data.py first.")
        sys.exit(1)

    sub = data_loader.load_subdataset(anomaly_paths[0])
    window = sub.prediction.tail(40).copy()
    window[config.TIME_COL] = window[config.TIME_COL].astype(str)
    drop_cols = [c for c in (config.ID_COL, config.ASSET_COL, config.SPLIT_COL, config.STATUS_COL)
                 if c in window.columns]
    readings = window.drop(columns=drop_cols).to_dict("records")

    status, body = _get("/health")
    assert status == 200, f"/health failed: {status} {body}"
    print("/health OK:", body)

    status, body = _post(f"/predict/{sub.name}", {"readings": readings})
    assert status == 200, f"/predict failed: {status} {body}"
    assert "anomaly_score" in body and "flagged" in body, f"Unexpected response shape: {body}"
    print(f"/predict/{sub.name} OK:", body)

    status, body = _post("/predict/NOT_A_REAL_MODEL", {"readings": readings})
    assert status == 404, f"expected 404 for unknown model, got {status}"
    print("Unknown-model 404 check OK")

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()

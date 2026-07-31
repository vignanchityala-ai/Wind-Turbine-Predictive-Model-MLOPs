"""
make_synthetic_data.py
========================
Generates small synthetic sub-datasets that MIMIC the Wind Farm A schema
(timestamp, asset_id, train_test, status_type_id, power/wind_speed + generic
sensor columns, event_info.csv) so the pipeline can be smoke-tested end to
end without the real Kaggle download. This is purely for validating that
the code runs correctly — not a substitute for the real data.
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)


def _make_series(n, freq="10min", start="2020-01-01"):
    return pd.date_range(start=start, periods=n, freq=freq)


def make_normal_dataset(out_dir: Path, asset_id="WT01", n_train_days=365, n_pred_days=30, n_sensors=20):
    out_dir.mkdir(parents=True, exist_ok=True)
    n_train = n_train_days * 144
    n_pred = n_pred_days * 144
    n = n_train + n_pred

    ts = _make_series(n)
    wind_speed = np.clip(RNG.weibull(2, n) * 7, 0, 25)
    rated = 2000.0
    power = np.clip((wind_speed / 12) ** 3 * rated, 0, rated) + RNG.normal(0, 20, n)
    power = np.clip(power, 0, rated * 1.05)

    data = {
        "id": np.arange(n),
        "time_stamp": ts,
        "asset_id": asset_id,
        "train_test": ["train"] * n_train + ["prediction"] * n_pred,
        "status_type_id": 0,
        "power": power,
        "wind_speed": wind_speed,
    }
    for i in range(n_sensors):
        base = RNG.normal(loc=RNG.uniform(10, 100), scale=1.0)
        data[f"sensor_{i}"] = base + 0.02 * power / 100 + RNG.normal(0, 2, n)

    df = pd.DataFrame(data)
    df.to_csv(out_dir / f"{asset_id}_normal.csv", index=False)

    event_info = pd.DataFrame([{
        "event_id": f"{asset_id}_normal",
        "event_label": "normal",
        "event_start": "",
        "event_end": "",
        "asset_id": asset_id,
    }])
    event_info.to_csv(out_dir / "event_info.csv", index=False)


def make_anomaly_dataset(out_dir: Path, asset_id="WT02", n_train_days=365, n_pred_days=40, n_sensors=20):
    out_dir.mkdir(parents=True, exist_ok=True)
    n_train = n_train_days * 144
    n_pred = n_pred_days * 144
    n = n_train + n_pred

    ts = _make_series(n)
    wind_speed = np.clip(RNG.weibull(2, n) * 7, 0, 25)
    rated = 2000.0
    power = np.clip((wind_speed / 12) ** 3 * rated, 0, rated) + RNG.normal(0, 20, n)
    power = np.clip(power, 0, rated * 1.05)

    status = np.zeros(n, dtype=int)

    # Inject a degrading fault in the last ~5 days of the prediction window:
    # power output progressively falls below the expected power curve, then
    # the turbine actually faults (status 4) for the final day.
    fault_lead_days = 5
    fault_lead_steps = fault_lead_days * 144
    fault_start_idx = n - 144  # last day = actual fault / down status
    degrade_start_idx = n - fault_lead_steps

    degrade_ramp = np.linspace(0, 1, n - degrade_start_idx)
    power[degrade_start_idx:] *= (1 - 0.6 * degrade_ramp)
    status[fault_start_idx:] = 4  # fault status for the final day

    data = {
        "id": np.arange(n),
        "time_stamp": ts,
        "asset_id": asset_id,
        "train_test": ["train"] * n_train + ["prediction"] * n_pred,
        "status_type_id": status,
        "power": power,
        "wind_speed": wind_speed,
    }
    for i in range(n_sensors):
        base = RNG.normal(loc=RNG.uniform(10, 100), scale=1.0)
        vals = base + 0.02 * power / 100 + RNG.normal(0, 2, n)
        if i < 3:  # a few sensors also drift during degradation (e.g. temp rise)
            vals[degrade_start_idx:] += np.linspace(0, 8, n - degrade_start_idx)
        data[f"sensor_{i}"] = vals

    df = pd.DataFrame(data)
    df.to_csv(out_dir / f"{asset_id}_anomaly.csv", index=False)

    event_info = pd.DataFrame([{
        "event_id": f"{asset_id}_anomaly",
        "event_label": "anomaly",
        "event_start": str(ts[degrade_start_idx]),
        "event_end": str(ts[fault_start_idx]),
        "asset_id": asset_id,
    }])
    event_info.to_csv(out_dir / "event_info.csv", index=False)


def build_synthetic_farm(root: Path, n_normal=2, n_anomaly=2):
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    for i in range(n_normal):
        d = root / f"dataset_normal_{i}"
        make_normal_dataset(d, asset_id=f"WT{i:02d}")

    for i in range(n_anomaly):
        d = root / f"dataset_anomaly_{i}"
        make_anomaly_dataset(d, asset_id=f"WT{100+i:02d}")

    print(f"Synthetic Wind Farm A-like data written to {root}")


if __name__ == "__main__":
    import sys
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/claude/wind_turbine_pipeline/data/raw/Wind Farm A")
    build_synthetic_farm(target)

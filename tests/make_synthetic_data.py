"""
make_synthetic_data.py
========================
Generates a small synthetic Wind Farm A-like dataset that mirrors the REAL
confirmed structure, not just a schema-compatible approximation:

  - A flat "datasets/" folder with bare-integer-named CSVs (0.csv, 1.csv,
    ...) -- matching the real Kaggle layout, not one folder per dataset.
  - ONE shared "comma_event_info.csv" at the farm root (sibling of
    "datasets/"), one row per dataset -- the real confirmed filename and
    layout, columns: event_id, event_label, event_start, event_start_id,
    event_end, event_end_id, event_description.
  - event_start/event_end in DD-MM-YYYY HH:MM format -- the real confirmed
    date format -- so this exercises the same date-parsing path the real
    data needs (_parse_event_datetime's DD-MM-YYYY branch), not only the
    ISO-format branch a schema-only approximation would ever hit.
  - event_start_id/event_end_id populated with the actual row 'id' values,
    exercising the id-based resolution path in data_loader.py too.
  - event_id is the bare integer matching the data CSV's filename, same as
    the real data (confirmed: event_id is literally the file number).

This closes a real testing gap an earlier version had: writing one
event_info.csv per sub-directory only worked because each dataset lived in
its own folder, and never exercised the shared-file, event_id-matching
logic the real Kaggle data actually needs. A regression there could have
shipped without CI catching it.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

FAULT_DESCRIPTIONS = [
    "Gearbox failure", "Generator bearing failure",
    "Transformer failure", "Hydraulic group",
]


def _make_series(n, freq="10min", start="2020-01-01"):
    return pd.date_range(start=start, periods=n, freq=freq)


def _make_dataset(event_id: int, asset_id: str, is_anomaly: bool,
                   n_train_days: int = 365, n_pred_days: int = None,
                   n_sensors: int = 20) -> tuple[pd.DataFrame, dict]:
    """Generate one dataset's raw data plus its event_info row."""
    n_pred_days = n_pred_days if n_pred_days is not None else (40 if is_anomaly else 30)
    n_train = n_train_days * 144
    n_pred = n_pred_days * 144
    n = n_train + n_pred

    ts = _make_series(n)
    wind_speed = np.clip(RNG.weibull(2, n) * 7, 0, 25)
    rated = 2000.0
    power = np.clip((wind_speed / 12) ** 3 * rated, 0, rated) + RNG.normal(0, 20, n)
    power = np.clip(power, 0, rated * 1.05)
    status = np.zeros(n, dtype=int)

    event_start_idx = event_end_idx = None
    description = None
    if is_anomaly:
        # Inject a degrading fault in the last ~5 days of the prediction
        # window: power progressively falls below the expected power curve,
        # then the turbine actually faults (status 4) for the final day.
        fault_lead_days = 5
        fault_lead_steps = fault_lead_days * 144
        fault_start_idx = n - 144
        degrade_start_idx = n - fault_lead_steps

        degrade_ramp = np.linspace(0, 1, n - degrade_start_idx)
        power[degrade_start_idx:] *= (1 - 0.6 * degrade_ramp)
        status[fault_start_idx:] = 4

        event_start_idx = degrade_start_idx
        event_end_idx = fault_start_idx
        description = FAULT_DESCRIPTIONS[event_id % len(FAULT_DESCRIPTIONS)]

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
        if is_anomaly and i < 3:  # a few sensors also drift during degradation
            vals[event_start_idx:] += np.linspace(0, 8, n - event_start_idx)
        data[f"sensor_{i}"] = vals

    df = pd.DataFrame(data)

    event_row = {
        "event_id": event_id,
        "event_label": "anomaly" if is_anomaly else "normal",
        "event_start": ts[event_start_idx].strftime("%d-%m-%Y %H:%M") if is_anomaly else "",
        "event_start_id": int(event_start_idx) if is_anomaly else "",
        "event_end": ts[event_end_idx].strftime("%d-%m-%Y %H:%M") if is_anomaly else "",
        "event_end_id": int(event_end_idx) if is_anomaly else "",
        "event_description": description or "",
    }
    return df, event_row


def build_synthetic_farm(root: Path, n_normal: int = 2, n_anomaly: int = 2) -> None:
    """
    Build a synthetic Wind Farm A-like dataset matching the real confirmed
    layout: root/datasets/<event_id>.csv (flat, bare-integer names) plus
    root/comma_event_info.csv (one shared file, one row per dataset).
    """
    if root.exists():
        shutil.rmtree(root)
    datasets_dir = root / "datasets"
    datasets_dir.mkdir(parents=True)

    event_rows = []
    event_id = 0
    for i in range(n_normal):
        df, row = _make_dataset(event_id, asset_id=f"WT{i:02d}", is_anomaly=False)
        df.to_csv(datasets_dir / f"{event_id}.csv", index=False)
        event_rows.append(row)
        event_id += 1

    for i in range(n_anomaly):
        df, row = _make_dataset(event_id, asset_id=f"WT{100 + i:02d}", is_anomaly=True)
        df.to_csv(datasets_dir / f"{event_id}.csv", index=False)
        event_rows.append(row)
        event_id += 1

    pd.DataFrame(event_rows).to_csv(root / "comma_event_info.csv", index=False)

    print(f"Synthetic Wind Farm A-like data written to {root} "
          f"({n_normal} normal + {n_anomaly} anomaly dataset(s), "
          f"real-structure event_info at {root / 'comma_event_info.csv'})")


def build_synthetic_feature_descriptions(path: Path) -> None:
    """
    Small feature-description fixture matching this generator's own sensor
    naming (sensor_0, power, wind_speed), with sensor_0 flagged as an angle.
    Exercises the circular-statistics / feature_descriptions code path in
    CI, which otherwise only ever gets tested manually.
    """
    rows = [
        {"sensor_name": "sensor_0", "statistics_type": "average",
         "description": "Test angle sensor", "unit": "degrees",
         "is_angle": "TRUE", "is_counter": "FALSE"},
        {"sensor_name": "power", "statistics_type": "average",
         "description": "Grid power", "unit": "kW",
         "is_angle": "FALSE", "is_counter": "FALSE"},
        {"sensor_name": "wind_speed", "statistics_type": "average",
         "description": "Windspeed", "unit": "m/s",
         "is_angle": "FALSE", "is_counter": "FALSE"},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Synthetic feature description fixture written to {path}")


if __name__ == "__main__":
    # Project-relative default instead of a machine-specific absolute path
    # -- the old default ("/home/claude/wind_turbine_pipeline/...") only
    # ever worked on the original dev sandbox, not on anyone else's machine.
    default_target = Path(__file__).resolve().parents[1] / "data" / "raw" / "Wind Farm A"
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else default_target

    build_synthetic_farm(target)
    build_synthetic_feature_descriptions(target / "wind_farm_a_feature_description.csv")
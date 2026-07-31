"""
data_loader.py
===============
Loading, validating, and lightly cleaning individual Wind Farm A sub-datasets.

Each sub-dataset lives in its own folder (naming varies by release, typically
something like `Wind Farm A/datasets/0.csv` plus an `event_info.csv`, or one
folder per event containing `<id>.csv` + metadata). This loader is written
defensively: it auto-detects the event metadata file and tolerates minor
schema drift (e.g. semicolon vs comma separators — the Kaggle mirror already
converts these to commas, but we handle both just in case).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from . import config


@dataclass
class SubDataset:
    """Container for one turbine's train+prediction episode."""
    name: str
    df: pd.DataFrame
    event_label: Optional[str] = None      # 'anomaly' | 'normal' | None
    event_start: Optional[pd.Timestamp] = None
    event_end: Optional[pd.Timestamp] = None
    asset_id: Optional[str] = None

    @property
    def train(self) -> pd.DataFrame:
        return self.df[self.df[config.SPLIT_COL] == config.TRAIN_VALUE]

    @property
    def prediction(self) -> pd.DataFrame:
        return self.df[self.df[config.SPLIT_COL] == config.PREDICTION_VALUE]

    @property
    def is_anomaly(self) -> bool:
        return (self.event_label or "").lower() == "anomaly"


def _read_csv_robust(path: Path) -> pd.DataFrame:
    """Read a SCADA csv, tolerating comma or semicolon separators."""
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path)
    return df


def _find_event_metadata(folder: Path) -> dict:
    """Look for event_info.csv / .json next to the data file; return {} if absent."""
    for candidate in ("event_info.csv", "event_info.json", "metadata.json"):
        p = folder / candidate
        if p.exists():
            if p.suffix == ".json":
                return json.loads(p.read_text())
            else:
                meta_df = pd.read_csv(p, sep=None, engine="python")
                return meta_df.iloc[0].to_dict() if len(meta_df) else {}
    return {}


def load_subdataset(csv_path: Path) -> SubDataset:
    """Load a single sub-dataset CSV plus its event metadata (if present)."""
    df = _read_csv_robust(csv_path)

    # Normalize column names (strip whitespace, lowercase-safe rename for
    # known variants across releases)
    df.columns = [c.strip() for c in df.columns]
    rename_map = {}
    lower_cols = {c.lower(): c for c in df.columns}
    for canon, variants in {
        config.TIME_COL: ["time_stamp", "timestamp", "time"],
        config.ASSET_COL: ["asset_id", "turbine_id", "wt_id"],
        config.SPLIT_COL: ["train_test", "split"],
        config.STATUS_COL: ["status_type_id", "status_id", "status"],
        config.ID_COL: ["id", "row_id"],
    }.items():
        for v in variants:
            if v in lower_cols and lower_cols[v] != canon:
                rename_map[lower_cols[v]] = canon
                break
    df = df.rename(columns=rename_map)

    missing = [c for c in (config.TIME_COL, config.SPLIT_COL, config.STATUS_COL)
               if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path.name}: missing expected columns {missing}. "
            f"Found columns: {list(df.columns)[:10]}..."
        )

    df[config.TIME_COL] = pd.to_datetime(df[config.TIME_COL], errors="coerce")
    df = df.sort_values(config.TIME_COL).reset_index(drop=True)
    df[config.STATUS_COL] = pd.to_numeric(df[config.STATUS_COL], errors="coerce")

    meta = _find_event_metadata(csv_path.parent)
    event_label = meta.get("event_label") or meta.get("label")
    event_start = meta.get("event_start")
    event_end = meta.get("event_end")
    asset_id = meta.get("asset_id") or (
        df[config.ASSET_COL].iloc[0] if config.ASSET_COL in df.columns else None
    )

    if event_start is not None:
        event_start = pd.to_datetime(event_start, errors="coerce")
    if event_end is not None:
        event_end = pd.to_datetime(event_end, errors="coerce")

    return SubDataset(
        name=csv_path.stem,
        df=df,
        event_label=event_label,
        event_start=event_start,
        event_end=event_end,
        asset_id=asset_id,
    )


def discover_subdatasets(raw_dir: Path = config.RAW_DATA_DIR) -> list[Path]:
    """
    Find candidate SCADA CSVs under raw_dir. Skips files that are clearly
    metadata (event_info*, sensor description files, etc.).
    """
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"{raw_dir} does not exist. Download the dataset from Kaggle "
            f"(https://www.kaggle.com/datasets/azizkasimov/"
            f"wind-turbine-scada-data-for-early-fault-detection), extract "
            f"'Wind Farm A' into that path, and re-run."
        )

    skip_name_fragments = ("event_info", "sensor", "description", "readme", "metadata")
    csvs = [
        p for p in raw_dir.rglob("*.csv")
        if not any(frag in p.stem.lower() for frag in skip_name_fragments)
    ]
    return sorted(csvs)


def load_all(raw_dir: Path = config.RAW_DATA_DIR) -> list[SubDataset]:
    paths = discover_subdatasets(raw_dir)
    if not paths:
        raise FileNotFoundError(f"No SCADA csv files found under {raw_dir}")
    return [load_subdataset(p) for p in paths]

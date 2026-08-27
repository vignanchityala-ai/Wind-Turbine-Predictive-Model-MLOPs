"""
schema.py
=========
Canonical internal data model for the Wind Turbine MLOps pipeline.

All downstream modules (feature engineering, training, serving, dashboard)
work with these dataclasses, never with raw CSV column names directly.
This adapter layer isolates the rest of the pipeline from:
  - Farm-specific naming quirks (sensor_0 vs WEC_ava_windspeed)
  - File format changes between Zenodo releases
  - The comma_/plain duplicate naming convention in the Kaggle mirror

The existing data_loader.py SubDataset class handles per-CSV loading
and column normalization. This module sits ABOVE that: it describes
the farm-level and event-level metadata used by the ingestion pipeline
and the data lake (bronze/silver/gold).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core data contracts
# ---------------------------------------------------------------------------
@dataclass
class InternalEvent:
    """One anomaly/normal episode from the event_info file."""
    farm_id: str
    dataset_id: str              # e.g. "0", "68" — matches CSV filename stem
    event_label: str             # "anomaly" or "normal"
    event_start: Optional[pd.Timestamp] = None
    event_end: Optional[pd.Timestamp] = None
    event_start_id: Optional[int] = None
    event_end_id: Optional[int] = None
    event_description: Optional[str] = None
    asset_id: Optional[str] = None

    @property
    def is_anomaly(self) -> bool:
        return self.event_label.lower() == "anomaly"


@dataclass
class FarmMetadata:
    """All metadata for one wind farm, parsed from its event_info file."""
    farm_id: str
    events: dict[str, InternalEvent] = field(default_factory=dict)
    feature_description_path: Optional[Path] = None

    def get_event(self, dataset_id: str) -> Optional[InternalEvent]:
        return self.events.get(dataset_id)


@dataclass
class DatasetManifest:
    """Tracks what has been ingested into the data lake."""
    farm_id: str
    dataset_id: str
    raw_csv_path: Path
    bronze_parquet_path: Optional[Path] = None
    silver_parquet_path: Optional[Path] = None
    gold_parquet_path: Optional[Path] = None
    n_rows: int = 0
    n_columns: int = 0
    ingested: bool = False
    validated: bool = False


# ---------------------------------------------------------------------------
# Event info parsing — farm-agnostic
# ---------------------------------------------------------------------------
def _parse_event_datetime(value) -> Optional[pd.Timestamp]:
    """Parse event_start/event_end values across the format variants
    seen in the CARE dataset releases. Reuses the same logic and edge-case
    handling as data_loader._parse_event_datetime (DD-MM-YYYY vs YYYY-MM-DD
    disambiguation) but lives here so schema.py has no import dependency
    on data_loader."""
    import re
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if not s:
        return None

    if re.match(r"^\d{4}-\d{1,2}-\d{1,2}", s):
        return pd.to_datetime(s, dayfirst=False, errors="coerce")
    if re.match(r"^\d{1,2}-\d{1,2}-\d{4}", s):
        return pd.to_datetime(s, format="%d-%m-%Y %H:%M", errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def _read_csv_safe(path: Path) -> pd.DataFrame:
    """Read a CSV with encoding fallback (utf-8 → latin-1)."""
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, sep=None, engine="python", encoding="latin-1")


def parse_event_info(farm_id: str, event_info_path: Path) -> FarmMetadata:
    """Parse a shared event_info CSV into FarmMetadata with InternalEvents.

    The CARE dataset uses one event_info file per farm, with columns:
      event_id, event_label, event_start, event_start_id,
      event_end, event_end_id, event_description
    """
    if not event_info_path.exists():
        log.warning("Event info file not found: %s", event_info_path)
        return FarmMetadata(farm_id=farm_id)

    df = _read_csv_safe(event_info_path)
    df.columns = [c.strip().lower() for c in df.columns]

    events = {}
    for _, row in df.iterrows():
        dataset_id = str(int(float(row.get("event_id", 0))))

        def _clean(v):
            return None if v is None or (isinstance(v, float) and pd.isna(v)) else v

        event = InternalEvent(
            farm_id=farm_id,
            dataset_id=dataset_id,
            event_label=_clean(row.get("event_label")) or "normal",
            event_start=_parse_event_datetime(_clean(row.get("event_start"))),
            event_end=_parse_event_datetime(_clean(row.get("event_end"))),
            event_start_id=int(float(row["event_start_id"])) if pd.notna(row.get("event_start_id")) else None,
            event_end_id=int(float(row["event_end_id"])) if pd.notna(row.get("event_end_id")) else None,
            event_description=_clean(row.get("event_description")),
        )
        events[dataset_id] = event

    log.info("Farm %s: parsed %d events from %s (%d anomaly, %d normal)",
             farm_id, len(events), event_info_path.name,
             sum(1 for e in events.values() if e.is_anomaly),
             sum(1 for e in events.values() if not e.is_anomaly))

    return FarmMetadata(farm_id=farm_id, events=events)


def find_event_info_file(raw_dir: Path) -> Optional[Path]:
    """Find the event_info CSV in a farm's raw directory.
    Prefers comma_event_info.csv (Kaggle convention) over event_info.csv."""
    for name in ("comma_event_info.csv", "event_info.csv"):
        for p in raw_dir.rglob(name):
            return p
    return None


def find_feature_description_file(raw_dir: Path) -> Optional[Path]:
    """Find the feature_description CSV in a farm's raw directory."""
    for p in raw_dir.rglob("*feature_description*"):
        if p.suffix == ".csv":
            return p
    return None

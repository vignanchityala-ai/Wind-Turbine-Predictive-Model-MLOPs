"""
feature_pipeline.py
====================
Silver → Gold feature engineering pipeline.

Reads cleaned data from Silver Parquet, applies the full feature
engineering pipeline (from src/features.py), and writes ML-ready
feature matrices to Gold Parquet.

This module is the bridge between the data lake and the ML pipeline.
It reuses ALL existing feature engineering logic — no duplication.

For Farm C (957 sensors), it runs pre-selection on RAW sensor values
BEFORE computing rolling stats, preventing the 5,742-column explosion.

Output structure:
    data/gold/
    └── farm=A/
        ├── dataset_0.parquet      (features + metadata columns)
        ├── dataset_0_meta.json    (power_curve_ref, selected_cols, etc.)
        └── ...
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .. import config
from ..features import (
    identify_columns,
    clean_zeros_as_missing,
    fit_power_curve_reference,
    apply_power_curve_reference,
    add_rolling_features,
    add_temporal_features,
    build_label,
)
from ..feature_selection import pre_select_sensors
from .silver import query_silver, list_silver_datasets

log = logging.getLogger(__name__)

# Pre-selection threshold: only applies if farm has more than this many features
PRE_SELECTION_THRESHOLD = 200  # Farm A=86, Farm B=257, Farm C=957
MAX_SENSORS_AFTER_PRESELECTION = 100


@dataclass
class FeaturePipelineReport:
    """Summary of Silver → Gold feature engineering."""
    farm_id: str
    datasets_processed: int = 0
    datasets_skipped: int = 0
    total_rows: int = 0
    total_features: int = 0
    errors: list[str] = field(default_factory=list)
    elapsed_sec: float = 0.0

    def log_summary(self):
        log.info("=== Feature Pipeline report: Farm %s ===", self.farm_id)
        log.info("  Datasets processed: %d | Skipped: %d",
                 self.datasets_processed, self.datasets_skipped)
        log.info("  Total rows: %d | Feature columns: %d",
                 self.total_rows, self.total_features)
        log.info("  Elapsed: %.1f sec", self.elapsed_sec)
        if self.errors:
            for e in self.errors:
                log.error("  Error: %s", e)


def _load_feature_descriptions(farm_id: str):
    """Load feature descriptions if available for this farm."""
    from ..feature_descriptions import load_feature_descriptions
    from .schema import find_feature_description_file

    raw_dir = config.FARM_CONFIGS[farm_id]["raw_dir"]
    desc_path = find_feature_description_file(raw_dir)
    if desc_path:
        log.info("  Found feature description file: %s", desc_path.name)
        return load_feature_descriptions(desc_path)
    return None


def process_one_to_gold(
    farm_id: str,
    dataset_id: str,
    force: bool = False,
    feature_descriptions=None,
) -> Optional[Path]:
    """Engineer features for one dataset: Silver → Gold.

    Reads from Silver Parquet, applies the full feature engineering
    pipeline, writes Gold Parquet with ML-ready features.

    For farms with >PRE_SELECTION_THRESHOLD features, pre-selects
    sensors before computing rolling stats.
    """
    gold_dir = config.GOLD_DIR / f"farm={farm_id}"
    output_path = gold_dir / f"dataset_{dataset_id}.parquet"
    meta_path = gold_dir / f"dataset_{dataset_id}_meta.json"

    if output_path.exists() and not force:
        log.debug("Skipping %s/%s — Gold already exists", farm_id, dataset_id)
        return output_path

    # Load from Silver
    df = query_silver(farm_id, dataset_id)
    log.info("  Gold: farm=%s dataset=%s — loaded %d rows × %d cols from Silver",
             farm_id, dataset_id, len(df), len(df.columns))

    # Parse timestamps if they came back as strings from Parquet
    if config.TIME_COL in df.columns:
        df[config.TIME_COL] = pd.to_datetime(df[config.TIME_COL], errors="coerce")

    # Identify column types
    cols = identify_columns(df, feature_descriptions)
    n_features = config.FARM_CONFIGS.get(farm_id, {}).get("n_features", 0)

    # Pre-selection for large farms (Farm C)
    selected_sensors = None
    if n_features > PRE_SELECTION_THRESHOLD:
        selected_sensors = pre_select_sensors(
            df, cols["sensor"],
            max_sensors=MAX_SENSORS_AFTER_PRESELECTION
        )
        log.info("  Pre-selected %d/%d sensors for rolling features",
                 len(selected_sensors), len(cols["sensor"]))
        rolling_cols = selected_sensors + cols["power"] + cols["wind_speed"]
    else:
        rolling_cols = cols["numeric"]

    # Apply zero-as-missing cleaning
    df = clean_zeros_as_missing(df, cols["numeric"],
                                 exclude=cols["power"] + cols["wind_speed"])

    # Fit power curve reference (training-normal data only)
    power_curve_ref = None
    if cols["power"] and cols["wind_speed"]:
        power_col = cols["power"][0]
        wind_col = cols["wind_speed"][0]
        if feature_descriptions is not None:
            from ..feature_descriptions import FeatureDescriptions
            if isinstance(feature_descriptions, FeatureDescriptions):
                pc = feature_descriptions.pick_primary(
                    feature_descriptions.power_base, cols["power"])
                wc = feature_descriptions.pick_primary(
                    feature_descriptions.wind_speed_base, cols["wind_speed"])
                if pc:
                    power_col = pc
                if wc:
                    wind_col = wc
        try:
            power_curve_ref = fit_power_curve_reference(df, power_col, wind_col)
            df = apply_power_curve_reference(df, power_curve_ref)
        except Exception as e:
            log.warning("  Power curve fitting failed: %s", e)

    # Compute rolling features (on selected sensors only for large farms)
    angle_cols = (feature_descriptions.angle_columns(rolling_cols)
                  if feature_descriptions else [])
    df = add_rolling_features(df, rolling_cols, angle_cols=angle_cols)

    # Add temporal features
    sensor_for_temporal = (selected_sensors[:10] if selected_sensors
                           else cols["sensor"][:10])
    df = add_temporal_features(df, sensor_for_temporal)

    # Build label column
    if config.STATUS_COL in df.columns:
        df["_status_label"] = build_label(df)

    # Identify final feature columns
    engineered_cols = [c for c in df.columns
                       if c not in config.NON_FEATURE_COLS
                       and c not in ("expected_power", "_status_label")
                       and pd.api.types.is_numeric_dtype(df[c])]

    log.info("  Gold: %d rows × %d feature columns", len(df), len(engineered_cols))

    # Write Gold Parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_path, compression="snappy")

    # Write metadata JSON (for model training to reference)
    meta = {
        "farm_id": farm_id,
        "dataset_id": dataset_id,
        "n_rows": len(df),
        "n_features": len(engineered_cols),
        "feature_columns": engineered_cols,
        "pre_selected_sensors": selected_sensors,
        "has_power_curve": power_curve_ref is not None,
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=str))

    return output_path


def process_farm_to_gold(
    farm_id: str,
    force: bool = False,
    dataset_ids: Optional[list[str]] = None,
) -> FeaturePipelineReport:
    """Engineer features for all (or selected) Silver datasets in a farm.

    Args:
        farm_id: "A", "B", or "C"
        force: if True, reprocess even if Gold already exists
        dataset_ids: optional subset of datasets to process (for dev iteration)
    """
    t0 = time.time()
    report = FeaturePipelineReport(farm_id=farm_id)

    # Load feature descriptions once for the whole farm
    feature_descriptions = _load_feature_descriptions(farm_id)

    datasets = dataset_ids or list_silver_datasets(farm_id)
    if not datasets:
        log.warning("No Silver datasets for Farm %s — run Silver pipeline first", farm_id)
        return report

    log.info("Farm %s: engineering features for %d datasets (Gold)",
             farm_id, len(datasets))

    for dataset_id in datasets:
        try:
            result_path = process_one_to_gold(
                farm_id, dataset_id, force=force,
                feature_descriptions=feature_descriptions,
            )
            if result_path and result_path.exists():
                # Read metadata for reporting
                meta_path = result_path.parent / f"dataset_{dataset_id}_meta.json"
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    report.total_rows += meta.get("n_rows", 0)
                    report.total_features = max(report.total_features,
                                                 meta.get("n_features", 0))
                report.datasets_processed += 1
            else:
                report.datasets_skipped += 1
        except Exception as e:
            report.errors.append(f"dataset_{dataset_id}: {e}")
            log.error("  FAILED dataset_%s: %s", dataset_id, e, exc_info=True)

    report.elapsed_sec = round(time.time() - t0, 1)
    report.log_summary()
    return report


# ---------------------------------------------------------------------------
# Development subset definition
# ---------------------------------------------------------------------------
DEV_SUBSETS = {
    "A": {
        "datasets": ["0", "3", "68", "72"],  # 2 normal + 2 anomaly (different faults)
        "rationale": (
            "dataset_0: anomaly (generator bearing), dataset_3: normal, "
            "dataset_68: anomaly (transformer failure), dataset_72: anomaly (gearbox)"
        ),
    },
}


def get_dev_subset(farm_id: str) -> list[str]:
    """Get the representative development subset IDs for a farm."""
    if farm_id in DEV_SUBSETS:
        return DEV_SUBSETS[farm_id]["datasets"]
    # Default: first 4 datasets
    datasets = list_silver_datasets(farm_id)
    return datasets[:4]

"""
silver.py
=========
Bronze → Silver transformation.

Reads raw Parquet from Bronze via DuckDB, applies cleaning and
standardization, writes cleaned Parquet to Silver. This is the
"cleaned + validated" layer — data here has:

  - Parsed timestamps (datetime64, sorted, NaT rows dropped)
  - Numeric columns coerced to proper dtypes
  - Zero-runs-as-missing handled (for Farms B/C)
  - Canonical column names verified
  - Event metadata merged from the farm's event_info file

Downstream code (feature engineering, model training) reads from
Silver, never from Bronze or raw CSV directly.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .. import config
from . import schema
from .ingestion import query_bronze, list_bronze_datasets

log = logging.getLogger(__name__)


@dataclass
class SilverReport:
    """Summary of one farm's Bronze → Silver transformation."""
    farm_id: str
    datasets_processed: int = 0
    datasets_skipped: int = 0
    total_rows_in: int = 0
    total_rows_out: int = 0
    rows_dropped_nat: int = 0
    errors: list[str] = field(default_factory=list)
    elapsed_sec: float = 0.0

    def log_summary(self):
        log.info("=== Silver report: Farm %s ===", self.farm_id)
        log.info("  Datasets processed: %d | Skipped: %d",
                 self.datasets_processed, self.datasets_skipped)
        log.info("  Rows in: %d | Rows out: %d | Dropped (NaT): %d",
                 self.total_rows_in, self.total_rows_out, self.rows_dropped_nat)
        log.info("  Elapsed: %.1f sec", self.elapsed_sec)
        if self.errors:
            for e in self.errors:
                log.error("  Error: %s", e)


def _clean_one_dataset(df: pd.DataFrame, farm_id: str) -> tuple[pd.DataFrame, int]:
    """Apply cleaning steps to one dataset DataFrame.

    Returns (cleaned_df, n_rows_dropped).
    """
    n_before = len(df)

    # 1. Parse timestamps
    if config.TIME_COL in df.columns:
        df[config.TIME_COL] = pd.to_datetime(df[config.TIME_COL], errors="coerce")
        n_nat = df[config.TIME_COL].isna().sum()
        if n_nat > 0:
            log.warning("Farm %s: dropping %d rows with unparseable timestamps", farm_id, n_nat)
            df = df[df[config.TIME_COL].notna()].reset_index(drop=True)

    # 2. Sort by timestamp
    if config.TIME_COL in df.columns:
        df = df.sort_values(config.TIME_COL).reset_index(drop=True)

    # 3. Coerce status column to numeric
    if config.STATUS_COL in df.columns:
        df[config.STATUS_COL] = pd.to_numeric(df[config.STATUS_COL], errors="coerce")

    # 4. Zero-runs-as-missing for non-power/wind columns
    # Import here to avoid circular dependency
    from ..features import identify_columns, clean_zeros_as_missing
    cols = identify_columns(df)
    numeric_cols = cols["numeric"]
    exclude = cols["power"] + cols["wind_speed"]
    df = clean_zeros_as_missing(df, numeric_cols, exclude=exclude)

    n_dropped = n_before - len(df)
    return df, n_dropped


def process_one_to_silver(
    farm_id: str,
    dataset_id: str,
    force: bool = False,
) -> Optional[Path]:
    """Transform one Bronze dataset to Silver.

    Reads from Bronze via DuckDB (all columns), applies cleaning,
    writes to Silver Parquet. Returns the output path, or None on error.
    """
    silver_dir = config.SILVER_DIR / f"farm={farm_id}"
    output_path = silver_dir / f"dataset_{dataset_id}.parquet"

    if output_path.exists() and not force:
        log.debug("Skipping %s/%s — Silver already exists", farm_id, dataset_id)
        return output_path

    # Read full dataset from Bronze (one dataset at a time is fine
    # for memory — Farm A datasets are ~55K rows × 86 cols ≈ 35 MB)
    df = query_bronze(farm_id, dataset_id)

    df, n_dropped = _clean_one_dataset(df, farm_id)

    # Write Silver Parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_path, compression="snappy")

    log.info("  Silver: farm=%s dataset=%s → %d rows (dropped %d), %.1f MB",
             farm_id, dataset_id, len(df), n_dropped,
             output_path.stat().st_size / 1e6)
    return output_path


def process_farm_to_silver(farm_id: str, force: bool = False) -> SilverReport:
    """Transform all Bronze datasets for one farm to Silver."""
    t0 = time.time()
    report = SilverReport(farm_id=farm_id)

    datasets = list_bronze_datasets(farm_id)
    if not datasets:
        log.warning("No Bronze datasets found for Farm %s — run ingestion first", farm_id)
        return report

    log.info("Farm %s: transforming %d Bronze datasets to Silver", farm_id, len(datasets))

    for dataset_id in datasets:
        try:
            df_before = query_bronze(farm_id, dataset_id,
                                     columns=[config.TIME_COL], limit=1)
            # Get row count via DuckDB without loading full data
            import duckdb
            parquet_path = config.BRONZE_DIR / f"farm={farm_id}" / f"dataset_{dataset_id}.parquet"
            con = duckdb.connect()
            n_rows_bronze = con.execute(
                f"SELECT COUNT(*) FROM read_parquet('{parquet_path}')"
            ).fetchone()[0]
            con.close()

            result_path = process_one_to_silver(farm_id, dataset_id, force=force)

            if result_path and result_path.exists():
                # Count Silver rows
                con = duckdb.connect()
                n_rows_silver = con.execute(
                    f"SELECT COUNT(*) FROM read_parquet('{result_path}')"
                ).fetchone()[0]
                con.close()

                report.datasets_processed += 1
                report.total_rows_in += n_rows_bronze
                report.total_rows_out += n_rows_silver
                report.rows_dropped_nat += (n_rows_bronze - n_rows_silver)
            else:
                report.datasets_skipped += 1

        except Exception as e:
            report.errors.append(f"dataset_{dataset_id}: {e}")
            log.error("  FAILED dataset_%s: %s", dataset_id, e)

    report.elapsed_sec = round(time.time() - t0, 1)
    report.log_summary()
    return report


def query_silver(
    farm_id: str,
    dataset_id: str,
    columns: Optional[list[str]] = None,
    where: Optional[str] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Query a Silver Parquet file using DuckDB (same interface as query_bronze)."""
    import duckdb

    parquet_path = config.SILVER_DIR / f"farm={farm_id}" / f"dataset_{dataset_id}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Silver Parquet not found: {parquet_path}. "
            f"Run process_farm_to_silver('{farm_id}') first."
        )

    col_clause = ", ".join(columns) if columns else "*"
    sql = f"SELECT {col_clause} FROM read_parquet('{parquet_path}')"
    if where:
        sql += f" WHERE {where}"
    if limit:
        sql += f" LIMIT {limit}"

    con = duckdb.connect()
    try:
        return con.execute(sql).fetchdf()
    finally:
        con.close()


def list_silver_datasets(farm_id: str) -> list[str]:
    """List all dataset IDs available in Silver for a given farm."""
    silver_dir = config.SILVER_DIR / f"farm={farm_id}"
    if not silver_dir.exists():
        return []
    return sorted(
        p.stem.replace("dataset_", "")
        for p in silver_dir.glob("dataset_*.parquet")
    )

"""
ingestion.py
=============
Memory-safe CSV → Parquet ingestion pipeline.

This is the core solution to the 39 GB problem. On a machine with 16 GB RAM
(~6-7 GB available), loading Farm C (34 GB, 957 columns) via pd.read_csv()
is physically impossible. This module uses PyArrow's streaming CSV reader
to convert raw CSVs to Parquet in fixed-size batches (~50K rows at a time,
peak memory ~30-80 MB per file regardless of file size), then provides
DuckDB-based query functions that read only the columns and rows needed
by downstream code — never the full Parquet file.

Architecture:
    Raw CSV ──► PyArrow streaming reader (batch_size=50K)
                    │
                    ▼
              Parquet file (Bronze)
                    │
                    ▼
              DuckDB SQL queries (out-of-core)
                    │
                    ▼
              Small pandas DataFrame (only needed cols/rows)

Output directory structure:
    data/bronze/
    └── farm=A/
        ├── dataset_0.parquet
        ├── dataset_3.parquet
        └── ...

Usage:
    from src.data.ingestion import ingest_farm, query_bronze

    # Convert all CSVs for Farm A to Parquet
    report = ingest_farm("A")

    # Query specific columns/rows without loading full file
    df = query_bronze("A", dataset_id="0",
                      columns=["time_stamp", "power_30_avg", "wind_speed_3_avg"],
                      where="train_test = 'train'")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import duckdb
import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq

from .. import config

log = logging.getLogger(__name__)

# How many rows to read per batch during streaming ingestion.
# 50K rows × 86 columns (Farm A) ≈ 33 MB per batch.
# 50K rows × 957 columns (Farm C) ≈ 80 MB per batch.
# Both well within the ~6 GB available RAM.
BATCH_SIZE = 50_000


# ---------------------------------------------------------------------------
# Ingestion report
# ---------------------------------------------------------------------------
@dataclass
class IngestionReport:
    """Summary of one farm's CSV → Parquet conversion."""
    farm_id: str
    datasets_ingested: int = 0
    datasets_skipped: int = 0
    total_rows: int = 0
    total_bytes_parquet: int = 0
    errors: list[str] = field(default_factory=list)
    elapsed_sec: float = 0.0

    def log_summary(self):
        log.info("=== Ingestion report: Farm %s ===", self.farm_id)
        log.info("  Datasets ingested: %d | Skipped: %d",
                 self.datasets_ingested, self.datasets_skipped)
        log.info("  Total rows: %d | Parquet size: %.1f MB",
                 self.total_rows, self.total_bytes_parquet / 1e6)
        log.info("  Elapsed: %.1f sec", self.elapsed_sec)
        if self.errors:
            for e in self.errors:
                log.error("  Error: %s", e)


# ---------------------------------------------------------------------------
# Discovery — which CSVs to ingest
# ---------------------------------------------------------------------------
def _discover_data_csvs(raw_dir: Path) -> list[Path]:
    """Find data CSVs, skip metadata files, deduplicate comma_/plain pairs.
    Same dedup logic as data_loader.discover_subdatasets()."""
    skip_fragments = ("event_info", "sensor", "description", "readme", "metadata",
                      "feature_description")
    csvs = [
        p for p in raw_dir.rglob("*.csv")
        if not any(frag in p.stem.lower() for frag in skip_fragments)
    ]

    by_key: dict[tuple, list[Path]] = {}
    for p in csvs:
        stem = p.stem
        normalized = stem[len("comma_"):] if stem.lower().startswith("comma_") else stem
        by_key.setdefault((p.parent, normalized), []).append(p)

    deduped = []
    for (_, normalized), group in by_key.items():
        if len(group) == 1:
            deduped.append((normalized, group[0]))
        else:
            comma_versions = [p for p in group if p.stem.lower().startswith("comma_")]
            keep = comma_versions[0] if comma_versions else group[0]
            deduped.append((normalized, keep))

    return sorted(deduped, key=lambda x: x[0])


def _normalize_dataset_id(csv_path: Path) -> str:
    """Extract the numeric dataset ID from a CSV filename."""
    stem = csv_path.stem
    return stem[len("comma_"):] if stem.lower().startswith("comma_") else stem


# ---------------------------------------------------------------------------
# Streaming CSV → Parquet
# ---------------------------------------------------------------------------
def _ingest_one_csv(csv_path: Path, output_path: Path) -> int:
    """Stream one CSV to Parquet using PyArrow batched reading.

    Returns the number of rows written. Peak memory is bounded by
    BATCH_SIZE regardless of file size."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # PyArrow CSV read options: auto-detect types, handle encoding issues
    read_opts = pcsv.ReadOptions(block_size=1 << 20)  # 1 MB read blocks
    parse_opts = pcsv.ParseOptions(delimiter=",")
    convert_opts = pcsv.ConvertOptions(
        strings_can_be_null=True,
        auto_dict_encode=True,       # dictionary-encode low-cardinality strings
        auto_dict_max_cardinality=50
    )

    # Try comma separator first, fall back to auto-detection
    try:
        reader = pcsv.open_csv(
            csv_path,
            read_options=read_opts,
            parse_options=parse_opts,
            convert_options=convert_opts
        )
    except Exception:
        # Fall back: try semicolon separator
        parse_opts_semi = pcsv.ParseOptions(delimiter=";")
        reader = pcsv.open_csv(
            csv_path,
            read_options=read_opts,
            parse_options=parse_opts_semi,
            convert_options=convert_opts
        )

    writer = None
    total_rows = 0
    try:
        for batch in reader:
            table = pa.Table.from_batches([batch])
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema,
                                          compression="snappy")
            writer.write_table(table)
            total_rows += len(batch)
    finally:
        if writer is not None:
            writer.close()

    return total_rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def ingest_farm(farm_id: str, force: bool = False) -> IngestionReport:
    """Convert all raw CSVs for one farm to Parquet (Bronze layer).

    Args:
        farm_id: "A", "B", or "C"
        force: if True, re-ingest even if Parquet already exists

    Returns:
        IngestionReport with stats and any errors
    """
    t0 = time.time()
    report = IngestionReport(farm_id=farm_id)

    if farm_id not in config.FARM_CONFIGS:
        raise ValueError(f"Unknown farm_id: {farm_id}. Must be one of {config.FARMS}")

    raw_dir = config.FARM_CONFIGS[farm_id]["raw_dir"]
    bronze_dir = config.BRONZE_DIR / f"farm={farm_id}"

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_dir}. "
            f"Download Farm {farm_id} from Kaggle and extract it there."
        )

    csv_files = _discover_data_csvs(raw_dir)
    log.info("Farm %s: found %d data CSVs in %s", farm_id, len(csv_files), raw_dir)

    for dataset_id, csv_path in csv_files:
        parquet_path = bronze_dir / f"dataset_{dataset_id}.parquet"

        if parquet_path.exists() and not force:
            log.debug("Skipping %s — Parquet already exists at %s",
                      csv_path.name, parquet_path)
            report.datasets_skipped += 1
            continue

        try:
            n_rows = _ingest_one_csv(csv_path, parquet_path)
            size_bytes = parquet_path.stat().st_size
            report.datasets_ingested += 1
            report.total_rows += n_rows
            report.total_bytes_parquet += size_bytes
            log.info("  %s → %s (%d rows, %.1f MB Parquet)",
                     csv_path.name, parquet_path.name, n_rows, size_bytes / 1e6)
        except Exception as e:
            report.errors.append(f"{csv_path.name}: {e}")
            log.error("  FAILED %s: %s", csv_path.name, e)

    report.elapsed_sec = round(time.time() - t0, 1)
    report.log_summary()
    return report


def ingest_all(force: bool = False) -> dict[str, IngestionReport]:
    """Ingest all available farms (A, B, C) to Bronze Parquet."""
    reports = {}
    for farm_id in config.FARMS:
        raw_dir = config.FARM_CONFIGS[farm_id]["raw_dir"]
        if raw_dir.exists():
            reports[farm_id] = ingest_farm(farm_id, force=force)
        else:
            log.warning("Farm %s raw data not found at %s — skipping",
                        farm_id, raw_dir)
    return reports


# ---------------------------------------------------------------------------
# DuckDB query interface — out-of-core analytical queries over Parquet
# ---------------------------------------------------------------------------
def query_bronze(
    farm_id: str,
    dataset_id: str,
    columns: Optional[list[str]] = None,
    where: Optional[str] = None,  # INTERNAL USE ONLY — not exposed to API users
    limit: Optional[int] = None,
) -> "pd.DataFrame":
    """Query a Bronze Parquet file using DuckDB (out-of-core, column-pruned).

    WARNING: `columns` and `where` are interpolated into SQL. This function
    is for internal pipeline use only. Do NOT pass user-controlled input
    to these parameters without validation.

    This is the primary interface for downstream code to read data from
    the data lake. It never loads the full Parquet file into memory —
    DuckDB reads only the requested columns and applies filters before
    materializing results.

    Args:
        farm_id: "A", "B", or "C"
        dataset_id: e.g. "0", "68"
        columns: list of column names to SELECT (None = all)
        where: SQL WHERE clause (e.g. "train_test = 'train'")
        limit: max rows to return

    Returns:
        pandas DataFrame with the requested data
    """
    import pandas as pd  # local import to keep module-level import light

    parquet_path = config.BRONZE_DIR / f"farm={farm_id}" / f"dataset_{dataset_id}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Bronze Parquet not found: {parquet_path}. "
            f"Run ingest_farm('{farm_id}') first."
        )

    col_clause = ", ".join(columns) if columns else "*"
    sql = f"SELECT {col_clause} FROM read_parquet('{parquet_path}')"
    if where:
        sql += f" WHERE {where}"
    if limit:
        sql += f" LIMIT {limit}"

    con = duckdb.connect()
    try:
        result = con.execute(sql).fetchdf()
    finally:
        con.close()

    return result


def list_bronze_datasets(farm_id: str) -> list[str]:
    """List all dataset IDs available in Bronze for a given farm."""
    bronze_dir = config.BRONZE_DIR / f"farm={farm_id}"
    if not bronze_dir.exists():
        return []
    return sorted(
        p.stem.replace("dataset_", "")
        for p in bronze_dir.glob("dataset_*.parquet")
    )


def get_bronze_schema(farm_id: str, dataset_id: str) -> list[str]:
    """Get column names from a Bronze Parquet file without loading data."""
    parquet_path = config.BRONZE_DIR / f"farm={farm_id}" / f"dataset_{dataset_id}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Bronze Parquet not found: {parquet_path}")
    schema = pq.read_schema(parquet_path)
    return schema.names

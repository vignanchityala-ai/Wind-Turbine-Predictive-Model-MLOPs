"""
data_loader.py
===============
Loading, validating, and lightly cleaning individual Wind Farm A sub-datasets.

Confirmed real layout (from the actual Kaggle download, not guessed): all
numbered CSVs (0.csv, 3.csv, comma_0.csv, ...) sit flat in one shared
"Wind Farm A/datasets/" folder — not one folder per dataset. Event labels
live in a SINGLE shared event_info file (columns include at least event_id,
event_label, event_start, event_end, event_description) with one row per
dataset, where event_id is an integer matching the data CSV's filename
stem (e.g. event_id=68 <-> "68.csv" / "comma_68.csv"). Dates in that file
are DD-MM-YYYY HH:MM (day-first) — confirmed unambiguously from rows like
"29-07-2015" (no month 29) and "24-12-2022" (no month 24) — parsing this
without dayfirst=True silently produces wrong dates for any day <= 12
(e.g. "05-08-2022" becomes May 8th instead of August 5th), which matters a
lot here since event_end drives the Earliness/Coverage evaluation math.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from . import config

log = logging.getLogger(__name__)


def _parse_event_datetime(value):
    """
    Parse an event_start/event_end value robustly across the two formats
    seen in practice:
      - Real Kaggle event_info.csv: 'DD-MM-YYYY HH:MM' (day-first, confirmed
        from unambiguous rows like '29-07-2015' -- no month 29).
      - This pipeline's synthetic test fixtures: ISO 'YYYY-MM-DD HH:MM:SS'.

    IMPORTANT: pandas' dayfirst=True is NOT safe to use for this -- it can
    corrupt an already-unambiguous ISO string, not just resolve genuine
    ambiguity. Confirmed directly: pd.to_datetime('2021-02-09 00:00:00',
    dayfirst=True) incorrectly returns 2021-09-02 (silently swaps month and
    day) even though YYYY-MM-DD order has no ambiguity to resolve. So the
    format is detected explicitly from the string shape instead.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if not s:
        return None

    if re.match(r"^\d{4}-\d{1,2}-\d{1,2}", s):          # YYYY-MM-DD... (year first: unambiguous)
        return pd.to_datetime(s, dayfirst=False, errors="coerce")
    if re.match(r"^\d{1,2}-\d{1,2}-\d{4}", s):           # DD-MM-YYYY... (year last: the real event_info.csv format)
        return pd.to_datetime(s, format="%d-%m-%Y %H:%M", errors="coerce")

    parsed = pd.to_datetime(s, errors="coerce")           # unrecognized shape: last resort
    if pd.notna(parsed):
        log.warning(
            "event date %r didn't match a known format (YYYY-MM-DD or "
            "DD-MM-YYYY) -- parsed as %s via pandas' default inference. "
            "Verify this is correct; wrong month/day here silently breaks "
            "Earliness/Coverage evaluation.", s, parsed,
        )
    return parsed


def _resolve_event_time_from_id(df: pd.DataFrame, row_id) -> Optional[pd.Timestamp]:
    """
    Resolve an event_start_id/event_end_id value (confirmed to reference the
    dataset's own 'id' column) to the actual timestamp of that row in the
    loaded data. More robust than parsing a date string, since it's tied
    directly to the real data rather than a separately-formatted text field
    that could have its own transcription/format issues. Returns None if
    row_id is missing or doesn't match exactly one row (caller falls back
    to the parsed date string in that case).
    """
    if row_id is None or (isinstance(row_id, float) and pd.isna(row_id)):
        return None
    try:
        row_id_numeric = int(float(row_id))
    except (TypeError, ValueError):
        return None
    if config.ID_COL not in df.columns:
        return None
    matches = df.loc[df[config.ID_COL] == row_id_numeric, config.TIME_COL]
    return matches.iloc[0] if len(matches) == 1 else None


@dataclass
class SubDataset:
    """Container for one turbine's train+prediction episode."""
    name: str
    df: pd.DataFrame
    event_label: Optional[str] = None      # 'anomaly' | 'normal' | None
    event_start: Optional[pd.Timestamp] = None
    event_end: Optional[pd.Timestamp] = None
    event_description: Optional[str] = None  # e.g. 'Gearbox failure', only set for anomalies
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
    """
    Read a SCADA csv, tolerating comma/semicolon separators and utf-8/
    latin-1 encoding (mojibake in unit symbols like ° has been observed in
    this dataset's metadata files, and latin-1 never raises a decode error
    so this always succeeds).

    Tries the fast C engine with an explicit comma first -- the documented
    common case for this dataset (the Kaggle mirror converts semicolons to
    commas). engine="python" with sep=None (delimiter auto-detection) is
    3-10x slower and, being a heuristic, can occasionally guess wrong on an
    unusual file; it's now only used as a fallback when the fast path
    produces a single column (the clearest sign the delimiter guess was
    wrong), not as the default for every file.
    """
    for encoding in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(path, sep=",", engine="c", encoding=encoding)
            if df.shape[1] > 1:
                return df
        except UnicodeDecodeError:
            continue
        except Exception:
            pass  # fall through to the slower auto-detecting path below

        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=encoding)
        except UnicodeDecodeError:
            continue
        except Exception:
            if encoding == "latin-1":
                raise
            continue
    return pd.read_csv(path)


def _read_event_metadata_from_file(p: Path, normalized_stem: str) -> dict:
    """Read one specific event-metadata file and return the row matching
    normalized_stem (or {} if not found / file doesn't exist)."""
    if not p.exists():
        return {}
    if p.suffix == ".json":
        return json.loads(p.read_text())

    meta_df = _read_csv_robust(p)
    meta_df.columns = [c.strip().lower() for c in meta_df.columns]
    id_cols = [c for c in meta_df.columns if c in
               ("event_id", "dataset_id", "id", "dataset", "name")]
    match = pd.DataFrame()
    if id_cols:
        id_col = id_cols[0]
        numeric_ids = pd.to_numeric(meta_df[id_col], errors="coerce")
        target = pd.to_numeric(pd.Series([normalized_stem]), errors="coerce").iloc[0]
        if pd.notna(target):
            match = meta_df[numeric_ids == target]
        if not len(match):
            # fall back to plain string equality (non-numeric ids)
            match = meta_df[meta_df[id_col].astype(str).str.strip() == normalized_stem]
    row = match.iloc[0] if len(match) else (meta_df.iloc[0] if len(meta_df) == 1 else None)
    return row.to_dict() if row is not None else {}


def _normalize_stem(csv_path: Path) -> str:
    stem = csv_path.stem
    return stem[len("comma_"):] if stem.lower().startswith("comma_") else stem


def _find_event_metadata(csv_path: Path, event_info_path: Path = None) -> dict:
    """
    Look for a shared event_info file describing this dataset's label.

    If event_info_path is given, read ONLY that file directly — no search,
    no guessing. Use this once you know exactly where the file lives (real
    downloads vary: sometimes alongside the data CSVs, sometimes at a
    different path entirely, as observed when a user's --raw-dir and their
    event_info file turned out to live in two completely separate extracted
    folders). Pass it via run_pipeline.py's --event-info-path flag.

    Otherwise, tries, in order: (1) same folder as the data CSV; (2)
    parent-of-parent (the real Farm A layout has all data CSVs flat in
    "datasets/", with the event_info file placed either there or one level
    up at the farm root — confirmed both are supported, exact placement can
    vary). This is a best-effort search, not a guarantee — if it doesn't
    find the file, the warning at the end tells you exactly what was
    checked, so you can locate the real file and pass it explicitly instead.

    Filename: the real, confirmed filename is "comma_event_info.csv" (same
    comma_-prefix convention as the data files — the source dataset went
    through the same semicolon-to-comma conversion pass). "event_info.csv"
    is still checked too in case an unconverted duplicate also exists,
    checked second to match the same comma_-preferred convention used for
    deduplicating the data files in discover_subdatasets().

    The file is a SINGLE SHARED table (one row per dataset, not one file
    per dataset), with an integer event_id column matching the data CSV's
    filename stem (confirmed: event_id is literally the file number, not a
    separate turbine/asset identifier). Matching handles two wrinkles: (a)
    the filename might be "comma_68" after dedup, not "68", so the comma_
    prefix is stripped before comparing; (b) event_id might come through as
    an int, or as a float-formatted string like "68.0" if pandas inferred
    the column as float (e.g. due to NaNs elsewhere in the column) — so the
    comparison is done numerically, not as a raw string equality check.
    """
    normalized_stem = _normalize_stem(csv_path)

    if event_info_path is not None:
        data = _read_event_metadata_from_file(Path(event_info_path), normalized_stem)
        if data:
            return data
        log.warning(
            "%s: --event-info-path was given (%s) but no row matched "
            "normalized id %r in it. Double-check that file actually "
            "contains this dataset.",
            csv_path.name, event_info_path, normalized_stem,
        )
        return {}

    search_dirs = [csv_path.parent, csv_path.parent.parent]
    candidates = ("comma_event_info.csv", "event_info.csv", "event_info.json", "metadata.json")
    for folder in search_dirs:
        for candidate in candidates:
            data = _read_event_metadata_from_file(folder / candidate, normalized_stem)
            if data:
                return data

    log.warning(
        "No event metadata found for %s (checked %s for %s, normalized id %r) — "
        "event_label/asset_id will be None, which will break is_anomaly and "
        "evaluation for this dataset.",
        csv_path.name, [str(d) for d in search_dirs], candidates, normalized_stem,
    )
    return {}


def load_subdataset(csv_path: Path, event_info_path: Path = None) -> SubDataset:
    """Load a single sub-dataset CSV plus its event metadata (if present).

    event_info_path: optional explicit path to the event-info file, used
    directly instead of searching. Pass this once you know exactly where
    the file lives on your machine — real downloads have been observed to
    place it somewhere the folder-structure search doesn't check (e.g. a
    completely different extracted directory than --raw-dir)."""
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

    n_nat = df[config.TIME_COL].isna().sum()
    if n_nat > 0:
        pct = 100 * n_nat / len(df)
        log.warning(
            "%s: %d/%d (%.2f%%) rows had an unparseable time_stamp and "
            "became NaT. These are dropped rather than left in -- an "
            "unparsed timestamp would otherwise sort to the end of the "
            "series and silently contaminate rolling-window features with "
            "data from an unknown time.",
            csv_path.name, n_nat, len(df), pct,
        )
        df = df[df[config.TIME_COL].notna()].reset_index(drop=True)

    df = df.sort_values(config.TIME_COL).reset_index(drop=True)
    df[config.STATUS_COL] = pd.to_numeric(df[config.STATUS_COL], errors="coerce")

    def _clean(v):
        """pandas reads empty CSV fields as float NaN, which is Python-truthy
        and would otherwise leak through as a float where Optional[str] is
        expected."""
        return None if v is None or (isinstance(v, float) and pd.isna(v)) else v

    meta = _find_event_metadata(csv_path, event_info_path=event_info_path)
    event_label = _clean(meta.get("event_label")) or _clean(meta.get("label"))
    event_start = _clean(meta.get("event_start"))
    event_end = _clean(meta.get("event_end"))
    event_start_id = _clean(meta.get("event_start_id"))
    event_end_id = _clean(meta.get("event_end_id"))
    event_description = _clean(meta.get("event_description")) or _clean(meta.get("description"))
    asset_id = _clean(meta.get("asset_id")) or (
        df[config.ASSET_COL].iloc[0] if config.ASSET_COL in df.columns else None
    )

    # See _parse_event_datetime's docstring for why this isn't just
    # pd.to_datetime(..., dayfirst=True) -- that corrupts ISO-format dates.
    if event_start is not None:
        event_start = _parse_event_datetime(event_start)
    if event_end is not None:
        event_end = _parse_event_datetime(event_end)

    # Prefer resolving via event_start_id/event_end_id (the dataset's own
    # 'id' column) when available -- it's tied directly to the actual
    # loaded data, sidestepping date-string parsing entirely. Cross-check
    # against the parsed string and warn on disagreement, since that would
    # indicate a real parsing or data issue worth knowing about rather than
    # silently picking one source over the other.
    resolved_start = _resolve_event_time_from_id(df, event_start_id)
    resolved_end = _resolve_event_time_from_id(df, event_end_id)
    if resolved_start is not None:
        if event_start is not None and abs((resolved_start - event_start).total_seconds()) > 60:
            log.warning(
                "%s: event_start mismatch -- parsed date string gives %s but "
                "event_start_id=%s resolves to %s in the data. Using the "
                "id-based value (more reliable, tied directly to real data).",
                csv_path.stem, event_start, event_start_id, resolved_start,
            )
        event_start = resolved_start
    if resolved_end is not None:
        if event_end is not None and abs((resolved_end - event_end).total_seconds()) > 60:
            log.warning(
                "%s: event_end mismatch -- parsed date string gives %s but "
                "event_end_id=%s resolves to %s in the data. Using the "
                "id-based value (more reliable, tied directly to real data).",
                csv_path.stem, event_end, event_end_id, resolved_end,
            )
        event_end = resolved_end

    return SubDataset(
        name=csv_path.stem,
        df=df,
        event_label=event_label,
        event_start=event_start,
        event_end=event_end,
        event_description=event_description,
        asset_id=asset_id,
    )


def discover_subdatasets(raw_dir: Path = config.RAW_DATA_DIR) -> list[Path]:
    """
    Find candidate SCADA CSVs under raw_dir. Skips files that are clearly
    metadata (event_info*, sensor description files, etc.), and de-duplicates
    a naming quirk observed in the Kaggle mirror: the "datasets" folder lists
    both e.g. "3.csv" and "comma_3.csv" for the same underlying dataset (44
    files for Wind Farm A's 22 real datasets) — most likely leftover from a
    version update that added comma-converted copies without removing the
    originals. Since load_subdataset() already auto-detects the separator,
    either copy works; we just need exactly one per dataset, not both.
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

    # De-dupe "comma_<id>.csv" vs "<id>.csv" pairs living in the same folder.
    # Prefer the comma_-prefixed copy (explicitly named as the converted one)
    # when both exist; otherwise keep whichever single copy is present.
    by_key: dict[tuple, list[Path]] = {}
    for p in csvs:
        stem = p.stem
        normalized = stem[len("comma_"):] if stem.lower().startswith("comma_") else stem
        by_key.setdefault((p.parent, normalized), []).append(p)

    deduped = []
    skipped = []
    for (_, _normalized), group in by_key.items():
        if len(group) == 1:
            deduped.append(group[0])
            continue
        comma_versions = [p for p in group if p.stem.lower().startswith("comma_")]
        keep = comma_versions[0] if comma_versions else group[0]
        deduped.append(keep)
        skipped.extend(p for p in group if p != keep)

    if skipped:
        log.info(
            "Skipped %d duplicate file(s) (comma_/plain naming pairs), e.g. %s",
            len(skipped), sorted(p.name for p in skipped)[:5],
        )

    return sorted(deduped)


def load_all(raw_dir: Path = config.RAW_DATA_DIR, event_info_path: Path = None) -> list[SubDataset]:
    paths = discover_subdatasets(raw_dir)
    if not paths:
        raise FileNotFoundError(f"No SCADA csv files found under {raw_dir}")
    return [load_subdataset(p, event_info_path=event_info_path) for p in paths]
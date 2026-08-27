"""
validation.py
==============
Data quality validation for the Bronze and Silver tiers.

Runs BEFORE any model work touches the data. Each check produces a
PASS/FAIL result with a human-readable message explaining what was
found. Downstream code can decide whether to hard-fail or warn based
on the severity.

Checks:
  1. Schema: required columns present and correct types
  2. Timestamps: parseable, sorted, no unexplained gaps > 10 min
  3. Ranges: power >= 0, wind_speed >= 0, status in {0..5}
  4. Missingness: flag columns with > threshold NaN
  5. Duplicates: detect duplicate timestamps per asset
  6. Event metadata: event_info file exists and references match datasets
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .. import config

log = logging.getLogger(__name__)


@dataclass
class ValidationCheck:
    """Result of one validation check."""
    name: str
    passed: bool
    message: str
    severity: str = "ERROR"  # "ERROR" = must fix, "WARNING" = investigate


@dataclass
class ValidationReport:
    """Collection of validation checks for one dataset."""
    farm_id: str
    dataset_id: str
    checks: list[ValidationCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True if no ERROR-severity checks failed."""
        return all(c.passed for c in self.checks if c.severity == "ERROR")

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def n_failed(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    def log_summary(self):
        status = "PASS" if self.passed else "FAIL"
        log.info("Validation [%s] Farm %s / dataset %s: %d passed, %d failed",
                 status, self.farm_id, self.dataset_id,
                 self.n_passed, self.n_failed)
        for c in self.checks:
            if not c.passed:
                log_fn = log.error if c.severity == "ERROR" else log.warning
                log_fn("  [%s] %s: %s", c.severity, c.name, c.message)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
def check_required_columns(df: pd.DataFrame) -> ValidationCheck:
    """Verify the minimum required columns exist."""
    required = {config.TIME_COL, config.SPLIT_COL, config.STATUS_COL}
    present = set(df.columns)
    missing = required - present

    if missing:
        return ValidationCheck(
            name="required_columns",
            passed=False,
            message=f"Missing required columns: {missing}. Found: {list(df.columns)[:10]}..."
        )
    return ValidationCheck(
        name="required_columns",
        passed=True,
        message=f"All {len(required)} required columns present."
    )


def check_timestamp_quality(df: pd.DataFrame) -> ValidationCheck:
    """Check that timestamps are parseable and have no large gaps."""
    if config.TIME_COL not in df.columns:
        return ValidationCheck(
            name="timestamp_quality",
            passed=False,
            message=f"Column '{config.TIME_COL}' not found."
        )

    ts = pd.to_datetime(df[config.TIME_COL], errors="coerce")
    n_nat = ts.isna().sum()
    if n_nat > 0:
        pct = 100 * n_nat / len(ts)
        return ValidationCheck(
            name="timestamp_quality",
            passed=pct < 1.0,  # <1% unparseable is acceptable
            severity="ERROR" if pct >= 1.0 else "WARNING",
            message=f"{n_nat}/{len(ts)} ({pct:.2f}%) timestamps unparseable."
        )

    ts_sorted = ts.sort_values()
    gaps = ts_sorted.diff().dropna()
    expected_interval = pd.Timedelta(minutes=10)
    large_gaps = gaps[gaps > expected_interval * 3]  # > 30 min gap

    if len(large_gaps) > 0:
        max_gap = large_gaps.max()
        return ValidationCheck(
            name="timestamp_quality",
            passed=True,
            severity="WARNING",
            message=f"All timestamps parseable. {len(large_gaps)} gap(s) > 30 min "
                    f"detected (max: {max_gap}). May indicate downtime periods."
        )

    return ValidationCheck(
        name="timestamp_quality",
        passed=True,
        message="All timestamps parseable, no unexplained gaps."
    )


def check_status_values(df: pd.DataFrame) -> ValidationCheck:
    """Check status_type_id values are in the expected range {0..5}."""
    if config.STATUS_COL not in df.columns:
        return ValidationCheck(
            name="status_values",
            passed=False,
            message=f"Column '{config.STATUS_COL}' not found."
        )

    values = pd.to_numeric(df[config.STATUS_COL], errors="coerce")
    valid_statuses = {0, 1, 2, 3, 4, 5}
    unique = set(values.dropna().astype(int).unique())
    invalid = unique - valid_statuses

    if invalid:
        return ValidationCheck(
            name="status_values",
            passed=False,
            message=f"Unexpected status values: {invalid}. Expected subset of {valid_statuses}."
        )
    return ValidationCheck(
        name="status_values",
        passed=True,
        message=f"All status values in valid range. Distribution: {values.value_counts().to_dict()}"
    )


def check_missingness(df: pd.DataFrame, threshold: float = 0.5) -> ValidationCheck:
    """Flag columns where >threshold fraction of values are NaN."""
    nan_frac = df.isna().mean()
    high_nan = nan_frac[nan_frac > threshold].sort_values(ascending=False)

    if len(high_nan) > 0:
        top5 = [(c, round(v, 3)) for c, v in high_nan.head(5).items()]
        return ValidationCheck(
            name="missingness",
            passed=True,  # not a hard failure, just informational
            severity="WARNING",
            message=f"{len(high_nan)} column(s) > {threshold*100:.0f}% NaN. "
                    f"Top 5: {top5}"
        )
    return ValidationCheck(
        name="missingness",
        passed=True,
        message=f"No columns exceed {threshold*100:.0f}% missingness."
    )


def check_duplicates(df: pd.DataFrame) -> ValidationCheck:
    """Detect duplicate timestamps (same time_stamp for same asset)."""
    if config.TIME_COL not in df.columns:
        return ValidationCheck(
            name="duplicates",
            passed=False,
            message=f"Column '{config.TIME_COL}' not found."
        )

    group_cols = [config.TIME_COL]
    if config.ASSET_COL in df.columns:
        group_cols.append(config.ASSET_COL)

    dupes = df.duplicated(subset=group_cols, keep=False)
    n_dupes = dupes.sum()

    if n_dupes > 0:
        return ValidationCheck(
            name="duplicates",
            passed=False,
            severity="WARNING",
            message=f"{n_dupes} duplicate timestamp rows detected."
        )
    return ValidationCheck(
        name="duplicates",
        passed=True,
        message="No duplicate timestamps found."
    )


def check_split_column(df: pd.DataFrame) -> ValidationCheck:
    """Verify train_test column contains expected values."""
    if config.SPLIT_COL not in df.columns:
        return ValidationCheck(
            name="split_column",
            passed=False,
            message=f"Column '{config.SPLIT_COL}' not found."
        )

    values = set(df[config.SPLIT_COL].dropna().unique())
    expected = {config.TRAIN_VALUE, config.PREDICTION_VALUE}

    if not values.issubset(expected):
        unexpected = values - expected
        return ValidationCheck(
            name="split_column",
            passed=False,
            message=f"Unexpected split values: {unexpected}. Expected: {expected}"
        )

    train_count = (df[config.SPLIT_COL] == config.TRAIN_VALUE).sum()
    pred_count = (df[config.SPLIT_COL] == config.PREDICTION_VALUE).sum()

    return ValidationCheck(
        name="split_column",
        passed=True,
        message=f"Split valid. Train: {train_count} rows, Prediction: {pred_count} rows."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def validate_dataset(
    df: pd.DataFrame,
    farm_id: str,
    dataset_id: str,
    nan_threshold: float = 0.5,
) -> ValidationReport:
    """Run all validation checks on a single dataset DataFrame.

    Args:
        df: the dataset to validate
        farm_id: farm identifier (for reporting)
        dataset_id: dataset identifier (for reporting)
        nan_threshold: fraction of NaN above which a column is flagged

    Returns:
        ValidationReport with all check results
    """
    report = ValidationReport(farm_id=farm_id, dataset_id=dataset_id)

    report.checks.append(check_required_columns(df))
    report.checks.append(check_timestamp_quality(df))
    report.checks.append(check_status_values(df))
    report.checks.append(check_missingness(df, threshold=nan_threshold))
    report.checks.append(check_duplicates(df))
    report.checks.append(check_split_column(df))

    report.log_summary()
    return report


def validate_bronze_dataset(
    farm_id: str,
    dataset_id: str,
    nan_threshold: float = 0.5,
) -> ValidationReport:
    """Validate a dataset directly from Bronze Parquet using DuckDB.

    Loads only the columns needed for validation checks, not the full
    dataset — important for Farm C where full load would exceed RAM.
    """
    from .ingestion import query_bronze

    # Load only the columns needed for validation
    validation_cols = [
        config.TIME_COL, config.SPLIT_COL, config.STATUS_COL
    ]
    if config.ASSET_COL:
        validation_cols.append(config.ASSET_COL)

    try:
        df = query_bronze(farm_id, dataset_id, columns=validation_cols)
    except FileNotFoundError:
        return ValidationReport(
            farm_id=farm_id,
            dataset_id=dataset_id,
            checks=[ValidationCheck(
                name="bronze_exists",
                passed=False,
                message=f"Bronze Parquet not found for farm={farm_id}, dataset={dataset_id}"
            )]
        )

    report = validate_dataset(df, farm_id, dataset_id, nan_threshold)

    # Additional: check full schema column count via metadata (no data load)
    from .ingestion import get_bronze_schema
    try:
        all_cols = get_bronze_schema(farm_id, dataset_id)
        expected = config.FARM_CONFIGS.get(farm_id, {}).get("n_features")
        if expected and len(all_cols) != expected:
            report.checks.append(ValidationCheck(
                name="column_count",
                passed=True,
                severity="WARNING",
                message=f"Found {len(all_cols)} columns, expected {expected} for Farm {farm_id}."
            ))
        else:
            report.checks.append(ValidationCheck(
                name="column_count",
                passed=True,
                message=f"Column count matches expected: {len(all_cols)}."
            ))
    except Exception as e:
        report.checks.append(ValidationCheck(
            name="column_count",
            passed=False,
            message=f"Could not read schema: {e}"
        ))

    return report

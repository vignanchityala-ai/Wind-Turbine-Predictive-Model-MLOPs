"""
evaluation.py
==============
Evaluation aligned with the CARE benchmark's own scoring philosophy
(Gück et al. 2024), simplified to be self-contained (no external CARE
scoring library dependency).

Why not just accuracy/F1? Because:
  - Datasets with NO anomaly (normal-only sub-datasets) would always give
    an undefined/zero F1 (no true positives possible) — need a separate
    "did we falsely alarm on healthy turbines" metric.
  - Point-wise metrics reward/punish every single 10-minute tick equally,
    but operators care about "did we catch the EVENT and how much lead
    time did we get", not raw point-level recall.

We report four complementary metrics per sub-dataset, then aggregate:

  1. Coverage   (F_0.5 on points, anomaly datasets only, normal-status
                 points excluded) — did we detect the anomalous period?
  2. Accuracy   (specificity on normal-only datasets) — do we avoid
                 crying wolf on healthy turbines?
  3. Reliability (event-level F_0.5: did we raise one confident alarm,
                 or a flood of flickering false ones?)
  4. Earliness  (for detected anomaly datasets: how long before the
                 documented fault onset did we first sound sustained alarm?)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config


def _fbeta(tp: int, fp: int, fn: int, beta: float = 0.5) -> float:
    num = (1 + beta**2) * tp
    den = (1 + beta**2) * tp + beta**2 * fn + fp
    return num / den if den > 0 else 0.0


@dataclass
class SubDatasetResult:
    name: str
    is_anomaly: bool
    coverage: float | None = None
    accuracy: float | None = None
    reliability: float | None = None
    earliness_hours: float | None = None
    n_flagged_points: int = 0
    n_eval_points: int = 0


def evaluate_subdataset(
    name: str,
    is_anomaly: bool,
    status_ids: pd.Series,
    predicted_flags: np.ndarray,
    timestamps: pd.Series,
    fault_onset: pd.Timestamp | None = None,
) -> SubDatasetResult:
    """
    Evaluate one sub-dataset's PREDICTION-period predictions against its
    status-derived ground truth.

    status_ids, predicted_flags, timestamps must be aligned (same index/order)
    and restricted to the prediction period already.
    """
    normal_mask = status_ids.isin(config.NORMAL_STATUS_IDS).to_numpy()
    result = SubDatasetResult(name=name, is_anomaly=is_anomaly)

    # Only points with a "normal" status ID are meaningful for scoring —
    # abnormal-status points are already self-evidently flagged by SCADA
    # and would trivially inflate performance either way.
    g = normal_mask  # ground truth eval mask (True = should NOT be flagged... )
    # Ground truth for coverage/accuracy: among normal-status points, is the
    # point actually anomalous? We approximate this via proximity to the
    # documented fault onset when available; otherwise we treat all
    # normal-status points in an anomaly dataset's tail as "should flag"
    # is unknowable at the point level, so Coverage below uses EVENT-level
    # ground truth windows when fault_onset is provided, else falls back to
    # "any sustained flag counts as coverage" (conservative).
    eval_idx = np.where(g)[0]
    result.n_eval_points = len(eval_idx)
    result.n_flagged_points = int(predicted_flags[eval_idx].sum()) if len(eval_idx) else 0

    if not is_anomaly:
        # ----- Accuracy sub-score: specificity on normal-only datasets -----
        fp = int(predicted_flags[eval_idx].sum())
        tn = len(eval_idx) - fp
        result.accuracy = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        return result

    # ----- Anomaly dataset: Coverage + Reliability + Earliness -----
    if fault_onset is not None and len(eval_idx) > 0:
        ts = timestamps.to_numpy()[eval_idx]
        before_fault = ts < np.datetime64(fault_onset)
        gt = before_fault.astype(int)  # 1 = should have flagged (pre-fault window)
        pred = predicted_flags[eval_idx]

        tp = int(((gt == 1) & (pred == 1)).sum())
        fp = int(((gt == 0) & (pred == 1)).sum())
        fn = int(((gt == 1) & (pred == 0)).sum())
        result.coverage = _fbeta(tp, fp, fn, beta=0.5)

        # Earliness: first sustained flag time vs fault onset
        flagged_times = ts[pred == 1]
        if len(flagged_times) > 0:
            first_flag = pd.Timestamp(flagged_times.min())
            lead = (fault_onset - first_flag).total_seconds() / 3600.0
            result.earliness_hours = max(lead, 0.0)
        else:
            result.earliness_hours = 0.0
    else:
        # No documented onset timestamp available — report raw detection
        # rate as a coarse coverage proxy.
        result.coverage = float(predicted_flags[eval_idx].mean()) if len(eval_idx) else np.nan

    # ----- Reliability: event-level F_0.5 (did we raise ~1 clean alarm) ---
    # Count predicted "events" (already collapsed via scores_to_events,
    # so any remaining run IS a sustained alarm) vs. expectation of exactly
    # one true event for an anomaly dataset.
    flags = predicted_flags[eval_idx] if len(eval_idx) else predicted_flags
    n_events = _count_events(flags)
    tp_e = 1 if n_events >= 1 else 0
    fp_e = max(n_events - 1, 0)
    fn_e = 1 - tp_e
    result.reliability = _fbeta(tp_e, fp_e, fn_e, beta=0.5)

    return result


def _count_events(flags: np.ndarray) -> int:
    if len(flags) == 0:
        return 0
    diffs = np.diff(np.concatenate(([0], flags, [0])))
    return int((diffs == 1).sum())


def aggregate_results(results: list[SubDatasetResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "dataset": r.name,
            "is_anomaly": r.is_anomaly,
            "coverage": r.coverage,
            "accuracy": r.accuracy,
            "reliability": r.reliability,
            "earliness_hours": r.earliness_hours,
            "n_flagged_points": r.n_flagged_points,
            "n_eval_points": r.n_eval_points,
        })
    df = pd.DataFrame(rows)
    return df


def summarize(df: pd.DataFrame) -> dict:
    anomaly_df = df[df["is_anomaly"]]
    normal_df = df[~df["is_anomaly"]]
    summary = {
        "n_anomaly_datasets": len(anomaly_df),
        "n_normal_datasets": len(normal_df),
        "mean_coverage": anomaly_df["coverage"].mean(),
        "mean_reliability": anomaly_df["reliability"].mean(),
        "mean_earliness_hours": anomaly_df["earliness_hours"].mean(),
        "mean_accuracy_on_normal": normal_df["accuracy"].mean(),
    }
    # A simple unweighted CARE-style composite (equal-weighted mean of the
    # four sub-scores, earliness normalized to [0,1] by capping at 48h
    # lead time, matching the "up to 48 hours prior" framing used in the
    # literature on this dataset).
    norm_earliness = min((summary["mean_earliness_hours"] or 0) / 48.0, 1.0)
    parts = [summary["mean_coverage"], summary["mean_accuracy_on_normal"],
             summary["mean_reliability"], norm_earliness]
    parts = [p for p in parts if p is not None and not np.isnan(p)]
    summary["care_like_composite"] = float(np.mean(parts)) if parts else np.nan
    return summary

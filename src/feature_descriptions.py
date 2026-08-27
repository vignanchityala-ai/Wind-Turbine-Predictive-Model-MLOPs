"""
feature_descriptions.py
=========================
Loads the per-wind-farm feature/sensor description file (the one with
columns sensor_name, statistics_type, description, unit, is_angle,
is_counter) and derives what the rest of the pipeline needs from it:

- Which base sensors are ANGLES (wind direction, pitch angle, nacelle
  direction) — these need circular statistics (mean/std computed via
  sin/cos decomposition), not linear rolling mean/std. Averaging a linear
  mean across the 359°/1° wrap gives ~180°, which is exactly backwards.
  This is a known, named issue for this dataset (the Zenodo record's
  "Known Data Issues" section calls out Pitch Angle wrapping specifically).
- Which base sensors are COUNTERS (monotonically increasing) — these need
  a rate-of-change feature, not raw rolling stats, though none of the
  Farm A rows seen so far are flagged this way. Supported anyway since the
  file format allows it and the pipeline should stay data-driven, not
  hardcode a column list from a partial view of the file.
- Which actual column is the "real" power signal, given the anonymization
  is inconsistent: some columns keep semantic prefixes (power_29, power_30,
  reactive_power_27, reactive_power_28) while physically-related sensors
  can still be fully anonymized (sensor_31 is actually "Grid reactive
  power" despite the generic name). A naive substring match on "power"
  would wrongly catch reactive_power_* columns. Given both power_29
  ("Possible grid active power") and power_30 ("Grid power") exist, this
  prefers power_30 — "Grid power" reads as the actually-delivered value,
  vs. power_29's "Possible" framing suggesting a theoretical/capacity
  figure less useful for spotting real degradation.
- Same idea for wind speed: wind_speed_3 ("Windspeed", fully measured with
  max/min/avg/std) vs wind_speed_4 ("Estimated windspeed", avg only) —
  prefers the measured one for the power-curve x-axis.

Column-name matching uses base-name + underscore-or-end-of-string, not a
bare prefix check, since "sensor_5" is a substring of "sensor_50" and a
naive startswith() would wrongly pull in unrelated sensors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


def _read_csv_encoding_robust(path) -> pd.DataFrame:
    """
    Try utf-8 first; fall back to latin-1 (mojibake source files are common
    here — unit symbols like ° tend to get exported in cp1252/latin-1 and
    then mis-decoded as utf-8 into "ï¿½" sequences). latin-1 never raises a
    decode error, so this always succeeds, worst case with odd-looking unit
    symbols rather than a crash.
    """
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, sep=None, engine="python", encoding="latin-1")


def _matches_base(column: str, base: str) -> bool:
    """True if `column` is exactly `base` or `base` followed by `_something`
    (e.g. base="sensor_5" matches "sensor_5" and "sensor_5_avg", but NOT
    "sensor_50" or "sensor_51")."""
    return column == base or column.startswith(base + "_")


@dataclass
class FeatureDescriptions:
    angle_bases: set[str] = field(default_factory=set)
    counter_bases: set[str] = field(default_factory=set)
    power_base: str | None = None
    wind_speed_base: str | None = None
    raw: pd.DataFrame | None = None

    def resolve(self, base: str, df_columns) -> list[str]:
        """Actual dataframe columns belonging to a given base sensor name."""
        return [c for c in df_columns if _matches_base(c, base)]

    def pick_primary(self, base: str, matches: list[str]) -> str | None:
        """Among a base's matched columns, prefer the average/mean statistic
        (the representative value for a 10-minute interval) over max/min/std."""
        if not matches:
            return None
        for m in matches:
            low = m.lower()
            if "avg" in low or "average" in low or "mean" in low:
                return m
        return matches[0]

    def angle_columns(self, df_columns) -> list[str]:
        cols = []
        for base in self.angle_bases:
            cols.extend(self.resolve(base, df_columns))
        return cols

    def counter_columns(self, df_columns) -> list[str]:
        cols = []
        for base in self.counter_bases:
            cols.extend(self.resolve(base, df_columns))
        return cols


def _score_power_candidate(description: str) -> int:
    """Lower is more preferred. Penalize language suggesting a theoretical
    /capacity figure rather than actually-delivered power."""
    low = description.lower()
    penalty = 0
    for word in ("possible", "estimated", "theoretical", "capacity"):
        if word in low:
            penalty += 1
    return penalty


def load_feature_descriptions(path) -> FeatureDescriptions:
    """
    Load a feature-description file (columns: sensor_name, statistics_type,
    description, unit, is_angle, is_counter — matching the format shipped
    with the CARE/Wind Farm SCADA dataset) and derive angle/counter base
    names plus preferred power/wind-speed base columns.
    """
    df = _read_csv_encoding_robust(path)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"sensor_name", "description", "is_angle"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path}: missing expected columns {missing}. Found: {list(df.columns)}"
        )

    def _to_bool(series):
        return series.astype(str).str.strip().str.upper().isin(("TRUE", "1", "YES"))

    is_angle = _to_bool(df["is_angle"])
    angle_bases = set(df.loc[is_angle, "sensor_name"].astype(str).str.strip())

    counter_bases = set()
    if "is_counter" in df.columns:
        is_counter = _to_bool(df["is_counter"])
        counter_bases = set(df.loc[is_counter, "sensor_name"].astype(str).str.strip())

    # Preferred power base: sensor_name contains "power" but not "reactive",
    # OR description contains "power" but not "reactive power" — pick the
    # lowest-penalty (least "theoretical-sounding") candidate.
    power_candidates = df[
        (df["sensor_name"].str.contains("power", case=False, na=False)
         & ~df["sensor_name"].str.contains("reactive", case=False, na=False))
        | (df["description"].str.contains("power", case=False, na=False)
           & ~df["description"].str.contains("reactive", case=False, na=False))
    ].copy()
    power_base = None
    if len(power_candidates):
        power_candidates["_penalty"] = power_candidates["description"].apply(_score_power_candidate)
        power_base = power_candidates.sort_values("_penalty").iloc[0]["sensor_name"]

    # Preferred wind-speed base: prefer NOT "estimated".
    wind_candidates = df[df["sensor_name"].str.contains("wind_speed", case=False, na=False)].copy()
    wind_speed_base = None
    if len(wind_candidates):
        wind_candidates["_penalty"] = wind_candidates["description"].apply(
            lambda d: 1 if "estimated" in d.lower() else 0
        )
        wind_speed_base = wind_candidates.sort_values("_penalty").iloc[0]["sensor_name"]

    return FeatureDescriptions(
        angle_bases=angle_bases,
        counter_bases=counter_bases,
        power_base=power_base,
        wind_speed_base=wind_speed_base,
        raw=df,
    )
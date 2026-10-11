"""Event-level evaluation (blueprint Step 8 / Section 8).

Rules that matter and are enforced here:

* **The unit of evaluation is the injected event.** An event counts as detected if
  any alert falls inside it (plus a tolerance of one interval). Neighbourhood
  events are counted once, even though they touch many households.
* **No point-adjusted F1.** Precision is measured over the alerts themselves.
* Thresholds are chosen on validation at a fixed **false-alarm budget per
  household-week**; metrics are then reported on test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from energymind import io

HOUSEHOLD_COL = io.HOUSEHOLD_COL
UTC_COL = io.UTC_COL
INTERVAL = pd.Timedelta(minutes=30)
WEEK = pd.Timedelta(weeks=1)


def _naive_utc(ts) -> np.datetime64:
    """A tz-aware timestamp as naive UTC ``datetime64`` (matches ``Series.to_numpy``)."""
    return pd.Timestamp(ts).tz_convert("UTC").tz_localize(None).to_datetime64()


def _time_array(frame: pd.DataFrame) -> np.ndarray:
    """Row timestamps as naive UTC ``datetime64`` for vectorised comparison."""
    return frame[UTC_COL].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()


EVENT_COLUMNS = [
    "anomaly_id",
    "household_id",
    "t_start",
    "t_end",
    "type",
    "origin",
    "severity",
    "variant",
    "scope",
]


# --------------------------------------------------------------------------- #
# Event expansion and row labels
# --------------------------------------------------------------------------- #


def expand_events(labels: pd.DataFrame) -> pd.DataFrame:
    """One row per (anomaly, household) interval; neighbourhood events explode."""
    records = []
    for row in labels.itertuples(index=False):
        households = str(row.household_id).split(",")
        for hh in households:
            records.append(
                {
                    "anomaly_id": row.anomaly_id,
                    "household_id": hh,
                    "t_start": row.t_start,
                    "t_end": row.t_end,
                    "type": row.type,
                    "origin": row.origin,
                    "severity": row.severity,
                    "variant": row.variant,
                    "scope": row.scope,
                }
            )
    return pd.DataFrame(records, columns=EVENT_COLUMNS)


def point_labels(frame: pd.DataFrame, events: pd.DataFrame, tol: pd.Timedelta) -> pd.Series:
    """Boolean per row: does it fall inside any event window (± tolerance)?"""
    hh = frame[HOUSEHOLD_COL].to_numpy()
    ts = _time_array(frame)
    flag = np.zeros(len(frame), dtype=bool)
    for event in events.itertuples(index=False):
        start = _naive_utc(event.t_start - tol)
        end = _naive_utc(event.t_end + tol)
        mask = (hh == event.household_id) & (ts >= start)
        mask &= ts <= end
        flag |= mask
    return pd.Series(flag, index=frame.index, name="in_event")


# --------------------------------------------------------------------------- #
# Threshold selection and alerts
# --------------------------------------------------------------------------- #


def select_threshold(
    frame: pd.DataFrame,
    scores: pd.Series,
    events: pd.DataFrame,
    budget_per_household_week: float,
    tol_intervals: int = 1,
) -> float:
    """Largest threshold whose false-alarm rate stays within the budget."""
    tol = tol_intervals * INTERVAL
    in_event = point_labels(frame, events, tol).to_numpy()
    clean = scores.to_numpy()[~in_event]
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return float("inf")

    span = frame[UTC_COL].max() - frame[UTC_COL].min()
    weeks = max(span / WEEK, 1e-9)
    n_households = frame[HOUSEHOLD_COL].nunique()
    allowed = budget_per_household_week * n_households * weeks
    allowed = int(np.floor(allowed))
    ordered = np.sort(clean)[::-1]
    if allowed >= len(ordered):
        return float(ordered[-1]) - 1e-9
    return float(ordered[allowed])


def alerts(frame: pd.DataFrame, scores: pd.Series, threshold: float) -> pd.DataFrame:
    """Rows whose score exceeds the threshold, as ``(household_id, ts_utc)``."""
    mask = scores.to_numpy() > threshold
    out = frame.loc[mask, [HOUSEHOLD_COL, UTC_COL]].copy()
    out["score"] = scores.to_numpy()[mask]
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def _detected_by_event(
    frame: pd.DataFrame,
    scores: pd.Series,
    events: pd.DataFrame,
    threshold: float,
    tol: pd.Timedelta,
) -> pd.DataFrame:
    """For each anomaly_id, whether it was detected and its latency (intervals)."""
    hh = frame[HOUSEHOLD_COL].to_numpy()
    ts = _time_array(frame)
    sc = scores.to_numpy()
    rows = []
    for anomaly_id, group in events.groupby("anomaly_id", sort=False):
        detected = False
        latency = np.nan
        for event in group.itertuples(index=False):
            start = _naive_utc(event.t_start - tol)
            end = _naive_utc(event.t_end + tol)
            mask = (hh == event.household_id) & (ts >= start)
            mask &= ts <= end
            hit = mask & (sc > threshold)
            if hit.any():
                detected = True
                first = ts[hit].min()
                lat = (first - _naive_utc(event.t_start)) / np.timedelta64(30, "m")
                latency = np.nanmin([latency, lat]) if not np.isnan(latency) else lat
        meta = group.iloc[0]
        rows.append(
            {
                "anomaly_id": anomaly_id,
                "detected": detected,
                "latency": latency,
                "type": meta["type"],
                "origin": meta["origin"],
                "severity": meta["severity"],
                "variant": meta["variant"],
                "scope": meta["scope"],
            }
        )
    return pd.DataFrame(rows)


def evaluate(
    frame: pd.DataFrame,
    scores: pd.Series,
    events: pd.DataFrame,
    threshold: float,
    budget_per_household_week: float,
    tol_intervals: int = 1,
) -> tuple[dict, pd.DataFrame]:
    """Event-level metrics for one detector/threshold on one scored frame."""
    tol = tol_intervals * INTERVAL
    in_event = point_labels(frame, events, tol).to_numpy()
    sc = scores.to_numpy()
    finite = np.isfinite(sc)
    alert = (sc > threshold) & finite
    n_alerts = int(alert.sum())
    in_event_alert = int((alert & in_event).sum())
    false_alarms = n_alerts - in_event_alert

    span = frame[UTC_COL].max() - frame[UTC_COL].min()
    weeks = max(span / WEEK, 1e-9)
    n_households = frame[HOUSEHOLD_COL].nunique()

    detected = _detected_by_event(frame, scores, events, threshold, tol)
    n_events = len(detected)
    hits = int(detected["detected"].sum())
    recall = hits / n_events if n_events else 0.0
    precision = in_event_alert / n_alerts if n_alerts else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    pr_auc = float(average_precision_score(in_event, np.nan_to_num(sc, nan=-1e9)))

    def _by(column: str) -> dict[str, float]:
        grouped = detected.groupby(column)["detected"].mean()
        return {str(k): round(float(v), 4) for k, v in grouped.items()}

    metrics = {
        "n_events": n_events,
        "n_alerts": n_alerts,
        "false_alarms": false_alarms,
        "event_recall": round(recall, 4),
        "alert_precision": round(precision, 4),
        "event_f1": round(f1, 4),
        "pr_auc": round(pr_auc, 4),
        "fa_per_household_week": round(false_alarms / (n_households * weeks), 4),
        "latency_median": round(float(detected["latency"].median()), 2)
        if detected["latency"].notna().any()
        else None,
        "threshold": round(float(threshold), 6),
        "recall_by_type": _by("type"),
        "recall_by_severity": _by("severity"),
        "recall_by_variant": _by("variant"),
        "recall_by_scope": _by("scope"),
    }
    return metrics, detected


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 10
) -> float:
    """Expected calibration error of a probability vector."""
    y_true = np.asarray(y_true, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    bins = np.clip((probabilities * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if not mask.any():
            continue
        conf = probabilities[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.mean()) * abs(conf - acc)
    return float(ece)

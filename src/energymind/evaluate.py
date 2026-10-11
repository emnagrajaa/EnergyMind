"""Event-level evaluation (blueprint Step 8 / Section 8).

Rules that matter and are enforced here:

* **The unit of evaluation is the injected event.** An event counts as detected if
  any alert falls inside it (plus a tolerance). Neighbourhood events are counted
  once, even though they touch many households.
* **No point-adjusted F1.** Precision is measured over the alerts themselves.
* Thresholds are chosen on validation at a fixed **false-alarm budget per
  household-week**, but *at event level*: among thresholds that respect the
  budget we pick the one that maximises event-F1 (not a point quantile). A
  **per-household** mode and an **oracle** (unconstrained) ceiling are also
  provided.
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
#: Event-length buckets (intervals) for the recall breakdown.
LENGTH_BINS = [(1, 1, "1"), (2, 6, "2-6"), (7, 48, "7-48"), (49, 10**9, "49+")]


# --------------------------------------------------------------------------- #
# Time helpers
# --------------------------------------------------------------------------- #


def _naive_utc(ts) -> np.datetime64:
    """A tz-aware timestamp as naive UTC ``datetime64`` (matches ``Series.to_numpy``)."""
    return pd.Timestamp(ts).tz_convert("UTC").tz_localize(None).to_datetime64()


def _time_array(frame: pd.DataFrame) -> np.ndarray:
    return frame[UTC_COL].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()


# --------------------------------------------------------------------------- #
# Event expansion and row labels
# --------------------------------------------------------------------------- #


def expand_events(labels: pd.DataFrame) -> pd.DataFrame:
    """One row per (anomaly, household) interval; neighbourhood events explode."""
    records = []
    for row in labels.itertuples(index=False):
        for hh in str(row.household_id).split(","):
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
        mask = (hh == event.household_id) & (ts >= _naive_utc(event.t_start - tol))
        mask &= ts <= _naive_utc(event.t_end + tol)
        flag |= mask
    return pd.Series(flag, index=frame.index, name="in_event")


def _event_windows(frame: pd.DataFrame, events: pd.DataFrame, tol: pd.Timedelta) -> dict:
    """Map each anomaly_id to the list of its per-household row masks."""
    hh = frame[HOUSEHOLD_COL].to_numpy()
    ts = _time_array(frame)
    windows: dict[str, list[np.ndarray]] = {}
    for event in events.itertuples(index=False):
        start = _naive_utc(event.t_start - tol)
        end = _naive_utc(event.t_end + tol)
        mask = (hh == event.household_id) & (ts >= start) & (ts <= end)
        windows.setdefault(event.anomaly_id, []).append(mask)
    return windows


def relevant_events(frame: pd.DataFrame, events: pd.DataFrame, tol: pd.Timedelta) -> pd.DataFrame:
    """Keep only event rows that overlap ``frame`` (time + household).

    Metrics are computed on the events that fall inside the evaluated split, so a
    test-split score is not diluted by the train/val events the detector never saw.
    """
    windows = _event_windows(frame, events, tol)
    present = {anomaly_id for anomaly_id, masks in windows.items() if any(m.any() for m in masks)}
    if not present:
        return events.iloc[0:0]
    return events[events["anomaly_id"].isin(present)].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Threshold selection
# --------------------------------------------------------------------------- #


def _budget_allowed(frame: pd.DataFrame, budget: float) -> float:
    span = frame[UTC_COL].max() - frame[UTC_COL].min()
    weeks = max(span / WEEK, 1e-9)
    return budget * frame[HOUSEHOLD_COL].nunique() * weeks


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall + 1e-9)


def sweep_thresholds(
    frame: pd.DataFrame,
    scores: pd.Series,
    events: pd.DataFrame,
    tol: pd.Timedelta,
    budget: float | None = None,
    n_candidates: int = 240,
) -> dict:
    """Sweep candidate thresholds; return the best event-F1 (optionally within budget).

    Event detection for a threshold is ``event_max > threshold`` where
    ``event_max`` is the largest score inside the event window, so the sweep is
    cheap regardless of window length.
    """
    sc = scores.to_numpy()
    finite = np.isfinite(sc)
    events = relevant_events(frame, events, tol)
    in_event = point_labels(frame, events, tol).to_numpy()
    clean = sc[~in_event & finite]
    if len(clean) == 0:
        return {"threshold": float("inf"), "f1": 0.0, "fa": 0, "recall": 0.0, "precision": 0.0}

    windows = _event_windows(frame, events, tol)
    n_events = len(windows)

    def _window_max(masks):
        combined = np.logical_or.reduce(masks) if masks else np.zeros(len(frame), dtype=bool)
        return float(np.nanmax(sc[combined])) if combined.any() else -np.inf

    event_max = np.array([_window_max(masks) for masks in windows.values()])

    allowed = _budget_allowed(frame, budget) if budget is not None else None
    # Sweep thresholds across the clean-score distribution (the operating curve);
    # the event-window maxima are compared against each candidate, not folded in.
    candidates = np.unique(np.quantile(clean, np.linspace(0.5, 0.99999, n_candidates)))
    candidates = np.append(candidates, np.inf)

    best = None
    for thr in candidates:
        fa = int((clean > thr).sum())
        if allowed is not None and fa > allowed:
            continue
        hits = int((event_max > thr).sum())
        recall = hits / n_events if n_events else 0.0
        in_alert = int(((sc > thr) & in_event & finite).sum())
        n_alerts = fa + in_alert
        precision = in_alert / n_alerts if n_alerts else 0.0
        f1 = _f1(precision, recall)
        if best is None or f1 > best["f1"]:
            best = {
                "threshold": float(thr),
                "f1": f1,
                "fa": fa,
                "recall": recall,
                "precision": precision,
            }
    if best is not None:
        return best
    # no candidate respected the budget: fall back to the largest threshold that does
    ordered = np.sort(clean)[::-1]
    idx = min(int(allowed), len(ordered) - 1) if allowed is not None else 0
    return {
        "threshold": float(ordered[idx]),
        "f1": 0.0,
        "fa": idx,
        "recall": 0.0,
        "precision": 0.0,
    }


def select_threshold(
    frame: pd.DataFrame,
    scores: pd.Series,
    events: pd.DataFrame,
    budget_per_household_week: float,
    tol_intervals: int = 1,
) -> float:
    """Event-F1-optimal threshold within the false-alarm budget."""
    result = sweep_thresholds(
        frame, scores, events, tol_intervals * INTERVAL, budget=budget_per_household_week
    )
    return result["threshold"]


def oracle_threshold(
    frame: pd.DataFrame, scores: pd.Series, events: pd.DataFrame, tol_intervals: int = 1
) -> float:
    """Threshold that maximises event-F1 with no false-alarm constraint (ceiling)."""
    return sweep_thresholds(frame, scores, events, tol_intervals * INTERVAL, budget=None)[
        "threshold"
    ]


def select_threshold_per_household(
    frame: pd.DataFrame,
    scores: pd.Series,
    events: pd.DataFrame,
    budget_per_household_week: float,
    tol_intervals: int = 1,
) -> pd.Series:
    """Per-household threshold meeting each household's share of the FA budget."""
    tol = tol_intervals * INTERVAL
    in_event = point_labels(frame, events, tol)
    allowed = _budget_allowed(frame, budget_per_household_week)
    households = frame[HOUSEHOLD_COL].unique()
    per_hh = allowed / len(households)
    thresholds = {}
    for hh in households:
        clean = scores[(frame[HOUSEHOLD_COL] == hh) & ~in_event].to_numpy()
        clean = clean[np.isfinite(clean)]
        if len(clean) == 0:
            thresholds[hh] = float("inf")
            continue
        ordered = np.sort(clean)[::-1]
        thresholds[hh] = float(ordered[min(int(per_hh), len(ordered) - 1)])
    return pd.Series(thresholds)


def _alert_mask(frame: pd.DataFrame, scores: pd.Series, threshold) -> np.ndarray:
    sc = scores.to_numpy()
    if isinstance(threshold, pd.Series):
        thr = frame[HOUSEHOLD_COL].map(threshold).to_numpy(dtype=float)
    else:
        thr = float(threshold)
    return (sc > thr) & np.isfinite(sc)


def alerts(frame: pd.DataFrame, scores: pd.Series, threshold) -> pd.DataFrame:
    """Rows whose score exceeds the threshold, as ``(household_id, ts_utc, score)``."""
    mask = _alert_mask(frame, scores, threshold)
    out = frame.loc[mask, [HOUSEHOLD_COL, UTC_COL]].copy()
    out["score"] = scores.to_numpy()[mask]
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def _detected_by_event(
    frame: pd.DataFrame, alert: np.ndarray, events: pd.DataFrame, tol: pd.Timedelta
) -> pd.DataFrame:
    """Per anomaly_id: detected?, latency in intervals, and labels."""
    ts = _time_array(frame)
    windows = _event_windows(frame, events, tol)
    meta = events.groupby("anomaly_id", sort=False).first()
    rows = []
    for anomaly_id, masks in windows.items():
        detected = False
        latency = np.nan
        for mask in masks:
            hit = mask & alert
            if hit.any():
                detected = True
                event = events[events.anomaly_id == anomaly_id].iloc[0]
                first = ts[hit].min()
                lat = (first - _naive_utc(event["t_start"])) / np.timedelta64(30, "m")
                latency = np.nanmin([latency, lat]) if not np.isnan(latency) else lat
        info = meta.loc[anomaly_id]
        rows.append(
            {
                "anomaly_id": anomaly_id,
                "detected": detected,
                "latency": latency,
                "type": info["type"],
                "origin": info["origin"],
                "severity": info["severity"],
                "variant": info["variant"],
                "scope": info["scope"],
            }
        )
    return pd.DataFrame(rows)


def _recall_by_length(detected: pd.DataFrame, events: pd.DataFrame) -> dict[str, float]:
    lengths = (
        events.groupby("anomaly_id")["t_start"]
        .first()
        .to_frame()
        .join(events.groupby("anomaly_id")["t_end"].first())
        .pipe(lambda d: (d["t_end"] - d["t_start"]).dt.total_seconds() / 1800)
        .rename("length")
    )
    merged = detected.set_index("anomaly_id").join(lengths)
    out = {}
    for lo, hi, label in LENGTH_BINS:
        subset = merged[(merged["length"] >= lo) & (merged["length"] <= hi)]
        if len(subset):
            out[label] = round(float(subset["detected"].mean()), 4)
    return out


def evaluate(
    frame: pd.DataFrame,
    scores: pd.Series,
    events: pd.DataFrame,
    threshold,
    budget_per_household_week: float,
    tol_intervals: int = 1,
) -> tuple[dict, pd.DataFrame]:
    """Event-level metrics for one detector/threshold on one scored frame."""
    tol = tol_intervals * INTERVAL
    events = relevant_events(frame, events, tol)
    in_event = point_labels(frame, events, tol).to_numpy()
    alert = _alert_mask(frame, scores, threshold)
    sc = scores.to_numpy()
    finite_vals = sc[np.isfinite(sc)]
    fill = (float(finite_vals.min()) - 1.0) if finite_vals.size else 0.0
    sc = np.where(np.isfinite(sc), sc, fill)
    n_alerts = int(alert.sum())
    in_event_alert = int((alert & in_event).sum())
    false_alarms = n_alerts - in_event_alert

    span = frame[UTC_COL].max() - frame[UTC_COL].min()
    weeks = max(span / WEEK, 1e-9)
    n_households = frame[HOUSEHOLD_COL].nunique()

    detected = _detected_by_event(frame, alert, events, tol)
    n_events = len(detected)
    hits = int(detected["detected"].sum())
    recall = hits / n_events if n_events else 0.0
    precision = in_event_alert / n_alerts if n_alerts else 0.0
    f1 = _f1(precision, recall)
    pr_auc = float(average_precision_score(in_event, sc))

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
        "threshold": float(threshold) if np.isscalar(threshold) else None,
        "recall_by_type": _by("type"),
        "recall_by_severity": _by("severity"),
        "recall_by_variant": _by("variant"),
        "recall_by_scope": _by("scope"),
        "recall_by_length": _recall_by_length(detected, events),
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
        ece += (mask.mean()) * abs(probabilities[mask].mean() - y_true[mask].mean())
    return float(ece)

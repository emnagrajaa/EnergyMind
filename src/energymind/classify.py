"""Type + origin classification (blueprint Step 8).

Detection and classification are kept separate: unsupervised detectors cannot emit
a *type*, so a supervised model predicts it. Each **injected event** becomes one
case: the detector-relevant features are aggregated over the event window, and the
classifier predicts the anomaly ``type`` and ``origin``. LightGBM is the primary
model (isotonic-calibrated); Random Forest is the TreeSHAP-friendly comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from energymind import io
from energymind.evaluate import expected_calibration_error

HOUSEHOLD_COL = io.HOUSEHOLD_COL
UTC_COL = io.UTC_COL

#: Per-row columns aggregated over an event window.
AGG_COLUMNS = [
    "p_resid_z",
    "neigh_share",
    "p_ramp",
    "unmetered_share",
    "sub2_share",
    "pv_offset_ratio",
    "i_roll_mean_2h",
    "v_roll_mean_2h",
]
#: Columns aggregated by max (physics residuals + neighbourhood share).
MAX_COLUMNS = ["r_S", "r_VI", "r_E", "r_PF"]


# --------------------------------------------------------------------------- #
# Per-event dataset
# --------------------------------------------------------------------------- #


def build_event_dataset(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    scores: dict[str, pd.Series] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(X, meta)``: one row per anomaly_id with aggregated features.

    ``events`` is the expanded per-household frame (``evaluate.expand_events``),
    and ``scores`` maps a detector name to its per-row score.
    """
    scores = scores or {}
    frame = frame.assign(
        _physics=frame[["r_S", "r_VI", "r_E", "r_PF"]].abs().max(axis=1).fillna(0.0)
    )
    det = pd.DataFrame(dict(scores))
    frame = pd.concat([frame, det], axis=1)

    feature_rows = []
    meta_rows = []
    for anomaly_id, group in events.groupby("anomaly_id", sort=False):
        mask = pd.Series(False, index=frame.index)
        for event in group.itertuples(index=False):
            mask |= (frame[HOUSEHOLD_COL] == event.household_id) & frame[UTC_COL].between(
                event.t_start, event.t_end, inclusive="left"
            )
        block = frame.loc[mask]
        row: dict[str, float] = {}
        for column in AGG_COLUMNS + ["_physics", *scores]:
            if column not in block.columns:
                continue
            values = block[column].astype(float)
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_max"] = float(values.max())
        for column in MAX_COLUMNS:
            row[f"{column}_max"] = float(block[column].abs().max())
        info = group.iloc[0]
        duration = (info["t_end"] - info["t_start"]).total_seconds() / 3600.0
        row["duration_h"] = float(duration)
        row["scope_neighbourhood"] = float(info["scope"] == "neighborhood")
        row["n_households"] = float(group[HOUSEHOLD_COL].nunique())
        feature_rows.append(row)
        meta_rows.append(
            {
                "anomaly_id": anomaly_id,
                "type": info["type"],
                "origin": info["origin"],
                "severity": info["severity"],
                "variant": info["variant"],
                "scope": info["scope"],
                "t_start": info["t_start"],
            }
        )
    return pd.DataFrame(feature_rows), pd.DataFrame(meta_rows)


def assign_event_split(meta: pd.DataFrame, settings, frame: pd.DataFrame) -> pd.Series:
    """Label each event's split (train/val/test) from its start time."""
    from energymind.splits import assign_temporal

    offset = settings.data.time.assume_utc_offset_h
    edges = {}
    t = settings.data.splits.temporal
    for name in ("train", "val", "test"):
        spec = getattr(t, name)
        edges[name] = (
            pd.Timestamp(spec.start).tz_localize("UTC") - pd.Timedelta(hours=offset),
            pd.Timestamp(spec.end).tz_localize("UTC") - pd.Timedelta(hours=offset),
        )
    out = pd.Series("train", index=meta.index, dtype="object")
    for name, (start, end) in edges.items():
        out[(meta["t_start"] >= start) & (meta["t_start"] < end)] = name
    _ = assign_temporal  # documented: same UTC edges as the row split
    _ = frame
    return out


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


def _base_lightgbm():
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=0
    )


def _base_rf():
    return RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1)


def _fit_calibrated(base, x: pd.DataFrame, y: pd.Series):
    """Fit ``base`` with isotonic calibration, degrading gracefully for rare classes."""
    counts = y.value_counts()
    min_count = int(counts.min()) if len(counts) else 0
    if min_count >= 3:
        return CalibratedClassifierCV(
            _base_lightgbm() if base == "lgbm" else _base_rf(), method="isotonic", cv=3
        ).fit(x, y)
    if min_count >= 2:
        return CalibratedClassifierCV(
            _base_lightgbm() if base == "lgbm" else _base_rf(), method="sigmoid", cv=2
        ).fit(x, y)
    return (_base_lightgbm() if base == "lgbm" else _base_rf()).fit(x, y)


@dataclass
class EventClassifier:
    """Two calibrated multiclass models: one for ``type``, one for ``origin``."""

    name: str
    type_model: object
    origin_model: object
    classes_type: list[str]
    classes_origin: list[str]

    @classmethod
    def fit(
        cls, x: pd.DataFrame, y_type: pd.Series, y_origin: pd.Series, name: str
    ) -> EventClassifier:
        base = "lgbm" if name == "lightgbm" else "rf"
        type_model = _fit_calibrated(base, x, y_type)
        origin_model = _fit_calibrated(base, x, y_origin)
        return cls(
            name=name,
            type_model=type_model,
            origin_model=origin_model,
            classes_type=list(type_model.classes_),
            classes_origin=list(origin_model.classes_),
        )

    def predict(self, x: pd.DataFrame) -> dict:
        type_proba = self.type_model.predict_proba(x)
        origin_proba = self.origin_model.predict_proba(x)
        return {
            "type_proba": type_proba,
            "origin_proba": origin_proba,
            "type_pred": [self.classes_type[i] for i in type_proba.argmax(axis=1)],
            "origin_pred": [self.classes_origin[i] for i in origin_proba.argmax(axis=1)],
        }


def classifier_metrics(y_type: pd.Series, y_origin: pd.Series, pred: dict) -> dict:
    """Macro-F1 (type), accuracy (origin) and ECE (origin confidence)."""
    origin_correct = (pred["origin_pred"] == y_origin.to_numpy()).astype(float)
    origin_conf = pred["origin_proba"].max(axis=1)
    return {
        "type_macro_f1": round(
            float(f1_score(y_type, pred["type_pred"], average="macro", zero_division=0)), 4
        ),
        "origin_accuracy": round(float(accuracy_score(y_origin, pred["origin_pred"])), 4),
        "origin_ece": round(expected_calibration_error(origin_correct, origin_conf), 4),
    }

"""Step 9 - anomaly records: 64-dim signature, record schema and template text.

The record separates **prediction** from **cause**:

* ``detection`` / ``classification`` are model outputs;
* ``explanation`` (SHAP/LIME) explains the model, not the physical cause;
* ``cause_hypotheses`` are written as hypotheses with supporting evidence;
* ``evidence`` holds observed-versus-expected values with units;
* ``text`` is filled from a **template over the evidence**, so numbers in the
  operator-facing sentence cannot be invented.

Schema matches blueprint Section 9.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

SIGNATURE_DIM = 64
SHAPE_POINTS = 16  # P, V, I each resampled to 16 -> 48 dims
_CHANNELS = ["active_power_kw", "voltage_v", "current_a"]
_NILM = [
    "nilm_resid_sub1_heating_kwh",
    "nilm_resid_sub2_cooling_kwh",
    "nilm_resid_sub3_lighting_kwh",
    "nilm_resid_sub4_appliance_kwh",
    "nilm_resid_total",
]


# --------------------------------------------------------------------------- #
# Signature
# --------------------------------------------------------------------------- #


def _shape(series: pd.Series, points: int = SHAPE_POINTS) -> np.ndarray:
    values = pd.Series(series, dtype=float).interpolate(limit_direction="both").to_numpy()
    values = np.nan_to_num(values, nan=0.0)
    if len(values) < 2:
        values = np.zeros(2) if len(values) == 0 else np.repeat(values, 2)
    resampled = np.interp(np.linspace(0, len(values) - 1, points), np.arange(len(values)), values)
    scale = resampled.std()
    return (resampled - resampled.mean()) / scale if scale > 0 else np.zeros(points)


def build_signature(window: pd.DataFrame, t_start: pd.Timestamp) -> np.ndarray:
    """Fixed 64-dim numeric signature of a case (see Section 10).

    48 dims shape (P, V, I resampled to 16 each) + duration + magnitude/sigma +
    4 identity residuals + neighbourhood share + 5 NILM residuals + hour/weekday
    encodings = 64.
    """
    parts: list[float] = []
    for column in _CHANNELS:
        parts.extend(
            _shape(window[column]).tolist() if column in window.columns else [0.0] * SHAPE_POINTS
        )

    duration_h = (window["ts_utc"].max() - window["ts_utc"].min()).total_seconds() / 3600.0
    parts.append(float(min(duration_h / 24.0, 1.0)))

    sigma = 1.4826 * float(
        np.nanmedian(np.abs(window["active_power_kw"] - window["active_power_kw"].median()))
    )
    magnitude = 0.0
    if sigma > 0 and "how_median" in window.columns:
        shift = float(np.nanmedian(window["active_power_kw"] - window["how_median"]))
        magnitude = float(np.tanh(shift / (5.0 * sigma)))
    parts.append(magnitude)

    for column in ["r_S", "r_VI", "r_E", "r_PF"]:
        value = float(np.nanmedian(np.abs(window[column]))) if column in window.columns else 0.0
        parts.append(float(np.tanh(value)))

    parts.append(float(window["neigh_share"].max()) if "neigh_share" in window.columns else 0.0)

    for column in _NILM:
        value = float(np.nanmedian(window[column])) if column in window.columns else 0.0
        parts.append(float(np.tanh(value)))

    local = pd.Timestamp(t_start).tz_convert("UTC").tz_localize(None) - pd.Timedelta(hours=6)
    parts.extend(
        [
            float(np.sin(2 * np.pi * local.hour / 24)),
            float(np.cos(2 * np.pi * local.hour / 24)),
            float(np.sin(2 * np.pi * local.dayofweek / 7)),
            float(np.cos(2 * np.pi * local.dayofweek / 7)),
        ]
    )

    signature = np.asarray(parts, dtype=np.float32)
    if len(signature) != SIGNATURE_DIM:  # guard the contract
        signature = np.pad(signature, (0, SIGNATURE_DIM - len(signature)))[:SIGNATURE_DIM]
    return signature


# --------------------------------------------------------------------------- #
# Cause hypotheses
# --------------------------------------------------------------------------- #


def cause_hypotheses(origin: str | None, evidence: list[dict]) -> list[str]:
    """Hypotheses with supporting evidence, keyed on origin and the checks."""
    by_name = {entry["check"]: entry for entry in evidence}
    identity = by_name.get("current_vs_power_identity", {})
    baseline = by_name.get("baseline_deviation", {})
    coherence = by_name.get("neighborhood_share", {})
    hypotheses: list[str] = []

    if identity.get("status") == "broken":
        hypotheses.append(
            "Single-channel corruption (transmission/storage): the electrical "
            f"identity is broken (rel. residual {identity.get('rel_residual')})."
        )
    if origin == "physical_event" or coherence.get("status") == "coherent":
        hypotheses.append(
            "Genuine grid event: the deviation is coherent across households "
            f"(share {coherence.get('value')})."
        )
    if origin == "adversarial":
        hypotheses.append(
            "Possible manipulation (under-reporting/theft): identities hold and "
            f"the level is depressed (z={baseline.get('z')})."
        )
    if not hypotheses or (identity.get("status") != "broken" and origin == "data_quality"):
        hypotheses.append(
            "Upstream data-quality fault: identities hold but the load departs from "
            f"the household baseline (observed {baseline.get('observed_kw')} kW vs "
            f"expected {baseline.get('expected_kw')} kW)."
        )
    return hypotheses


# --------------------------------------------------------------------------- #
# Record
# --------------------------------------------------------------------------- #


def render_text(record: dict) -> str:
    """Operator sentence, filled only from the evidence numbers."""
    evidence = {entry["check"]: entry for entry in record["evidence"]}
    baseline = evidence.get("baseline_deviation", {})
    coherence = evidence.get("neighborhood_share", {})
    identity = evidence.get("current_vs_power_identity", {})
    hh = (
        str(record["household_id"]).split("_")[1]
        if "_" in str(record["household_id"])
        else record["household_id"]
    )
    duration_h = record.get("duration_h")

    sentences = []
    if baseline.get("observed_kw") is not None and baseline.get("expected_kw") is not None:
        sentences.append(
            f"Household {hh} used {baseline['observed_kw']} kW for {duration_h} h versus an "
            f"expected {baseline['expected_kw']} kW (z={baseline.get('z')})."
        )
    if coherence.get("value") is not None:
        share = coherence["value"]
        note = (
            "so a meter fault is more likely than real demand."
            if share < 0.4
            else "consistent with a network event."
        )
        sentences.append(f"{share:.0%} of neighbours rose, {note}")
    if identity.get("status") == "broken":
        sentences.append(
            f"An electrical identity is broken (rel. residual {identity.get('rel_residual')})."
        )
    return " ".join(sentences)


def build_record(
    *,
    anomaly_id: str,
    household_id: str,
    t_start,
    t_end,
    detection: dict,
    classification: dict,
    evidence: list[dict],
    explanation: dict,
    signature: np.ndarray,
    severity: str | None = None,
    limitations: list[str] | None = None,
    versions: dict | None = None,
) -> dict:
    """Assemble the anomaly record (blueprint Section 9)."""
    type_probs = classification.get("type_probs", {})
    record = {
        "anomaly_id": anomaly_id,
        "household_id": household_id,
        "t_start": str(t_start),
        "t_end": str(t_end),
        "duration_h": round(
            (pd.Timestamp(t_end) - pd.Timestamp(t_start)).total_seconds() / 3600.0, 2
        ),
        "detection": detection,
        "classification": classification,
        "evidence": evidence,
        "explanation": explanation,
        "cause_hypotheses": cause_hypotheses(classification.get("origin_pred"), evidence),
        "severity": severity,
        "limitations": limitations
        or ["weather is synthetic and weakly informative", "operator behaviour is simulated"],
        "signature": [round(float(x), 5) for x in np.asarray(signature)],
        "versions": versions or {},
        "type_pred": classification.get("type_pred"),
        "origin_pred": classification.get("origin_pred"),
        "text": "",
    }
    _ = type_probs
    record["text"] = render_text(record)
    return record


def record_to_json(record: dict) -> str:
    return json.dumps(record, ensure_ascii=False, indent=2)

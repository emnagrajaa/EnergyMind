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


def _safe_median(values) -> float:
    """Median that returns 0.0 instead of NaN (and no warning) on an all-NaN input."""
    arr = np.asarray(values, dtype=float)
    return float(np.nanmedian(arr)) if np.isfinite(arr).any() else 0.0


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

    p = window["active_power_kw"]
    sigma = 1.4826 * _safe_median(np.abs(p - p.median()))
    magnitude = 0.0
    if sigma > 0 and "how_median" in window.columns:
        shift = _safe_median(p - window["how_median"])
        magnitude = float(np.tanh(shift / (5.0 * sigma)))
    parts.append(magnitude)

    for column in ["r_S", "r_VI", "r_E", "r_PF"]:
        value = _safe_median(np.abs(window[column])) if column in window.columns else 0.0
        parts.append(float(np.tanh(value)))

    parts.append(float(window["neigh_share"].max()) if "neigh_share" in window.columns else 0.0)

    for column in _NILM:
        value = _safe_median(window[column]) if column in window.columns else 0.0
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

    signature = np.nan_to_num(np.asarray(parts, dtype=np.float32))
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


def facts_from_evidence(evidence: list[dict]) -> dict:
    """Minimal facts derived from the evidence list (used when no window is given)."""
    by = {entry["check"]: entry for entry in evidence}
    baseline = by.get("baseline_deviation", {})
    identity = by.get("current_vs_power_identity", {})
    coherence = by.get("neighborhood_share", {})
    return {
        "observed_kw": baseline.get("observed_kw"),
        "expected_kw": baseline.get("expected_kw"),
        "z": baseline.get("z"),
        "share": coherence.get("value"),
        "coherent": (coherence.get("value") or 0) >= 0.4,
        "identity_rel": identity.get("rel_residual") or 0.0,
        "identity_broken": identity.get("status") == "broken",
        "zero_frac": 0.0,
        "zero_reading": False,
        "flat_run": 0,
        "voltage_dev": None,
        "nilm_resid_total": None,
        "nilm_resid_sub2": None,
    }


def narrative(facts: dict, origin: str | None, type_pred: str | None) -> list[str]:
    """Case-specific sentences built from measured facts (never pre-written)."""
    obs = facts.get("observed_kw")
    exp = facts.get("expected_kw")
    z = facts.get("z")
    share = facts.get("share")
    sentences: list[str] = []

    if facts.get("zero_reading"):
        sentences.append(
            f"The meter read exactly zero for {facts.get('zero_frac'):.0%} of the window, "
            f"although this household normally draws {exp} kW at this hour. "
            "Conclusion: sensor outage (dropout), not zero consumption."
        )
    elif type_pred == "dead_sensor":
        sentences.append(
            f"The meter reports zero active power while voltage and current stay normal "
            f"(normally {exp} kW here). Conclusion: dead sensor, not a real outage."
        )

    if facts.get("identity_broken"):
        sentences.append(
            f"The sample breaks the electrical identity (V*I != S; relative residual "
            f"{facts.get('identity_rel')}). Conclusion: corrupt or falsified sample."
        )

    neighbor = facts.get("neighbor_median_kw")
    local = facts.get("local_before_kw")
    if (
        obs is not None
        and neighbor is not None
        and neighbor > 0
        and abs(obs - neighbor) / neighbor > 0.5
        and not facts.get("coherent")
    ):
        reference = (
            f"neighbouring intervals ({local} kW) and the neighbourhood median ({neighbor} kW)"
            if local is not None
            else f"the neighbourhood median ({neighbor} kW)"
        )
        sentences.append(
            f"The reading {obs} kW is inconsistent with {reference} and the equipment "
            "estimates. Conclusion: likely falsified or corrupt sample."
        )

    if facts.get("nilm_resid_total") is not None and abs(facts["nilm_resid_total"]) > 0.2:
        sentences.append(
            f"The measured sub-meters disagree with the NILM equipment estimate "
            f"(residual {facts['nilm_resid_total']} kWh)."
        )

    if share is not None:
        if facts.get("coherent"):
            sentences.append(
                f"The deviation is shared by {share:.0%} of neighbours at the same time. "
                "Conclusion: genuine network event, not a single-meter fault."
            )
        else:
            sentences.append(
                f"Only {share:.0%} of neighbours moved with it, so a single-meter fault is "
                "more likely than real demand."
            )

    if facts.get("flat_run", 0) >= 8:
        sentences.append(
            f"Active power is constant for {facts['flat_run']} intervals "
            f"({facts['flat_run'] * 0.5:.1f} h). Conclusion: frozen / stuck sensor."
        )

    if facts.get("voltage_dev") is not None and facts["voltage_dev"] > 5:
        sentences.append(
            f"Voltage reads {facts.get('voltage_median')} V against the 230 V nominal "
            f"(deviation {facts['voltage_dev']} V)."
        )

    if type_pred == "sub_meter_bias" or (
        facts.get("nilm_resid_sub2") is not None and abs(facts["nilm_resid_sub2"]) > 0.2
    ):
        sentences.append(
            f"The cooling sub-meter disagrees with the aggregate and the equipment estimate "
            f"(residual {facts.get('nilm_resid_sub2')} kWh). Conclusion: biased sub-meter."
        )

    if origin == "adversarial":
        sentences.append(
            "Load is depressed while the identities and neighbours stay normal. "
            "Conclusion: under-reporting / theft is more likely than a sensor fault."
        )

    if exp is not None and obs is not None and not facts.get("zero_reading"):
        sentences.append(
            f"Observed {obs} kW versus an expected {exp} kW for this hour-of-week (z={z})."
        )
    return sentences


def render_text(record: dict) -> str:
    """Operator text: the stored narrative, or one rebuilt from the evidence."""
    if record.get("text"):
        return record["text"]
    facts = record.get("facts") or facts_from_evidence(record["evidence"])
    return " ".join(narrative(facts, record.get("origin_pred"), record.get("type_pred")))


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
    facts: dict | None = None,
) -> dict:
    """Assemble the anomaly record (blueprint Section 9)."""
    if facts is None:
        facts = facts_from_evidence(evidence)
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
        "facts": facts,
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
    record["text"] = " ".join(
        narrative(facts, classification.get("origin_pred"), classification.get("type_pred"))
    )
    return record


def record_to_json(record: dict) -> str:
    return json.dumps(record, ensure_ascii=False, indent=2)

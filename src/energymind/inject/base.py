"""Anomaly injection core: dataclasses, channel propagators and sampling helpers.

Anomalies are injected in two variants (blueprint Section 7):

* **naive** - edits one/few channels, leaving the electrical identities broken
  (models corruption in transmission or storage; a physics check catches it);
* **consistent** - edits the *root* quantity and recomputes the dependent
  channels (models an upstream sensor fault; the identities still hold).

All edits go through the propagators below so a consistent variant can never
leave an identity broken by accident.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from energymind import io
from energymind.validate import SUB_COLS, recompute_dependents

HOUSEHOLD_COL = io.HOUSEHOLD_COL
UTC_COL = io.UTC_COL

ORIGINS = ("data_quality", "adversarial", "physical_event")
VARIANTS = ("naive", "consistent")
SCOPES = ("single", "neighborhood")

#: Severity thresholds in units of the household robust scale (sigma).
SEVERITY_LOW = 2.0
SEVERITY_HIGH = 5.0

#: Measured channels blanked / copied by whole-block faults.
ELECTRICAL_COLUMNS = [
    "voltage_v",
    "current_a",
    "active_power_kw",
    "reactive_power_kvar",
    "apparent_power_kva",
    "power_factor",
    "frequency_hz",
    "total_energy_kwh",
    *SUB_COLS,
    "ev_load_kw",
    "solar_generation_kw",
    "battery_charge_kw",
    "battery_discharge_kw",
]


# --------------------------------------------------------------------------- #
# Label
# --------------------------------------------------------------------------- #


@dataclass
class Injection:
    anomaly_id: str
    household_id: str
    t_start: pd.Timestamp
    t_end: pd.Timestamp
    type: str
    origin: str
    severity: str
    variant: str
    scope: str = "single"
    params: dict = field(default_factory=dict)
    seed: int = 0
    injector_version: str = "0.1.0"
    provenance: str = "simulated"

    def as_row(self) -> dict:
        return {
            "anomaly_id": self.anomaly_id,
            "household_id": self.household_id,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "type": self.type,
            "origin": self.origin,
            "severity": self.severity,
            "variant": self.variant,
            "scope": self.scope,
            "params": json.dumps(self.params, sort_keys=True, default=str),
            "seed": self.seed,
            "injector_version": self.injector_version,
            "provenance": self.provenance,
        }


# --------------------------------------------------------------------------- #
# Severity
# --------------------------------------------------------------------------- #


def effect_in_sigma(p_before, p_after, sigma: float) -> float:
    """Level shift of a channel over the window, in units of ``sigma``.

    NaNs count as 0 (an outage or dropout shifts the level to zero).
    """
    if not sigma or not np.isfinite(sigma) or sigma <= 0:
        return 0.0
    before = np.nan_to_num(np.asarray(p_before, dtype=float), nan=0.0)
    after = np.nan_to_num(np.asarray(p_after, dtype=float), nan=0.0)
    return float(abs(np.median(after) - np.median(before)) / sigma)


def severity_from_sigma(effect: float) -> str:
    """Map an effect size in sigma units to ``low`` / ``medium`` / ``high``."""
    e = abs(effect)
    if e < SEVERITY_LOW:
        return "low"
    if e < SEVERITY_HIGH:
        return "medium"
    return "high"


# --------------------------------------------------------------------------- #
# Channel propagators
# --------------------------------------------------------------------------- #


def apply_power(df: pd.DataFrame, pos, p_new, q_new) -> None:
    """Set P and Q, then rebuild S, I, PF and E (identities hold)."""
    pos = np.asarray(pos)
    v = df.loc[pos, "voltage_v"].to_numpy(dtype=float)
    recompute_dependents(df, pos, np.asarray(p_new, dtype=float), np.asarray(q_new, dtype=float), v)


def apply_current(df: pd.DataFrame, pos, i_new) -> None:
    """Set I, then rebuild S = V*I/1000, P = PF*S, Q and E."""
    pos = np.asarray(pos)
    v = df.loc[pos, "voltage_v"].to_numpy(dtype=float)
    pf = df.loc[pos, "power_factor"].to_numpy(dtype=float)
    i = np.asarray(i_new, dtype=float)
    s = v * i / 1000.0
    p = pf * s
    q = np.sqrt(np.maximum(s**2 - p**2, 0.0))
    recompute_dependents(df, pos, p, q, v)


def apply_voltage_keep_power(df: pd.DataFrame, pos, v_new) -> None:
    """Set V but keep P and Q: S is unchanged, I = S/V moves (identities hold)."""
    pos = np.asarray(pos)
    p = df.loc[pos, "active_power_kw"].to_numpy(dtype=float)
    q = df.loc[pos, "reactive_power_kvar"].to_numpy(dtype=float)
    df.loc[pos, "voltage_v"] = np.asarray(v_new, dtype=float)
    recompute_dependents(df, pos, p, q, np.asarray(v_new, dtype=float))


def copy_block(df: pd.DataFrame, pos, source_pos) -> None:
    """Copy every electrical channel from ``source_pos`` onto ``pos`` (consistent)."""
    cols = [c for c in ELECTRICAL_COLUMNS if c in df.columns]
    df.loc[np.asarray(pos), cols] = df.loc[np.asarray(source_pos), cols].to_numpy()


def null_block(df: pd.DataFrame, pos) -> None:
    """Blank every electrical channel (dropout / communications gap)."""
    cols = [c for c in ELECTRICAL_COLUMNS if c in df.columns]
    df.loc[np.asarray(pos), cols] = np.nan


def bias_submeter(df: pd.DataFrame, pos, sub: str, gain: float) -> None:
    """Multiply one sub-meter channel by ``(1 + gain)`` (breaks the energy balance)."""
    df.loc[np.asarray(pos), sub] = df.loc[np.asarray(pos), sub].to_numpy(dtype=float) * (1.0 + gain)


def scaled_reactive(p, p_new, q) -> np.ndarray:
    """Reactive power that keeps ``S > 0`` when ``P`` is clipped to zero.

    Scaling ``Q`` with ``P`` keeps the power factor stable, but if ``P`` hits 0
    both would vanish, giving ``S = 0`` where the generator's ``PF = 1`` convention
    makes the identity residual read 1. In that case the original reactive power
    is kept, so ``S = Q > 0`` and ``PF = 0`` as expected.
    """
    p = np.asarray(p, dtype=float)
    p_new = np.asarray(p_new, dtype=float)
    q = np.asarray(q, dtype=float)
    ratio = np.where(p > 0, p_new / np.where(p > 0, p, 1.0), 1.0)
    return np.where(p_new > 0, q * ratio, q)


# --------------------------------------------------------------------------- #
# Sampling helpers
# --------------------------------------------------------------------------- #


def window_ok(quality: np.ndarray, occupied: np.ndarray, length: int, allowed=None) -> np.ndarray:
    """Boolean mask of start positions whose ``length``-window is clean and free."""
    n = len(quality)
    if length > n:
        return np.zeros(0, dtype=int)
    if allowed is None:
        allowed = np.ones(n, dtype=bool)
    q = (quality.astype(bool) & allowed.astype(bool)).astype(np.int64)
    o = occupied.astype(np.int64)
    ones = np.ones(length, dtype=np.int64)
    valid = np.convolve(q, ones, mode="valid") == length
    free = np.convolve(o, ones, mode="valid") == 0
    return np.flatnonzero(valid & free)


def draw_start(
    quality: np.ndarray, occupied: np.ndarray, length: int, rng, allowed=None
) -> int | None:
    """Pick a random valid start index, or ``None`` if none exists."""
    starts = window_ok(quality, occupied, length, allowed)
    if len(starts) == 0:
        return None
    return int(rng.choice(starts))


def mark(occupied: np.ndarray, start: int, length: int) -> None:
    occupied[start : start + length] = True


def parse_duration(text: str) -> int:
    """Convert a config duration (``30min`` / ``12h`` / ``7d``) to grid intervals."""
    text = text.strip()
    if text.endswith("min"):
        return max(1, round(int(text[:-3]) / 30))
    if text.endswith("h"):
        return int(text[:-1]) * 2
    if text.endswith("d"):
        return int(text[:-1]) * 48
    raise ValueError(f"unrecognised duration: {text!r}")


def robust_sigma(values: pd.Series) -> float:
    """1.4826 * MAD, the robust scale used throughout the project."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    mad = np.median(np.abs(v - np.median(v)))
    return float(1.4826 * mad)


# --------------------------------------------------------------------------- #
# Grid + context
# --------------------------------------------------------------------------- #


@dataclass
class Grid:
    """Positional view of the clean baseline, one array per household.

    The baseline has a full regular grid and every household has the same
    timestamps, so an integer position ``i`` maps to the same instant for every
    household -- which is what makes multi-household physical events easy.
    """

    df: pd.DataFrame
    times: np.ndarray
    n: int
    quality: dict[str, np.ndarray]
    occupied: dict[str, np.ndarray]
    households: list[str]
    rows: dict[str, np.ndarray]

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> Grid:
        out = df.sort_values([HOUSEHOLD_COL, UTC_COL]).reset_index(drop=True)
        times = np.sort(out[UTC_COL].unique())
        quality: dict[str, np.ndarray] = {}
        occupied: dict[str, np.ndarray] = {}
        rows: dict[str, np.ndarray] = {}
        households: list[str] = []
        for hh, group in out.groupby(HOUSEHOLD_COL, sort=True):
            if len(group) != len(times):
                raise ValueError(f"{hh} has {len(group)} rows, expected {len(times)}")
            households.append(hh)
            quality[hh] = group["quality_mask"].to_numpy().astype(bool)
            occupied[hh] = np.zeros(len(times), dtype=bool)
            rows[hh] = group.index.to_numpy()
        return cls(out, times, len(times), quality, occupied, households, rows)


@dataclass
class InjectContext:
    sigma_p: dict[str, float]
    sigma_v: dict[str, float]
    profile_p: dict[str, pd.Series]
    settings: object
    injector_version: str = "0.1.0"


def build_context(df: pd.DataFrame, settings) -> InjectContext:
    """Per-household robust scales (P, V) and hour-of-week load profile.

    ``profile_p[hh]`` is a Series indexed by the baseline's row labels, so callers
    can index it with the same positional labels used elsewhere.
    """
    local = df[UTC_COL].dt.tz_convert(None) - pd.Timedelta(hours=6)
    how = local.dt.dayofweek * 48 + local.dt.hour * 2 + local.dt.minute // 30
    work = df.assign(_how=how)

    sigma_p: dict[str, float] = {}
    sigma_v: dict[str, float] = {}
    profile_p: dict[str, pd.Series] = {}
    for hh, group in work.groupby(HOUSEHOLD_COL, sort=True):
        good = group[group["quality_mask"] == 1]
        sigma_p[hh] = robust_sigma(good["active_power_kw"])
        sigma_v[hh] = robust_sigma(good["voltage_v"])
        median = good.groupby("_how")["active_power_kw"].median()
        profile_p[hh] = group["_how"].map(median).fillna(good["active_power_kw"].median())
    return InjectContext(sigma_p, sigma_v, profile_p, settings)


def measure_severity(
    before_p, after_p, before_v, after_v, sigma_p: float, sigma_v: float
) -> tuple[str, dict]:
    """Severity label plus the measured effects, from P and V shifts."""
    eff_p = effect_in_sigma(before_p, after_p, sigma_p)
    eff_v = effect_in_sigma(before_v, after_v, sigma_v)
    severity = severity_from_sigma(max(eff_p, eff_v))
    return severity, {"effect_p_sigma": round(eff_p, 3), "effect_v_sigma": round(eff_v, 3)}

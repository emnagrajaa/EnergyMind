"""Step 2 - physics validation utilities.

The generated dataset satisfies its electrical identities almost exactly
(finding F1): ``S = sqrt(P^2 + Q^2)``, ``PF = P/S``, ``S = V*I/1000`` and
``E = 0.5*P``. That makes a physics check an *exact* detector, and it means any
imputation or injection must keep the identities consistent.

This module provides:

* :func:`physics_residuals` - relative residuals of the identities;
* :func:`recompute_dependents` - rebuild dependent channels from root values;
* :func:`bounds_flags` - plausibility flags for impossible values;
* :func:`flat_run_length` - length of the flat run ending at each row (finding F9).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energymind.config import DataConfig

#: Guard against divide-by-zero in the relative residuals.
EPS = 1e-6
#: The four measured sub-meter categories (kWh per 30 min).
SUB_COLS = [
    "sub1_heating_kwh",
    "sub2_cooling_kwh",
    "sub3_lighting_kwh",
    "sub4_appliance_kwh",
]


def physics_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """Relative residuals of the electrical identities.

    All four are ~0 when the data is internally consistent and grow when a single
    channel is edited (a naive injection), which is exactly the signal the physics
    detector relies on.
    """
    p = df["active_power_kw"].to_numpy(dtype=float)
    q = df["reactive_power_kvar"].to_numpy(dtype=float)
    s = df["apparent_power_kva"].to_numpy(dtype=float)
    v = df["voltage_v"].to_numpy(dtype=float)
    i = df["current_a"].to_numpy(dtype=float)
    e = df["total_energy_kwh"].to_numpy(dtype=float)
    pf = df["power_factor"].to_numpy(dtype=float)
    subs = df[SUB_COLS].sum(axis=1).to_numpy(dtype=float)
    ev = df["ev_load_kw"].to_numpy(dtype=float)

    return pd.DataFrame(
        {
            "r_S": (s - np.sqrt(p**2 + q**2)) / (np.abs(s) + EPS),
            "r_VI": (s - v * i / 1000.0) / (np.abs(s) + EPS),
            "r_E": (e - 0.5 * p) / (np.abs(e) + EPS),
            "r_PF": pf - p / np.maximum(s, EPS),
            "r_sub": (subs + 0.5 * ev - e) / (np.abs(e) + EPS),
        },
        index=df.index,
    )


def recompute_dependents(
    df: pd.DataFrame,
    index,
    p,
    q,
    v,
) -> None:
    """Rebuild ``S, I, PF, E`` from root values ``P, Q, V`` **in place**.

    Used by imputation (Step 2) and by the consistent injection variants
    (Step 5), so a rebuilt row keeps the identities ``S=sqrt(P^2+Q^2)``,
    ``I=S*1000/V``, ``PF=P/S`` and ``E=0.5*P``.
    """
    p = np.asarray([p] if np.isscalar(p) else p, dtype=float)
    q = np.asarray([q] if np.isscalar(q) else q, dtype=float)
    v = np.asarray([v] if np.isscalar(v) else v, dtype=float)
    s = np.sqrt(p**2 + q**2)
    safe_v = np.where(v > 0, v, np.nan)
    safe_s = np.where(s > 0, s, np.nan)

    df.loc[index, "active_power_kw"] = p
    df.loc[index, "reactive_power_kvar"] = q
    df.loc[index, "apparent_power_kva"] = s
    df.loc[index, "current_a"] = s * 1000.0 / safe_v
    df.loc[index, "power_factor"] = np.where(s > 0, p / safe_s, 1.0)
    df.loc[index, "total_energy_kwh"] = 0.5 * p


def bounds_flags(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Boolean plausibility flags: ``True`` means the value is impossible.

    NaNs (masked or missing rows) yield ``False`` -- they are handled separately
    by the quality mask, not by these range checks.
    """
    b = cfg.bounds
    v = df["voltage_v"]
    f = df["frequency_hz"]
    pf = df["power_factor"]
    p = df["active_power_kw"]
    e = df["total_energy_kwh"]
    subs = df[SUB_COLS].sum(axis=1) + 0.5 * df["ev_load_kw"]

    return pd.DataFrame(
        {
            "flag_voltage": v.lt(b.voltage_v[0]) | v.gt(b.voltage_v[1]),
            "flag_frequency": f.lt(b.frequency_hz[0]) | f.gt(b.frequency_hz[1]),
            "flag_power_factor": pf.lt(b.power_factor[0]) | pf.gt(b.power_factor[1]),
            "flag_active_power_negative": p.lt(b.active_power_kw[0]),
            "flag_submeter_over_total": subs.gt(e * (1.0 + b.sub_meter_tolerance)),
        },
        index=df.index,
    )


def flat_run_length(series: pd.Series) -> pd.Series:
    """Length of the run of identical consecutive values ending at each row.

    Finding F9: natural flat runs of active power reach 16 intervals (8 h), so a
    stuck-at shorter than that cannot be found by flatness alone.
    """
    s = pd.Series(series).reset_index(drop=True)
    run_id = s.ne(s.shift()).cumsum()
    return (s.groupby(run_id).cumcount() + 1).astype("int32").set_axis(series.index)

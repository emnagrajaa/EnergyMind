"""Physical grid events (blueprint Section 7): coherent peak, outage, voltage sag.

These are the key discriminator: they are applied **simultaneously across many
households**, so a single-meter fault and a genuine grid event look identical on
one meter and differ only in neighbourhood coherence and physical consistency.
All are consistent by construction and have ``scope = neighborhood``.
"""

from __future__ import annotations

import numpy as np

from energymind.inject.base import (
    apply_power,
    apply_voltage_keep_power,
    measure_severity,
    null_block,
)

#: Physical events always span at least this share of households.
MIN_HOUSEHOLD_SHARE = 0.4


def _aggregate(df, targets, column: str) -> np.ndarray:
    return np.concatenate([df.loc[pos, column].to_numpy(dtype=float) for _, pos in targets])


def _severity(df, targets, before_p, before_v, ctx) -> dict:
    after_p = _aggregate(df, targets, "active_power_kw")
    after_v = _aggregate(df, targets, "voltage_v")
    sigma_p = float(np.median([ctx.sigma_p[hh] for hh, _ in targets]))
    sigma_v = float(np.median([ctx.sigma_v[hh] for hh, _ in targets]))
    sev, meta = measure_severity(before_p, after_p, before_v, after_v, sigma_p, sigma_v)
    return {"severity": sev, "scope": "neighborhood", **meta}


def coherent_peak(df, targets, params, variant, rng, ctx) -> dict:
    """A demand peak felt by many households at once (``P *= 1 + p*shape``)."""
    before_p = _aggregate(df, targets, "active_power_kw")
    before_v = _aggregate(df, targets, "voltage_v")
    p = params["p"]
    for _, pos in targets:
        length = len(pos)
        frac = np.arange(length, dtype=float) / max(length - 1, 1)
        shape = np.sin(np.pi * frac)  # 0 at the edges, 1 in the middle
        old = df.loc[pos, "active_power_kw"].to_numpy(dtype=float)
        p_new = old * (1.0 + p * shape)
        q = df.loc[pos, "reactive_power_kvar"].to_numpy(dtype=float)
        ratio = np.where(old > 0, p_new / np.where(old > 0, old, 1.0), 1.0)
        apply_power(df, pos, p_new, q * ratio)
    return _severity(df, targets, before_p, before_v, ctx)


def neighborhood_outage(df, targets, params, variant, rng, ctx) -> dict:
    """Every channel to zero on every household (a neighbourhood outage)."""
    before_p = _aggregate(df, targets, "active_power_kw")
    before_v = _aggregate(df, targets, "voltage_v")
    for _, pos in targets:
        null_block(df, pos)
        df.loc[pos, "voltage_v"] = 0.0
    return _severity(df, targets, before_p, before_v, ctx)


def voltage_sag(df, targets, params, variant, rng, ctx) -> dict:
    """Voltage dips while power is unchanged, so current rises (``V *= factor``)."""
    before_p = _aggregate(df, targets, "active_power_kw")
    before_v = _aggregate(df, targets, "voltage_v")
    factor = params["factor"]
    for _, pos in targets:
        v_new = df.loc[pos, "voltage_v"].to_numpy(dtype=float) * factor
        apply_voltage_keep_power(df, pos, v_new)
    return _severity(df, targets, before_p, before_v, ctx)

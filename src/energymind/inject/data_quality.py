"""Data-quality anomaly types (blueprint Section 7).

Each function edits ``df`` in place at the positional window ``pos`` and returns
the event severity and its measured effect sizes. ``variant`` selects naive
(single-channel edit, identities break) or consistent (root edit + recompute).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energymind.inject.base import (
    apply_current,
    apply_power,
    apply_voltage_keep_power,
    bias_submeter,
    copy_block,
    measure_severity,
    null_block,
    scaled_reactive,
)


def _before(df: pd.DataFrame, pos):
    return (
        df.loc[pos, "active_power_kw"].to_numpy(dtype=float),
        df.loc[pos, "voltage_v"].to_numpy(dtype=float),
    )


def _finish(df, pos, before_p, before_v, ctx, hh):
    sev, meta = measure_severity(
        before_p,
        df.loc[pos, "active_power_kw"].to_numpy(dtype=float),
        before_v,
        df.loc[pos, "voltage_v"].to_numpy(dtype=float),
        ctx.sigma_p[hh],
        ctx.sigma_v[hh],
    )
    return {"severity": sev, **meta}


def additive_bias(df, pos, params, variant, rng, ctx, hh) -> dict:
    """``P += k * sigma_h`` (a constant offset)."""
    before_p, before_v = _before(df, pos)
    k = params["k"]
    sigma = ctx.sigma_p[hh]
    p = df.loc[pos, "active_power_kw"].to_numpy(dtype=float)
    p_new = np.clip(p + k * sigma, 0.0, None)
    if variant == "naive":
        df.loc[pos, "active_power_kw"] = p_new
    else:
        q = df.loc[pos, "reactive_power_kvar"].to_numpy(dtype=float)
        apply_power(df, pos, p_new, scaled_reactive(p, p_new, q))
    return _finish(df, pos, before_p, before_v, ctx, hh)


def gain_error(df, pos, params, variant, rng, ctx, hh) -> dict:
    """``I *= (1 + g)`` (a miscalibrated current transformer)."""
    before_p, before_v = _before(df, pos)
    i = df.loc[pos, "current_a"].to_numpy(dtype=float) * (1.0 + params["g"])
    if variant == "naive":
        df.loc[pos, "current_a"] = i
    else:
        apply_current(df, pos, i)
    return _finish(df, pos, before_p, before_v, ctx, hh)


def voltage_drift(df, pos, params, variant, rng, ctx, hh) -> dict:
    """A slow voltage ramp ``V += d * (t - t0) / L``."""
    before_p, before_v = _before(df, pos)
    length = len(pos)
    frac = np.arange(length, dtype=float) / max(length, 1)
    v_new = df.loc[pos, "voltage_v"].to_numpy(dtype=float) + params["d_v"] * frac
    if variant == "naive":
        # Edit V only: S = V*I/1000 no longer holds, so the identities break.
        df.loc[pos, "voltage_v"] = v_new
    else:
        apply_voltage_keep_power(df, pos, v_new)
    return _finish(df, pos, before_p, before_v, ctx, hh)


def stuck_at(df, pos, params, variant, rng, ctx, hh) -> dict:
    """Hold the value at ``t0`` for the whole window (already consistent)."""
    before_p, before_v = _before(df, pos)
    source = np.full(len(pos), pos[0])
    copy_block(df, pos, source)
    return _finish(df, pos, before_p, before_v, ctx, hh)


def dropout(df, pos, params, variant, rng, ctx, hh) -> dict:
    """Blank every channel (a communications gap)."""
    before_p, before_v = _before(df, pos)
    null_block(df, pos)
    return _finish(df, pos, before_p, before_v, ctx, hh)


def dead_sensor(df, pos, params, variant, rng, ctx, hh) -> dict:
    """``P = 0`` only (the flat meter's own computation is dead). Naive only."""
    before_p, before_v = _before(df, pos)
    df.loc[pos, "active_power_kw"] = 0.0
    return _finish(df, pos, before_p, before_v, ctx, hh)


def noise_burst(df, pos, params, variant, rng, ctx, hh) -> dict:
    """``P += N(0, (alpha * sigma_h)^2)`` (a short, noisy burst)."""
    before_p, before_v = _before(df, pos)
    alpha = params["alpha"]
    sigma = ctx.sigma_p[hh]
    eps = rng.normal(0.0, alpha * sigma, size=len(pos))
    p = df.loc[pos, "active_power_kw"].to_numpy(dtype=float)
    p_new = np.clip(p + eps, 0.0, None)
    if variant == "naive":
        df.loc[pos, "active_power_kw"] = p_new
    else:
        q = df.loc[pos, "reactive_power_kvar"].to_numpy(dtype=float)
        apply_power(df, pos, p_new, scaled_reactive(p, p_new, q))
    return _finish(df, pos, before_p, before_v, ctx, hh)


def unit_error(df, pos, params, variant, rng, ctx, hh) -> dict:
    """``P *= 1000`` (watts written as kilowatts). Naive only."""
    before_p, before_v = _before(df, pos)
    df.loc[pos, "active_power_kw"] = (
        df.loc[pos, "active_power_kw"].to_numpy(dtype=float) * params["factor"]
    )
    return _finish(df, pos, before_p, before_v, ctx, hh)


def sub_meter_bias(df, pos, params, variant, rng, ctx, hh) -> dict:
    """``sub2 *= (1 + g)`` (a biased cooling sub-meter). Consistent only."""
    before_p, before_v = _before(df, pos)
    bias_submeter(df, pos, "sub2_cooling_kwh", params["g"])
    return _finish(df, pos, before_p, before_v, ctx, hh)

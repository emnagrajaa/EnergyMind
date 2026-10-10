"""Adversarial anomaly types (blueprint Section 7): under-reporting, replay, mimicry.

These are deliberately *consistent*: a theft or a replay that broke the
electrical identities would be trivial to detect. They are also withheld from
train/val (``configs/injection.yaml``) so the "unknown type -> escalate" path
can be tested.
"""

from __future__ import annotations

import numpy as np

from energymind.inject.base import apply_power, copy_block, parse_duration, scaled_reactive
from energymind.inject.data_quality import _before, _finish


def under_reporting(df, pos, params, variant, rng, ctx, hh) -> dict:
    """``P *= r`` (energy theft). Consistent by scaling Q with P."""
    before_p, before_v = _before(df, pos)
    r = params["r"]
    p = df.loc[pos, "active_power_kw"].to_numpy(dtype=float)
    p_new = p * r
    if variant == "naive":
        df.loc[pos, "active_power_kw"] = p_new
    else:
        q = df.loc[pos, "reactive_power_kvar"].to_numpy(dtype=float)
        apply_power(df, pos, p_new, scaled_reactive(p, p_new, q))
    return _finish(df, pos, before_p, before_v, ctx, hh)


def replay(df, pos, params, variant, rng, ctx, hh) -> dict:
    """Copy the same window from ``lag`` earlier (a replayed day)."""
    before_p, before_v = _before(df, pos)
    lag = parse_duration(str(params["lag"]))
    source = np.asarray(pos) - lag
    copy_block(df, pos, source)
    return _finish(df, pos, before_p, before_v, ctx, hh)


def mimicry(df, pos, params, variant, rng, ctx, hh) -> dict:
    """Replace the window with the household's typical profile plus small noise."""
    before_p, before_v = _before(df, pos)
    sigma = ctx.sigma_p[hh]
    noise = rng.normal(0.0, params["noise"] * sigma, size=len(pos))
    p = df.loc[pos, "active_power_kw"].to_numpy(dtype=float)
    p_new = np.clip(ctx.profile_p[hh].loc[np.asarray(pos)].to_numpy(dtype=float) + noise, 0.0, None)
    q = df.loc[pos, "reactive_power_kvar"].to_numpy(dtype=float)
    apply_power(df, pos, p_new, scaled_reactive(p, p_new, q))
    return _finish(df, pos, before_p, before_v, ctx, hh)

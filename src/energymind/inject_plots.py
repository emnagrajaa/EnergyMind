"""Injection gallery figures (blueprint Step 5 / notebook 03).

Styled after ``eda_plots``: same paper rcParams, Okabe-Ito palette, PNG + PDF at
300 dpi, bold ``(a)`` panel labels. The goal is to make the *naive vs consistent*
distinction visible: a naive edit leaves the electrical identities broken, a
consistent edit does not.

Build everything with::

    python -m energymind.inject_plots --prevalence 0.05 --seed 1
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from energymind import clean, inject, io
from energymind import eda_plots as base
from energymind.config import Settings, load_settings
from energymind.validate import physics_residuals

BLUE, ORANGE, GREEN, VERM = base.BLUE, base.ORANGE, base.GREEN, base.VERM
SKY, PURPLE, BLACK, GREY = base.SKY, base.PURPLE, base.BLACK, base.GREY
W2 = base.W2

ORIGIN_COLORS = {"data_quality": BLUE, "adversarial": VERM, "physical_event": GREEN}
VARIANT_ORDER = ["naive", "consistent"]
PAD_H = 3  # hours of context shown around an event
TOL = 1e-2  # identity tolerance


def _window(baseline, injected, event, pad_h: int = PAD_H):
    """Baseline and injected active-power series around a single event."""
    hh = event["household_id"].split(",")[0]
    a = event["t_start"] - pd.Timedelta(hours=pad_h)
    b = event["t_end"] + pd.Timedelta(hours=pad_h)
    mb = (baseline["household_id"] == hh) & baseline["ts_utc"].between(a, b)
    mi = (injected["household_id"] == hh) & injected["ts_utc"].between(a, b)
    return baseline.loc[mb], injected.loc[mi]


def _pick(labels: pd.DataFrame, name: str, variant: str):
    sub = labels[(labels["type"] == name) & (labels["variant"] == variant)]
    return None if sub.empty else sub.iloc[0]


def fig_injection_overview(labels: pd.DataFrame, prevalence: float, out_dir=None):
    """Event counts by type/origin, severity mix and variant/scope."""
    fig = base._fig(W2, 4.6)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0])

    ax = fig.add_subplot(gs[0, 0])
    counts = labels["type"].value_counts().sort_values()
    colors = [
        ORIGIN_COLORS[labels.loc[labels["type"] == t, "origin"].iloc[0]] for t in counts.index
    ]
    ax.barh(counts.index, counts.to_numpy(), color=colors, height=0.75)
    base._label(ax, "(a)")
    base._sub(ax, "events by type")
    ax.set_xlabel("count")
    handles = [plt.Line2D([], [], color=c, lw=4) for c in ORIGIN_COLORS.values()]
    ax.legend(handles, list(ORIGIN_COLORS), loc="lower right", fontsize=6)

    ax = fig.add_subplot(gs[0, 1])
    order = ["low", "medium", "high"]
    sev = labels["severity"].value_counts().reindex(order).fillna(0)
    ax.bar(sev.index, sev.to_numpy(), color=[SKY, ORANGE, VERM])
    base._label(ax, "(b)")
    base._sub(ax, "severity (effect size in sigma)")
    ax.set_ylabel("events")

    ax = fig.add_subplot(gs[1, 0])
    variant = labels["variant"].value_counts().reindex(VARIANT_ORDER).fillna(0)
    axios = ax.twinx()
    scope = labels["scope"].value_counts()
    x = np.arange(2)
    ax.bar(x - 0.2, variant.to_numpy(), width=0.4, color=BLUE, label="variant")
    axios.bar(
        x + 0.2,
        scope.reindex(["single", "neighborhood"]).fillna(0).to_numpy(),
        width=0.4,
        color=GREEN,
    )
    ax.set_xticks(x, ["naive", "consistent"])
    base._label(ax, "(c)")
    base._sub(ax, "variant (blue) and scope (green)")
    ax.set_ylabel("events")
    axios.set_ylabel("neighbourhood events")

    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    lines = [
        f"requested prevalence : {prevalence:.0%}",
        f"events               : {len(labels)}",
        f"origins              : {labels['origin'].nunique()}",
        f"median length        : {labels['t_end'].sub(labels['t_start']).median()}",
        f"scope neighbourhood  : {(labels['scope'] == 'neighborhood').sum()}",
    ]
    ax.text(0.0, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=7)
    base._label(ax, "(d)")
    return base._finish(fig, out_dir, "injection_overview")


def fig_type_gallery(baseline, injected, labels, out_dir=None):
    """One example per type x variant: active power (baseline vs injected)."""
    types = sorted(labels["type"].unique())
    variants = [v for v in VARIANT_ORDER if v in set(labels["variant"])]
    fig, axes = plt.subplots(
        len(types),
        len(variants),
        figsize=(W2, 0.72 * len(types)),
        layout="constrained",
        squeeze=False,
    )
    for row, name in enumerate(types):
        origin = labels.loc[labels["type"] == name, "origin"].iloc[0]
        for col, variant in enumerate(variants):
            ax = axes[row][col]
            event = _pick(labels, name, variant)
            if event is None:
                ax.axis("off")
                continue
            b, i = _window(baseline, injected, event)
            ax.plot(b["ts_utc"], b["active_power_kw"], color=GREY, lw=0.8, label="baseline")
            ax.plot(i["ts_utc"], i["active_power_kw"], color=ORIGIN_COLORS[origin], lw=1.0)
            ax.axvspan(event["t_start"], event["t_end"], color=ORIGIN_COLORS[origin], alpha=0.12)
            if col == 0:
                ax.set_ylabel(f"{name}\n[ {origin[:3]} ]", fontsize=6)
            ax.set_title(f"{variant} | sev={event['severity']}", fontsize=6.5, style="italic")
            ax.tick_params(labelbottom=False)
    return base._finish(fig, out_dir, "injection_type_gallery")


def fig_residual_gallery(baseline, injected, labels, out_dir=None):
    """Same layout, but the max identity residual: naive breaks, consistent holds."""
    types = sorted(labels["type"].unique())
    variants = [v for v in VARIANT_ORDER if v in set(labels["variant"])]
    resid = physics_residuals(injected)[["r_S", "r_VI", "r_E", "r_PF"]].abs().max(axis=1)
    injected = injected.assign(_resid=resid.to_numpy())

    fig, axes = plt.subplots(
        len(types),
        len(variants),
        figsize=(W2, 0.72 * len(types)),
        layout="constrained",
        squeeze=False,
    )
    for row, name in enumerate(types):
        for col, variant in enumerate(variants):
            ax = axes[row][col]
            event = _pick(labels, name, variant)
            if event is None:
                ax.axis("off")
                continue
            hh = event["household_id"].split(",")[0]
            m = (injected["household_id"] == hh) & injected["ts_utc"].between(
                event["t_start"] - pd.Timedelta(hours=PAD_H),
                event["t_end"] + pd.Timedelta(hours=PAD_H),
            )
            block = injected.loc[m]
            ax.semilogy(block["ts_utc"], block["_resid"].clip(lower=1e-12), color=BLACK, lw=0.8)
            ax.axhline(TOL, color=VERM, lw=0.8, ls="--")
            ax.axvspan(event["t_start"], event["t_end"], color=GREY, alpha=0.12)
            if col == 0:
                ax.set_ylabel(name, fontsize=6)
            ax.set_title(variant, fontsize=6.5, style="italic")
            ax.tick_params(labelbottom=False)
    base._finish(fig, out_dir, "injection_residual_gallery")
    return fig


def fig_variant_comparison(baseline, injected, labels, name: str, out_dir=None):
    """Detailed P / V / identity residual for one type, naive vs consistent."""
    fig = base._fig(W2, 3.4)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.1])
    resid = physics_residuals(injected)[["r_S", "r_VI", "r_E", "r_PF"]].abs().max(axis=1)
    injected = injected.assign(_resid=resid.to_numpy())
    origin = labels.loc[labels["type"] == name, "origin"].iloc[0]

    for col, variant in enumerate(VARIANT_ORDER):
        event = _pick(labels, name, variant)
        ax = fig.add_subplot(gs[0, col])
        if event is not None:
            b, i = _window(baseline, injected, event)
            ax.plot(b["ts_utc"], b["active_power_kw"], color=GREY, lw=0.8)
            ax.plot(i["ts_utc"], i["active_power_kw"], color=ORIGIN_COLORS[origin], lw=1.1)
            ax.axvspan(event["t_start"], event["t_end"], color=ORIGIN_COLORS[origin], alpha=0.12)
        base._label(ax, f"({chr(97 + col)})")
        base._sub(ax, f"{variant}: active power")
        ax.tick_params(labelbottom=False)

        ax = fig.add_subplot(gs[1, col])
        if event is not None:
            b, i = _window(baseline, injected, event)
            ax.plot(b["ts_utc"], b["voltage_v"], color=GREY, lw=0.8)
            ax.plot(i["ts_utc"], i["voltage_v"], color=ORANGE, lw=1.1)
            ax.axvspan(event["t_start"], event["t_end"], color=ORANGE, alpha=0.12)
        base._label(ax, f"({chr(100 + col)})")
        base._sub(ax, "voltage")
        ax.tick_params(labelbottom=False)

    ax = fig.add_subplot(gs[:, 2])
    for variant, color in zip(VARIANT_ORDER, [BLUE, VERM], strict=False):
        event = _pick(labels, name, variant)
        if event is None:
            continue
        hh = event["household_id"].split(",")[0]
        m = (injected["household_id"] == hh) & injected["ts_utc"].between(
            event["t_start"] - pd.Timedelta(hours=PAD_H),
            event["t_end"] + pd.Timedelta(hours=PAD_H),
        )
        block = injected.loc[m]
        ax.semilogy(
            block["ts_utc"], block["_resid"].clip(lower=1e-12), color=color, lw=1.0, label=variant
        )
    ax.axhline(TOL, color=BLACK, lw=0.8, ls="--", label="tolerance")
    ax.legend(loc="lower right")
    base._label(ax, "(f)")
    base._sub(ax, "max identity residual")
    base._finish(fig, out_dir, f"injection_variant_{name}")
    return fig


def generate_all(
    baseline, injected, labels, prevalence: float, seed: int, out_dir=None
) -> list[str]:
    """Write every injection figure; return their names."""
    plt.close("all")
    names = ["injection_overview", "injection_type_gallery", "injection_residual_gallery"]
    fig_injection_overview(labels, prevalence, out_dir)
    fig_type_gallery(baseline, injected, labels, out_dir)
    fig_residual_gallery(baseline, injected, labels, out_dir)
    for name in ["additive_bias", "gain_error", "voltage_drift", "noise_burst", "stuck_at"]:
        if name in set(labels["type"]):
            fig_variant_comparison(baseline, injected, labels, name, out_dir)
            names.append(f"injection_variant_{name}")
    plt.close("all")
    return names


def run(settings: Settings | None = None, *, prevalence: float = 0.05, seed: int = 1, out_dir=None):
    """Inject once and build the gallery figures."""
    settings = settings or load_settings()
    baseline = clean.load_baseline(settings)
    injected, labels = inject.inject_all(baseline, settings, prevalence, seed)
    figures = io.resolve(out_dir) if out_dir else io.resolve("results/figures")
    return generate_all(baseline, injected, labels, prevalence, seed, figures), labels, injected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Step 5: injection gallery figures.")
    parser.add_argument("--configs-dir", default=None)
    parser.add_argument("--prevalence", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)

    settings = load_settings(args.configs_dir) if args.configs_dir else load_settings()
    names, labels, _ = run(settings, prevalence=args.prevalence, seed=args.seed)
    print("Injection gallery complete")
    print(f"  events  : {len(labels)}")
    for name in names:
        print(f"  figure  : {name}")
    print(f"  figures -> {io.resolve('results/figures')}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

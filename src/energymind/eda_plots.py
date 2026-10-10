# ruff: noqa: E501  (long caption / label strings)

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from energymind import io
from energymind.validate import SUB_COLS

HOUSEHOLD_COL = io.HOUSEHOLD_COL
UTC_COL = io.UTC_COL

W1, W2 = 3.5, 7.16  # IEEE single / double column width (inches)

# Okabe-Ito
BLUE, ORANGE, GREEN, VERM = "#0072B2", "#E69F00", "#009E73", "#D55E00"
SKY, PURPLE, YELLOW, BLACK, GREY = "#56B4E9", "#CC79A7", "#F0E442", "#000000", "#7f7f7f"

TYPE_ORDER = [
    "apartment_small",
    "apartment_large",
    "house_small",
    "house_large",
    "commercial_small",
]
TYPE_COLORS = {
    "apartment_small": BLUE,
    "apartment_large": SKY,
    "house_small": GREEN,
    "house_large": ORANGE,
    "commercial_small": VERM,
}
TYPE_LABELS = {
    "apartment_small": "apt. small",
    "apartment_large": "apt. large",
    "house_small": "house small",
    "house_large": "house large",
    "commercial_small": "commercial",
}
SEASON_ORDER = ["winter", "spring", "summer", "autumn"]
SEASON_COLORS = {"winter": BLUE, "spring": GREEN, "summer": VERM, "autumn": ORANGE}
SPLIT_COLORS = {"train": "#d9e8f5", "val": "#fdebc8", "test": "#f8d5cc"}
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
CORR_LABELS = {
    "active_power_kw": "P",
    "p_ramp": "ramp",
    "p_roll_mean_2h": "mean 2h",
    "p_roll_mean_24h": "mean 24h",
    "p_roll_mean_7d": "mean 7d",
    "p_roll_std_24h": "std 24h",
    "base_load_24h": "base load",
    "load_factor_24h": "load factor",
    "p_resid_z": "resid. z",
    "unmetered_share": "unmetered",
    "sub1_share": "heating sh.",
    "sub4_share": "appliance sh.",
    "neigh_share": "neigh. share",
}


# --------------------------------------------------------------------------- #
# Style and helpers
# --------------------------------------------------------------------------- #


def apply_style() -> None:
    """Install the paper style (idempotent)."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "axes.titleweight": "normal",
            "axes.titlelocation": "left",
            "axes.titlepad": 4,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "lines.linewidth": 1.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.4,
            "axes.axisbelow": True,
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


apply_style()


def _fig(width: float, height: float, **kw) -> plt.Figure:
    return plt.figure(figsize=(width, height), layout="constrained", **kw)


def _label(ax: plt.Axes, text: str, dx: float = -0.02, dy: float = 1.02) -> None:
    """Bold panel label such as ``(a)`` just above the top-left corner."""
    ax.text(
        dx,
        dy,
        text,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=8.5,
        ha="right",
        va="bottom",
    )


def _sub(ax: plt.Axes, text: str) -> None:
    ax.set_title(text, fontsize=7.5, style="italic")


def _finish(fig: plt.Figure, out_dir: Path | str | None, name: str) -> plt.Figure:
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f"{name}.png")
        fig.savefig(out / f"{name}.pdf")
    return fig


def _short(hh: str) -> str:
    return hh.split("_")[1]


def _hh_meta(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "household_type",
        "has_solar",
        "has_battery",
        "has_ev",
        "size_m2",
        "occupants",
        "solar_capacity_kwp",
        "battery_capacity_kwh",
        "ev_charger_kw",
    ]
    return df.drop_duplicates(HOUSEHOLD_COL).set_index(HOUSEHOLD_COL)[cols].sort_index()


def _binned(ax, x, y, bins, color=BLACK, label=None, band=True):
    """Binned mean (line) with inter-quartile band."""
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    d["b"] = pd.cut(d["x"], bins)
    g = d.groupby("b", observed=True).agg(
        x=("x", "mean"),
        m=("y", "mean"),
        lo=("y", lambda s: s.quantile(0.25)),
        hi=("y", lambda s: s.quantile(0.75)),
        n=("y", "size"),
    )
    g = g[g["n"] >= 30]
    if band:
        ax.fill_between(g["x"], g["lo"], g["hi"], color=color, alpha=0.18, lw=0)
    ax.plot(g["x"], g["m"], color=color, lw=1.4, label=label)
    return g


def _type_legend(ax, loc="upper right", ncol=1, **kw):
    handles = [
        Line2D([0], [0], color=TYPE_COLORS[t], lw=1.6, label=TYPE_LABELS[t]) for t in TYPE_ORDER
    ]
    ax.legend(handles=handles, loc=loc, ncol=ncol, **kw)


def _der_tag(meta: pd.DataFrame, hh: str) -> str:
    """Compact DER tag: S = PV, B = battery, E = EV ('-' when none)."""
    tag = "".join(
        c
        for c, f in (("S", "has_solar"), ("B", "has_battery"), ("E", "has_ev"))
        if meta.loc[hh, f] == 1
    )
    return tag or "-"


def _row_status(df: pd.DataFrame) -> pd.Series:
    """Mutually exclusive row status, highest-priority first."""
    s = pd.Series("clean", index=df.index, dtype="object")
    s[df["quality_mask"].eq(0)] = "other excluded"
    s[df[[c for c in df.columns if c.startswith("flag_")]].any(axis=1)] = "sub-meter flag"
    s[df["provided_anomaly"].eq(1)] = "provided anomaly (unfilled)"
    s[df["provided_anomaly"].eq(1) & df["is_imputed"].eq(1)] = "provided anomaly (imputed)"
    s[df["is_missing_record"]] = "missing record"
    return s


STATUS_ORDER = [
    "clean",
    "sub-meter flag",
    "other excluded",
    "provided anomaly (imputed)",
    "provided anomaly (unfilled)",
    "missing record",
]
STATUS_COLORS = {
    "clean": "#cfe3d8",
    "sub-meter flag": "#b9c4d3",
    "other excluded": "#e0e0e0",
    "provided anomaly (imputed)": ORANGE,
    "provided anomaly (unfilled)": VERM,
    "missing record": BLACK,
}


# --------------------------------------------------------------------------- #
# Fig 1 - dataset overview (analogue of the multi-panel PV overview)
# --------------------------------------------------------------------------- #


def fig_dataset_overview(fx: pd.DataFrame, out_dir=None) -> plt.Figure:
    meta = _hh_meta(fx)
    fig = _fig(W2, 4.5)
    gs = fig.add_gridspec(2, 3)

    # (a) row status per household
    ax = fig.add_subplot(gs[0, 0])
    st = _row_status(fx)
    tab = (
        pd.crosstab(fx[HOUSEHOLD_COL], st, normalize="index").reindex(
            columns=STATUS_ORDER, fill_value=0
        )
        * 100
    )
    left = np.zeros(len(tab))
    for c in STATUS_ORDER:
        ax.barh(range(len(tab)), tab[c], left=left, color=STATUS_COLORS[c], height=0.78, lw=0)
        left += tab[c].to_numpy()
    ax.set_yticks(range(len(tab)), [_short(h) for h in tab.index])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of records (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(
        handles=[Patch(color=STATUS_COLORS[c], label=c) for c in STATUS_ORDER],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=2,
        fontsize=6,
    )
    _sub(ax, "Record status per household")
    _label(ax, "(a)")

    # (b) hours covered per household-day
    ax = fig.add_subplot(gs[0, 1])
    cov = (~fx["is_missing_record"]).groupby([fx[HOUSEHOLD_COL], fx["date_local"]]).sum() / 2.0
    cov = cov[cov > 0]
    ax.hist(cov, bins=np.arange(0, 25, 0.5), color=BLUE, lw=0)
    ax.set_xlim(22, 24.6)
    ax.set_yscale("log")
    ax.axvline(24, color=VERM, ls="--", lw=1, label="complete day")
    ax.set_xlabel("hours covered per household-day")
    ax.set_ylabel("household-days")
    ax.legend(loc="upper left")
    n_inc = int((cov < 24).sum())
    ax.text(
        0.03, 0.5, f"{n_inc} incomplete\nof {len(cov):,} days", transform=ax.transAxes, fontsize=6.5
    )
    _sub(ax, "Temporal coverage")
    _label(ax, "(b)")

    # (c) power distribution per type
    ax = fig.add_subplot(gs[0, 2])
    bins = np.linspace(0, 20, 81)
    for t in TYPE_ORDER:
        v = fx.loc[fx["household_type"].eq(t), "active_power_kw"].dropna()
        h, _ = np.histogram(v, bins=bins, density=True)
        ax.step(bins[:-1], h, where="post", color=TYPE_COLORS[t], label=TYPE_LABELS[t])
    ax.set_yscale("log")
    ax.set_xlabel("active power (kW)")
    ax.set_ylabel("density")
    ax.set_ylim(top=40)
    ax.legend(loc="upper right", ncol=2, columnspacing=0.8)
    _sub(ax, "Active-power distribution")
    _label(ax, "(c)")

    # (d) daily energy per household, 7-day mean, by type
    ax = fig.add_subplot(gs[1, 0])
    daily = (
        fx.groupby([HOUSEHOLD_COL, "date_local"])["total_energy_kwh"]
        .sum()
        .rename("kwh")
        .reset_index()
    )
    daily["household_type"] = daily[HOUSEHOLD_COL].map(meta["household_type"])
    for t in TYPE_ORDER:
        s = daily[daily["household_type"].eq(t)].groupby("date_local")["kwh"].mean()
        ax.plot(
            s.index,
            s.rolling(7, center=True, min_periods=3).mean(),
            color=TYPE_COLORS[t],
            label=TYPE_LABELS[t],
        )
    ax.set_ylabel("daily energy (kWh), 7-day mean")
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(mpl.dates.MonthLocator(bymonth=[1, 3, 5, 7, 9, 11]))
    ax.set_xlim(daily["date_local"].min(), daily["date_local"].max())
    _sub(ax, "Daily energy by household type")
    _label(ax, "(d)")

    # (e) mean daily profile
    ax = fig.add_subplot(gs[1, 1])
    for t in TYPE_ORDER:
        s = fx[fx["household_type"].eq(t)].groupby("slot")["active_power_kw"].mean()
        ax.plot(s.index / 2, s.to_numpy(), color=TYPE_COLORS[t])
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 6))
    ax.set_xlabel("local hour of day")
    ax.set_ylabel("mean active power (kW)")
    _sub(ax, "Mean daily profile")
    _label(ax, "(e)")

    # (f) load vs temperature
    ax = fig.add_subplot(gs[1, 2])
    d = fx[["temperature_c", "active_power_kw"]].dropna()
    hb = ax.hexbin(
        d["temperature_c"],
        d["active_power_kw"],
        gridsize=40,
        bins="log",
        cmap="Greys",
        mincnt=1,
        extent=(0, 42, 0, 14),
        linewidths=0,
        rasterized=True,
    )
    _binned(
        ax, d["temperature_c"], d["active_power_kw"], np.arange(0, 44, 3), color=VERM, band=False
    )
    ax.set_xlabel("temperature (°C)")
    ax.set_ylabel("active power (kW)")
    ax.set_ylim(0, 14)
    fig.colorbar(hb, ax=ax, label="records (log)", pad=0.02, fraction=0.06)
    _sub(ax, "Load vs temperature (U-shape)")
    _label(ax, "(f)")
    return _finish(fig, out_dir, "dataset_overview")


# --------------------------------------------------------------------------- #
# Fig 2 - missing / masked / imputed record map
# --------------------------------------------------------------------------- #


def fig_missing_record_map(fx: pd.DataFrame, out_dir=None) -> plt.Figure:
    st = _row_status(fx)
    prio = {"missing record": 4, "provided anomaly (unfilled)": 3, "provided anomaly (imputed)": 2}
    code = st.map(prio).fillna(0).astype(int)
    day = code.groupby([fx[HOUSEHOLD_COL], fx["date_local"]]).max().unstack(0)
    hhs = list(day.columns)
    cmap = ListedColormap(["#f1f1f1", "#f1f1f1", ORANGE, VERM, BLACK])

    fig = _fig(W2, 2.9)
    gs = fig.add_gridspec(1, 2, width_ratios=[4.2, 1])
    ax = fig.add_subplot(gs[0])
    x = mpl.dates.date2num(day.index.to_pydatetime())
    ax.pcolormesh(
        np.append(x, x[-1] + 1),
        np.arange(len(hhs) + 1),
        day.to_numpy().T,
        cmap=cmap,
        vmin=0,
        vmax=4,
        rasterized=True,
        edgecolor="none",
    )
    ax.set_yticks(np.arange(len(hhs)) + 0.5, [_short(h) for h in hhs])
    ax.invert_yaxis()
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(mpl.dates.MonthLocator())
    ax.grid(False)
    ax.set_xlabel("date (2020)")
    ax.set_ylabel("household")
    n = st.value_counts()
    ax.legend(
        handles=[
            Patch(
                color=ORANGE,
                label=f"provided anomaly, single slot imputed ({n.get('provided anomaly (imputed)', 0)})",
            ),
            Patch(
                color=VERM,
                label=f"provided anomaly, left NaN ({n.get('provided anomaly (unfilled)', 0)})",
            ),
            Patch(color=BLACK, label=f"missing record ({n.get('missing record', 0)})"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=3,
    )
    _sub(ax, "Household-days containing a non-clean record")
    _label(ax, "(a)")

    ax = fig.add_subplot(gs[1])
    cats = ["provided anomaly (imputed)", "provided anomaly (unfilled)", "missing record"]
    tab = pd.crosstab(fx[HOUSEHOLD_COL], st).reindex(columns=cats, fill_value=0).reindex(hhs)
    left = np.zeros(len(tab))
    for c, col in zip(cats, [ORANGE, VERM, BLACK], strict=True):
        ax.barh(np.arange(len(tab)) + 0.5, tab[c], left=left, color=col, height=0.75)
        left += tab[c].to_numpy()
    ax.set_ylim(len(hhs), 0)
    ax.set_yticks([])
    ax.set_xlabel("records")
    ax.grid(axis="y", visible=False)
    _sub(ax, "Count")
    _label(ax, "(b)")
    return _finish(fig, out_dir, "missing_record_map")


# --------------------------------------------------------------------------- #
# Fig 3 - F1/F2 power identity and provided labels
# --------------------------------------------------------------------------- #


def fig_power_identity(raw: pd.DataFrame, out_dir=None) -> plt.Figure:
    d = raw.copy()
    d["derived"] = d["voltage_v"] * d["current_a"] * d["power_factor"] / 1000.0
    d["rel"] = (d["active_power_kw"] - d["derived"]).abs() / d["derived"].abs().replace(0, np.nan)
    normal = d[d["anomaly_flag"].eq(0)]
    spike = d[d["anomaly_type"].eq("spike")]
    drop = d[d["anomaly_type"].eq("dropout")]

    fig = _fig(W2, 2.9)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
    ax = fig.add_subplot(gs[0])
    s = normal.sample(min(len(normal), 25_000), random_state=0)
    ax.scatter(
        s["derived"],
        s["active_power_kw"],
        s=1.5,
        color=GREY,
        alpha=0.35,
        lw=0,
        rasterized=True,
        label="unlabelled",
    )
    ax.scatter(
        spike["derived"],
        spike["active_power_kw"],
        s=7,
        color=VERM,
        marker="^",
        lw=0,
        label=f"spike (n={len(spike)})",
    )
    ax.scatter(
        drop["derived"],
        drop["active_power_kw"],
        s=7,
        color=BLUE,
        marker="s",
        lw=0,
        label=f"dropout (n={len(drop)})",
    )
    lim = float(np.nanmax(d["derived"])) * 1.03
    ax.plot([0, lim], [0, lim], color=BLACK, ls="--", lw=0.8, label="identity")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, max(lim, float(d["active_power_kw"].max())) * 1.03)
    ax.set_xlabel(r"$V\!\cdot\! I\!\cdot\! PF/1000$ (kW)")
    ax.set_ylabel("recorded active power $P$ (kW)")
    ax.legend(loc="upper left", markerscale=2)
    _sub(ax, "P vs. the electrical identity")
    _label(ax, "(a)")

    ax = fig.add_subplot(gs[1])
    grid = np.logspace(-8, 1, 400)
    for df_, lab, col, ls in (
        (normal, "unlabelled", GREY, "-"),
        (spike, "spike", VERM, "--"),
        (drop, "dropout", BLUE, ":"),
    ):
        v = np.sort(df_["rel"].dropna().to_numpy())
        v = np.clip(v, 1e-8, None)
        ax.plot(
            grid,
            np.searchsorted(v, grid, side="right") / len(v),
            color=col,
            ls=ls,
            lw=1.5,
            label=lab,
        )
    ax.axvline(1e-2, color=BLACK, lw=0.8)
    ax.text(1.3e-2, 0.08, "tolerance $10^{-2}$", fontsize=6.5, rotation=90, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel(r"relative break $|P-VI\,PF/1000|\,/\,(VI\,PF/1000)$")
    ax.set_ylabel("empirical CDF")
    ax.legend(loc="center left")
    med = float(normal["rel"].median())
    ax.text(
        0.03,
        0.97,
        f"median unlabelled: {med:.1e}\nmedian spike: {spike['rel'].median():.2f}\n"
        f"dropout: {drop['rel'].median():.2f}",
        transform=ax.transAxes,
        fontsize=6.5,
        va="top",
    )
    _sub(ax, "A physics check separates provided labels exactly")
    _label(ax, "(b)")
    return _finish(fig, out_dir, "power_identity_scatter")


# --------------------------------------------------------------------------- #
# Fig 4 - F1 residuals per identity and per household
# --------------------------------------------------------------------------- #


def fig_physics_residuals(fx: pd.DataFrame, tol: float, out_dir=None) -> plt.Figure:
    ids = ["r_S", "r_VI", "r_E", "r_PF"]
    names = {"r_S": r"$r_S$", "r_VI": r"$r_{VI}$", "r_E": r"$r_E$", "r_PF": r"$r_{PF}$"}
    q = fx[fx["quality_mask"].eq(1)]
    fig = _fig(W2, 2.6)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1])
    ax = fig.add_subplot(gs[0])
    grid = np.linspace(-8, 0, 300)
    for r, col, ls in zip(ids, [BLUE, VERM, GREEN, ORANGE], ["-", "--", "-.", ":"], strict=True):
        v = np.sort(np.log10(np.clip(q[r].abs().to_numpy(), 1e-8, None)))
        ax.plot(
            grid,
            np.searchsorted(v, grid, side="right") / len(v),
            color=col,
            ls=ls,
            lw=1.4,
            label=names[r],
        )
    ax.axvline(np.log10(tol), color=BLACK, lw=0.8)
    ax.text(np.log10(tol) - 0.12, 0.45, f"tolerance {tol:g}", ha="right", fontsize=6.5, rotation=90)
    ax.set_xlabel(r"$\log_{10}|r|$ (exact zeros plotted at $-8$)")
    ax.set_ylabel("empirical CDF")
    ax.legend(loc="upper left")
    _sub(ax, f"Identity residuals on {len(q):,} clean records")
    _label(ax, "(a)")

    ax = fig.add_subplot(gs[1])
    m = q.groupby(HOUSEHOLD_COL)[ids].agg(lambda s: s.abs().max())
    im = ax.imshow(
        np.log10(m.clip(lower=1e-9)).to_numpy(), cmap="Blues", vmin=-6, vmax=-2, aspect="auto"
    )
    ax.set_xticks(range(4), [names[r] for r in ids])
    ax.set_yticks(range(len(m)), [_short(h) for h in m.index])
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(
                j,
                i,
                f"{m.iloc[i, j]:.0e}".replace("e-0", "e-"),
                ha="center",
                va="center",
                fontsize=5.5,
                color="white" if np.log10(max(m.iloc[i, j], 1e-9)) > -3.6 else BLACK,
            )
    ax.grid(False)
    fig.colorbar(im, ax=ax, label=r"$\log_{10}\max|r|$", pad=0.02, fraction=0.07)
    _sub(ax, "Worst-case residual per household")
    _label(ax, "(b)")
    return _finish(fig, out_dir, "physics_residuals")


# --------------------------------------------------------------------------- #
# Fig 5 - F4 electrical parameters
# --------------------------------------------------------------------------- #


def fig_electrical_parameters(fx: pd.DataFrame, bounds: dict, out_dir=None) -> plt.Figure:
    fig = _fig(W2, 2.2)
    gs = fig.add_gridspec(1, 3)
    specs = [
        ("voltage_v", "voltage (V)", BLUE, bounds.get("voltage_v"), 230, 0.5),
        ("frequency_hz", "frequency (Hz)", GREEN, bounds.get("frequency_hz"), 50, 0.01),
        ("power_factor", "power factor", ORANGE, None, None, 0.005),
    ]
    for k, (col, xl, color, bd, nominal, bw) in enumerate(specs):
        ax = fig.add_subplot(gs[k])
        v = fx[col].dropna()
        ax.hist(v, bins=np.arange(v.min(), v.max() + bw, bw), color=color, lw=0)
        if bd:
            for b in bd:
                if b is not None:
                    ax.axvline(b, color=VERM, ls="--", lw=1)
        if nominal:
            ax.axvline(nominal, color=BLACK, lw=0.8)
        ax.set_xlabel(xl)
        if k == 0:
            ax.set_ylabel("records")
        ax.text(
            0.03,
            0.97,
            f"mean {v.mean():.4g} ± {v.std():.2g}\nmin {v.min():.4g}, max {v.max():.4g}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.3,
        )
        _label(ax, f"({'abc'[k]})")
    axs = fig.axes
    _sub(axs[0], "Voltage (230 V nominal)")
    _sub(axs[1], "Frequency (50 Hz)")
    _sub(axs[2], "Power factor")
    axs[0].legend(
        handles=[
            Line2D([0], [0], color=VERM, ls="--", label="plausibility bound"),
            Line2D([0], [0], color=BLACK, label="nominal"),
        ],
        loc="center left",
    )
    pf = fx["power_factor"].dropna()
    axs[2].annotate(
        f"capped at 1.0\n({(pf >= 0.9999).mean():.1%} of records)",
        xy=(1.0, 6200),
        xytext=(0.80, 9000),
        fontsize=6.3,
        arrowprops={"arrowstyle": "-", "lw": 0.6},
    )
    return _finish(fig, out_dir, "electrical_parameters")


# --------------------------------------------------------------------------- #
# Fig 6 - F2 provided labels
# --------------------------------------------------------------------------- #


def fig_provided_labels(fx: pd.DataFrame, labels: pd.DataFrame, out_dir=None) -> plt.Figure:
    lab = labels.copy()
    local = lab[UTC_COL].dt.tz_convert(None) - pd.Timedelta(hours=6)
    lab["month"], lab["hour"] = local.dt.month, local.dt.hour
    lab = lab.sort_values([HOUSEHOLD_COL, UTC_COL])
    # run lengths (consecutive 30-min slots of the same household)
    gap = (lab[UTC_COL].diff() != pd.Timedelta(minutes=30)) | lab[HOUSEHOLD_COL].ne(
        lab[HOUSEHOLD_COL].shift()
    )
    runs = lab.groupby(gap.cumsum()).size()

    fig = _fig(W2, 2.3)
    gs = fig.add_gridspec(1, 4, width_ratios=[1.35, 1.1, 1.1, 0.8])
    cols = {"spike": VERM, "dropout": BLUE}

    ax = fig.add_subplot(gs[0])
    t = pd.crosstab(lab[HOUSEHOLD_COL], lab["anomaly_type"])
    left = np.zeros(len(t))
    for c in ("spike", "dropout"):
        ax.bar(range(len(t)), t[c], bottom=left, color=cols[c], label=c, width=0.75)
        left += t[c].to_numpy()
    ax.set_xticks(range(len(t)), [_short(h)[-3:] for h in t.index], rotation=90)
    ax.set_ylabel("provided labels")
    ax.set_xlabel("household (last 3 digits)")
    ax.legend(loc="upper right")
    _sub(ax, "Per household")
    _label(ax, "(a)")

    ax = fig.add_subplot(gs[1])
    m = pd.crosstab(lab["month"], lab["anomaly_type"]).reindex(range(1, 13), fill_value=0)
    ax.bar(m.index, m["spike"], color=cols["spike"], width=0.75)
    ax.bar(m.index, m["dropout"], bottom=m["spike"], color=cols["dropout"], width=0.75)
    ax.set_xticks(range(1, 13, 2))
    ax.set_xlabel("month")
    _sub(ax, "Per month")
    _label(ax, "(b)")

    ax = fig.add_subplot(gs[2])
    h = pd.crosstab(lab["hour"], lab["anomaly_type"]).reindex(range(24), fill_value=0)
    ax.bar(h.index, h["spike"], color=cols["spike"], width=0.75)
    ax.bar(h.index, h["dropout"], bottom=h["spike"], color=cols["dropout"], width=0.75)
    ax.set_xticks(range(0, 24, 6))
    ax.set_xlabel("local hour")
    _sub(ax, "Per hour of day")
    _label(ax, "(c)")

    ax = fig.add_subplot(gs[3])
    vc = runs.value_counts().sort_index()
    ax.bar(vc.index.astype(str), vc.to_numpy(), color=GREY)
    ax.set_yscale("log")
    ax.set_xlabel("run length (intervals)")
    for i, v in enumerate(vc.to_numpy()):
        ax.text(i, v * 1.15, str(v), ha="center", fontsize=6.5)
    _sub(ax, "Run length")
    _label(ax, "(d)")
    return _finish(fig, out_dir, "provided_labels")


# --------------------------------------------------------------------------- #
# Fig 7 - F3 leakage
# --------------------------------------------------------------------------- #


def fig_leakage(raw: pd.DataFrame, out_dir=None) -> plt.Figure:
    r = raw.copy()
    g = r.groupby("household_id")["active_power_kw"]
    r["incl"] = g.transform(lambda s: s.rolling(4, min_periods=1).mean())
    r["excl"] = g.transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    e_incl = (r["power_2h_mean"] - r["incl"]).abs()
    e_excl = (r["power_2h_mean"] - r["excl"]).abs()

    sp = r[r["anomaly_type"].eq("spike")].copy()
    sp["ratio"] = sp["active_power_kw"] / sp["power_7d_mean"]
    pick = sp.sort_values("ratio", ascending=False).iloc[0]
    i0 = r.index[
        (r["household_id"] == pick["household_id"]) & (r["timestamp"] == pick["timestamp"])
    ][0]
    w = r.loc[i0 - 10 : i0 + 10].reset_index(drop=True)
    x = np.arange(len(w)) - 10

    fig = _fig(W2, 2.7)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1])
    ax = fig.add_subplot(gs[0])
    ax.plot(x, w["active_power_kw"], color=BLACK, marker="o", ms=2.5, label="measured $P$")
    ax.plot(x, w["power_2h_mean"], color=VERM, marker="s", ms=2.5, label="provided 2 h mean")
    ax.plot(
        x, w["excl"], color=BLUE, ls="--", marker="^", ms=2.5, label="past-only 2 h mean (ours)"
    )
    ax.axvline(0, color=GREY, lw=0.8)
    ax.set_xticks(range(-10, 11, 5))
    ax.set_xlabel("intervals relative to a provided spike")
    ax.set_ylabel("kW")
    ax.legend(loc="upper left")
    _sub(ax, "The provided feature reacts at the anomaly itself")
    _label(ax, "(a)")

    ax = fig.add_subplot(gs[1])
    grid = np.logspace(-9, 1, 400)
    for e, lab, col, ls in (
        (e_incl, "trailing mean incl. $t$", VERM, "-"),
        (e_excl, "trailing mean excl. $t$", BLUE, "--"),
    ):
        v = np.sort(np.clip(e.dropna().to_numpy(), 1e-9, None))
        ax.plot(
            grid,
            np.searchsorted(v, grid, side="right") / len(v),
            color=col,
            ls=ls,
            lw=1.5,
            label=lab,
        )
    ax.set_xscale("log")
    ax.set_xlabel(r"$|$provided $-$ recomputed$|$ (kW)")
    ax.set_ylabel("empirical CDF")
    ax.legend(loc="center right")
    ax.text(
        0.03,
        0.88,
        f"median error incl.: {e_incl.median():.1e} kW\nmedian error excl.: {e_excl.median():.3f} kW",
        transform=ax.transAxes,
        va="top",
        fontsize=6.5,
    )
    _sub(ax, "Which window does the provided column use?")
    _label(ax, "(b)")
    return _finish(fig, out_dir, "leakage_f3")


# --------------------------------------------------------------------------- #
# Fig 8 - weekly profile per household type (F6 / baseline structure)
# --------------------------------------------------------------------------- #


def fig_weekly_profile(fx: pd.DataFrame, out_dir=None) -> plt.Figure:
    fig = _fig(W2, 4.3)
    gs = fig.add_gridspec(2, 3)
    vmax = float(fx.groupby(["household_type", "slot", "dow"])["active_power_kw"].mean().max())
    im = None
    for k, t in enumerate(TYPE_ORDER):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        pv = fx[fx["household_type"].eq(t)].pivot_table(
            index="slot", columns="dow", values="active_power_kw", aggfunc="mean"
        )
        im = ax.imshow(
            pv.to_numpy(),
            aspect="auto",
            origin="lower",
            cmap="viridis",
            vmin=0,
            vmax=vmax,
            extent=(-0.5, 6.5, 0, 24),
        )
        ax.set_xticks(range(7), [d[0] for d in DAYS])
        ax.set_yticks(range(0, 25, 6))
        ax.grid(False)
        if k % 3 == 0:
            ax.set_ylabel("local hour")
        _sub(ax, TYPE_LABELS[t])
        _label(ax, f"({'abcde'[k]})")
    fig.colorbar(
        im, ax=fig.axes[:5], label="mean active power (kW)", pad=0.01, fraction=0.04, shrink=0.9
    )

    ax = fig.add_subplot(gs[1, 2])
    dm = fx.groupby(["household_type", "dow"])["active_power_kw"].mean().unstack()
    rel = dm.div(dm.mean(axis=1), axis=0)
    for t in TYPE_ORDER:
        ax.plot(range(7), rel.loc[t], color=TYPE_COLORS[t], marker="o", ms=2.5)
    ax.axhline(1, color=GREY, lw=0.6)
    ax.set_xticks(range(7), [d[0] for d in DAYS])
    ax.set_ylim(0.9, 1.1)
    ax.set_ylabel("load / weekly mean")
    wk = fx["dow"] >= 5
    up = (
        fx[wk].groupby("household_type")["active_power_kw"].mean()
        / fx[~wk].groupby("household_type")["active_power_kw"].mean()
        - 1
    )
    ax.text(
        0.04,
        0.05,
        f"weekend vs. weekday: {up.min():+.1%} to {up.max():+.1%}",
        transform=ax.transAxes,
        fontsize=6.5,
    )
    _sub(ax, "Small weekend uplift, no other weekday effect")
    _label(ax, "(f)")
    return _finish(fig, out_dir, "weekly_profile_heatmap")


# --------------------------------------------------------------------------- #
# Fig 9 - daily profiles by season, weekday, occupancy
# --------------------------------------------------------------------------- #


def fig_daily_profiles(fx: pd.DataFrame, out_dir=None) -> plt.Figure:
    fig = _fig(W2, 2.5)
    gs = fig.add_gridspec(1, 3)
    x = np.arange(48) / 2

    def prof(d):
        return d.groupby([HOUSEHOLD_COL, "slot"])["active_power_kw"].mean().unstack(0)

    ax = fig.add_subplot(gs[0])
    for sname in SEASON_ORDER:
        p = prof(fx[fx["season"].eq(sname)])
        ax.fill_between(
            x,
            p.quantile(0.25, axis=1),
            p.quantile(0.75, axis=1),
            color=SEASON_COLORS[sname],
            alpha=0.07,
            lw=0,
        )
        ax.plot(x, p.mean(axis=1), color=SEASON_COLORS[sname], label=sname)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 6))
    ax.set_xlabel("local hour")
    ax.set_ylabel("active power (kW)")
    ax.legend(loc="upper right", ncol=2)
    _sub(ax, "By season (mean, household IQR)")
    _label(ax, "(a)")

    ax = fig.add_subplot(gs[1])
    wk = fx["dow"] >= 5
    for lab, m, col, ls in (("weekday", ~wk, BLUE, "-"), ("weekend", wk, VERM, "--")):
        p = prof(fx[m])
        ax.fill_between(
            x, p.quantile(0.25, axis=1), p.quantile(0.75, axis=1), color=col, alpha=0.08, lw=0
        )
        ax.plot(x, p.mean(axis=1), color=col, ls=ls, label=lab)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 6))
    ax.set_xlabel("local hour")
    ax.legend(loc="upper right")
    _sub(ax, "Weekday vs. weekend")
    _label(ax, "(b)")

    ax = fig.add_subplot(gs[2])
    pm = fx.groupby("slot")["active_power_kw"].mean()
    ax.plot(x, pm, color=BLUE, label="load (kW)")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 6))
    ax.set_xlabel("local hour")
    ax.set_ylabel("mean active power (kW)", color=BLUE)
    ax2 = ax.twinx()
    ax2.plot(
        x,
        fx.groupby("slot")["occupancy_probability"].mean(),
        color=ORANGE,
        ls="--",
        label="occupancy probability",
    )
    ax2.set_ylabel("occupancy probability", color=ORANGE)
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)
    ax2.set_ylim(0, 1)
    r = fx[["active_power_kw", "occupancy_probability"]].corr().iloc[0, 1]
    ax.text(0.30, 0.46, f"corr(P, occupancy) = {r:.2f}", transform=ax.transAxes, fontsize=6.5)
    _sub(ax, "Load follows the occupancy schedule")
    _label(ax, "(c)")
    return _finish(fig, out_dir, "daily_profiles")


# --------------------------------------------------------------------------- #
# Fig 10 - F6 seasonality, temperature and sub-meters
# --------------------------------------------------------------------------- #


def fig_monthly_load_temperature(fx: pd.DataFrame, out_dir=None) -> plt.Figure:
    fig = _fig(W2, 4.2)
    gs = fig.add_gridspec(2, 2)

    ax = fig.add_subplot(gs[0, 0])
    hm = fx.groupby(["month", HOUSEHOLD_COL])["active_power_kw"].mean().unstack()
    mo = hm.index
    ax.fill_between(
        mo, hm.quantile(0.25, axis=1), hm.quantile(0.75, axis=1), color=BLUE, alpha=0.18, lw=0
    )
    ax.plot(mo, hm.mean(axis=1), color=BLUE, marker="o", ms=3)
    ax.set_ylabel("mean active power (kW)", color=BLUE)
    ax.set_xticks(range(1, 13), list("JFMAMJJASOND"))
    ax2 = ax.twinx()
    tm = fx.groupby("month")["temperature_c"].mean()
    ax2.plot(mo, tm, color=VERM, marker="s", ms=3, ls="--")
    ax2.set_ylabel("temperature (°C)", color=VERM)
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)
    ax.set_xlabel("month")
    lm = hm.mean(axis=1)
    ax.text(
        0.5,
        0.05,
        f"peak {lm.max():.2f} kW (month {lm.idxmax()}), trough {lm.min():.2f} kW (month {lm.idxmin()})",
        transform=ax.transAxes,
        fontsize=6.3,
        ha="center",
    )
    _sub(ax, "Monthly load (household IQR) and temperature")
    _label(ax, "(a)")

    ax = fig.add_subplot(gs[0, 1])
    bins = np.arange(2, 42, 3)
    for t in TYPE_ORDER:
        d = fx[fx["household_type"].eq(t)]
        _binned(
            ax, d["temperature_c"], d["active_power_kw"], bins, color=TYPE_COLORS[t], band=False
        )
    ax.set_xlabel("temperature (°C)")
    ax.set_ylabel("mean active power (kW)")
    _type_legend(ax, loc="upper center", ncol=2)
    _sub(ax, "U-shaped response, all household types")
    _label(ax, "(b)")

    ax = fig.add_subplot(gs[1, 0])
    mm = fx.groupby("month")[SUB_COLS].mean()
    for col, c, ls, lab in zip(
        SUB_COLS,
        [VERM, BLUE, GREEN, ORANGE],
        ["-", "--", "-.", ":"],
        ["heating", "cooling", "lighting", "appliance"],
        strict=True,
    ):
        ax.plot(mm.index, mm[col], color=c, ls=ls, marker="o", ms=2.5, label=lab)
    ax.set_xticks(range(1, 13), list("JFMAMJJASOND"))
    ax.set_xlabel("month")
    ax.set_ylabel("sub-meter energy (kWh / 30 min)")
    ax.legend(loc="center right")
    _sub(ax, "Thermal sub-meters follow the temperature season")
    _label(ax, "(c)")

    ax = fig.add_subplot(gs[1, 1])
    d = fx[["temperature_c", "sub2_cooling_kwh"]].dropna()
    hb = ax.hexbin(
        d["temperature_c"],
        d["sub2_cooling_kwh"],
        gridsize=38,
        bins="log",
        cmap="Greys",
        mincnt=1,
        extent=(0, 42, 0, 0.8),
        linewidths=0,
        rasterized=True,
    )
    _binned(
        ax, d["temperature_c"], d["sub2_cooling_kwh"], np.arange(2, 42, 3), color=VERM, band=False
    )
    r = d.corr().iloc[0, 1]
    v = d["sub2_cooling_kwh"].corr((d["temperature_c"] - 20).abs())
    ax.axvline(20, color=BLACK, ls=":", lw=0.8)
    ax.text(
        0.04,
        0.95,
        f"r(T) = {r:.2f}\nr(|T $-$ 20 °C|) = {v:.2f}",
        transform=ax.transAxes,
        fontsize=7,
        va="top",
    )
    ax.set_ylim(0, 0.8)
    ax.set_xlabel("temperature (°C)")
    ax.set_ylabel("cooling sub-meter (kWh)")
    fig.colorbar(hb, ax=ax, label="records (log)", pad=0.02, fraction=0.06)
    _sub(ax, "Cooling energy is V-shaped in temperature")
    _label(ax, "(d)")
    return _finish(fig, out_dir, "monthly_load_temperature")


# --------------------------------------------------------------------------- #
# Fig 11 - temporal splits
# --------------------------------------------------------------------------- #


def fig_temporal_splits(fx: pd.DataFrame, splits: dict, out_dir=None) -> plt.Figure:
    fig = _fig(W2, 2.4)
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1.1])
    daily = fx.groupby([HOUSEHOLD_COL, "date_local"])["total_energy_kwh"].sum().unstack(0)
    ax = fig.add_subplot(gs[0])
    for name, (a, b) in splits.items():
        ax.axvspan(a, b, color=SPLIT_COLORS[name], lw=0)
        ax.text(
            a + (b - a) / 2,
            0.97,
            name,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7,
            style="italic",
        )
    ax.fill_between(
        daily.index,
        daily.min(axis=1),
        daily.max(axis=1),
        color=GREY,
        alpha=0.3,
        lw=0,
        label="min-max across households",
    )
    ax.plot(daily.index, daily.mean(axis=1), color=BLACK, lw=0.7, label="mean (daily)")
    ax.plot(
        daily.index,
        daily.mean(axis=1).rolling(7, center=True, min_periods=3).mean(),
        color=VERM,
        lw=1.4,
        label="7-day mean",
    )
    ax.set_xlim(daily.index.min(), daily.index.max())
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%b"))
    ax.set_ylabel("daily energy per household (kWh)")
    ax.set_ylim(0, daily.max().max() * 1.1)
    ax.legend(loc="lower left", ncol=3)
    _sub(ax, "Temporal split (train / validation / test)")
    _label(ax, "(a)")

    ax = fig.add_subplot(gs[1])
    data, labs = [], []
    for name, (a, b) in splits.items():
        m = (daily.index >= a) & (daily.index < b)
        data.append(daily[m].stack().to_numpy())
        labs.append(f"{name}\n(n={m.sum()} d)")
    bp = ax.boxplot(
        data,
        tick_labels=labs,
        showfliers=False,
        patch_artist=True,
        widths=0.6,
        medianprops={"color": BLACK},
    )
    for patch, name in zip(bp["boxes"], splits, strict=True):
        patch.set(facecolor=SPLIT_COLORS[name], lw=0.8)
    ax.set_ylabel("household-day energy (kWh)")
    _sub(ax, "Level shift across splits")
    _label(ax, "(b)")
    return _finish(fig, out_dir, "temporal_splits")


# --------------------------------------------------------------------------- #
# Fig 12 - power distribution per household
# --------------------------------------------------------------------------- #


def fig_power_distribution(fx: pd.DataFrame, out_dir=None) -> plt.Figure:
    meta = _hh_meta(fx)
    fig = _fig(W2, 2.8)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25])

    ax = fig.add_subplot(gs[0])
    for hh in meta.index:
        v = np.sort(fx.loc[fx[HOUSEHOLD_COL].eq(hh), "active_power_kw"].dropna().to_numpy())
        sf = 1 - np.arange(1, len(v) + 1) / len(v)
        ev = meta.loc[hh, "has_ev"] == 1
        ax.plot(
            v[:-1],
            sf[:-1],
            color=TYPE_COLORS[meta.loc[hh, "household_type"]],
            ls="--" if ev else "-",
            lw=1.2 if ev else 0.9,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.3, 30)
    ax.set_ylim(5e-5, 1.5)
    ax.set_xlabel("active power $x$ (kW)")
    ax.set_ylabel(r"$P(\mathrm{power} > x)$")
    ax.legend(
        handles=[Line2D([0], [0], color=TYPE_COLORS[t], label=TYPE_LABELS[t]) for t in TYPE_ORDER]
        + [Line2D([0], [0], color=BLACK, ls="--", label="has EV")],
        loc="lower left",
    )
    _sub(ax, "Survival function per household")
    _label(ax, "(a)")

    ax = fig.add_subplot(gs[1])
    order = fx.groupby(HOUSEHOLD_COL)["active_power_kw"].median().sort_values().index
    data = [fx.loc[fx[HOUSEHOLD_COL].eq(h), "active_power_kw"].dropna().to_numpy() for h in order]
    bp = ax.boxplot(
        data,
        vert=True,
        showfliers=False,
        patch_artist=True,
        whis=(1, 99),
        widths=0.65,
        medianprops={"color": BLACK},
    )
    for patch, h in zip(bp["boxes"], order, strict=True):
        patch.set(facecolor=TYPE_COLORS[meta.loc[h, "household_type"]], alpha=0.85, lw=0.6)
    der = []
    for h in order:
        tag = "".join(
            c
            for c, f in (("S", "has_solar"), ("B", "has_battery"), ("E", "has_ev"))
            if meta.loc[h, f] == 1
        )
        der.append(f"{_short(h)[-3:]}\n{tag or '-'}")
    ax.set_xticks(range(1, len(order) + 1), der, fontsize=6)
    ax.set_ylim(0, 20)
    ax.set_xlabel(
        "household (S: PV, B: battery, E: EV; box = IQR, whiskers = 1-99 %)", fontsize=6.8
    )
    ax.set_ylabel("active power (kW)")
    _sub(ax, "Households sorted by median load")
    _label(ax, "(b)")
    return _finish(fig, out_dir, "power_distribution_log")


# --------------------------------------------------------------------------- #
# Fig 13 - DER profiles (PV, EV, battery)
# --------------------------------------------------------------------------- #


def fig_der_profiles(fx: pd.DataFrame, out_dir=None) -> plt.Figure:
    meta = _hh_meta(fx)
    fig = _fig(W2, 4.4)
    gs = fig.add_gridspec(2, 3)
    x = np.arange(48) / 2
    solar = fx[fx["has_solar"].eq(1)]

    ax = fig.add_subplot(gs[0, 0])
    d = solar[(solar["hour_local"] >= 10) & (solar["hour_local"] <= 14)].copy()
    d["cf"] = d["solar_generation_kw"] / d["solar_capacity_kwp"]
    d = d.dropna(subset=["cf", "cloud_cover_pct"])
    hb = ax.hexbin(
        d["cloud_cover_pct"],
        d["cf"],
        gridsize=30,
        bins="log",
        cmap="Greys",
        mincnt=1,
        linewidths=0,
        rasterized=True,
    )
    _binned(ax, d["cloud_cover_pct"], d["cf"], np.arange(0, 101, 10), color=VERM, band=False)
    r = d[["cf", "cloud_cover_pct"]].corr().iloc[0, 1]
    ax.text(0.97, 0.95, f"r = {r:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=7)
    ax.set_xlabel("cloud cover (%)")
    ax.set_ylabel("PV output / capacity (10-14 h)")
    fig.colorbar(hb, ax=ax, label="records (log)", pad=0.02, fraction=0.06)
    _sub(ax, "PV output vs. cloud cover")
    _label(ax, "(a)")

    ax = fig.add_subplot(gs[0, 1])
    for hh, col in zip(solar[HOUSEHOLD_COL].unique(), [BLUE, VERM, GREEN, ORANGE], strict=False):
        p = solar[solar[HOUSEHOLD_COL].eq(hh)].groupby("slot")["solar_generation_kw"].mean()
        ax.plot(
            x,
            p.to_numpy(),
            color=col,
            label=f"{_short(hh)} ({meta.loc[hh, 'solar_capacity_kwp']:.1f} kWp)",
        )
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 6))
    ax.set_xlabel("local hour")
    ax.set_ylabel("PV generation (kW)")
    ax.legend(loc="upper left", fontsize=6)
    _sub(ax, "Mean PV profile per household")
    _label(ax, "(b)")

    ax = fig.add_subplot(gs[0, 2])
    plain = fx[fx["has_solar"].eq(0) & fx["has_ev"].eq(0)]
    ax.plot(x, plain.groupby("slot")["active_power_kw"].mean(), color=GREY, label="no PV, no EV")
    ax.plot(
        x, solar.groupby("slot")["active_power_kw"].mean(), color=BLUE, label="with PV (net load)"
    )
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 6))
    ax.set_xlabel("local hour")
    ax.set_ylabel("active power (kW)")
    ax2 = ax.twinx()
    ax2.fill_between(
        x, solar.groupby("slot")["solar_generation_kw"].mean(), color=ORANGE, alpha=0.25, lw=0
    )
    ax2.set_ylabel("PV generation (kW)", color=ORANGE)
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)
    ax.legend(loc="lower left")
    _sub(ax, "Net load dips at the PV peak")
    _label(ax, "(c)")

    ev = fx[fx["has_ev"].eq(1)]
    ax = fig.add_subplot(gs[1, 0])
    for hh, col in zip(ev[HOUSEHOLD_COL].unique(), [BLUE, VERM, GREEN], strict=False):
        e = ev[ev[HOUSEHOLD_COL].eq(hh)]
        ax.plot(
            x,
            e.assign(on=e["ev_load_kw"].gt(0)).groupby("slot")["on"].mean(),
            color=col,
            label=f"{_short(hh)} ({meta.loc[hh, 'ev_charger_kw']:.1f} kW)",
        )
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 6))
    ax.set_xlabel("local hour")
    ax.set_ylabel("P(EV charging)")
    ax.legend(loc="center", bbox_to_anchor=(0.5, 0.5))
    _sub(ax, "EV charging window")
    _label(ax, "(d)")

    ax = fig.add_subplot(gs[1, 1])
    bl = ev.loc[ev["ev_session_len"] > 0, "ev_session_len"]
    runs = bl.groupby((ev["ev_session_len"].ne(ev["ev_session_len"].shift())).cumsum()).first()
    vc = runs.value_counts().sort_index()
    ax.bar(vc.index, vc.to_numpy(), color=BLUE, width=0.75)
    ax.set_xlabel("consecutive charging intervals")
    ax.set_ylabel("charging bursts")
    on = ev.loc[ev["ev_load_kw"] > 0, "ev_load_kw"]
    ax.text(
        0.97,
        0.95,
        f"on-power: median {on.median():.1f} kW\n(range {on.min():.1f}-{on.max():.1f})",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
    )
    _sub(ax, "Charging bursts are short and random")
    _label(ax, "(e)")

    ax = fig.add_subplot(gs[1, 2])
    bat = fx[fx["has_battery"].eq(1)]
    for hh, col in zip(bat[HOUSEHOLD_COL].unique(), [BLUE, VERM], strict=False):
        b = bat[bat[HOUSEHOLD_COL].eq(hh)].groupby("slot")
        ax.plot(x, b["battery_charge_kw"].mean(), color=col, label=f"{_short(hh)[-3:]} charge")
        ax.plot(
            x,
            -b["battery_discharge_kw"].mean(),
            color=col,
            ls="--",
            label=f"{_short(hh)[-3:]} discharge",
        )
    ax.axhline(0, color=BLACK, lw=0.6)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 6))
    ax.set_xlabel("local hour")
    ax.set_ylabel("battery power (kW)")
    ax.legend(loc="upper right", ncol=1, fontsize=6)
    _sub(ax, "Battery: midday charge, evening discharge")
    _label(ax, "(f)")
    return _finish(fig, out_dir, "der_profiles")


# --------------------------------------------------------------------------- #
# Fig 14 - F5 weather
# --------------------------------------------------------------------------- #


def fig_weather(fx: pd.DataFrame, out_dir=None) -> plt.Figure:
    fig = _fig(W2, 2.6)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 1.3])
    d = fx.dropna(subset=["temperature_c", "weather_condition"])
    order = d["weather_condition"].value_counts().index.tolist()

    ax = fig.add_subplot(gs[0])
    data = [d.loc[d["weather_condition"].eq(c), "temperature_c"].to_numpy() for c in order]
    bp = ax.boxplot(
        data,
        tick_labels=[c.replace("_", "\n") for c in order],
        showfliers=False,
        patch_artist=True,
        widths=0.6,
        whis=(1, 99),
        medianprops={"color": BLACK},
    )
    for p in bp["boxes"]:
        p.set(facecolor=SKY, lw=0.6)
    ax.axhline(10, color=VERM, ls="--", lw=0.9)
    snow = d[d["weather_condition"].eq("snow")]
    ax.set_ylim(-9, 40)
    ax.text(
        0.98,
        0.02,
        f"'snow' rows above 10 °C: {(snow['temperature_c'] > 10).mean():.0%} "
        f"({int((snow['temperature_c'] > 10).sum()):,})",
        transform=ax.transAxes,
        ha="right",
        fontsize=6,
    )
    ax.tick_params(axis="x", labelsize=6)
    ax.set_ylabel("temperature (°C)")
    _sub(ax, "Condition vs. temperature")
    _label(ax, "(a)")

    ax = fig.add_subplot(gs[1])
    sd = fx.groupby(UTC_COL)["temperature_c"].std().dropna()
    ax.hist(sd, bins=60, color=BLUE, lw=0)
    ax.axvline(sd.median(), color=VERM, ls="--", lw=1)
    ax.text(
        0.97,
        0.95,
        f"median {sd.median():.2f} °C",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        color=VERM,
    )
    ax.set_xlabel("std of temperature across households (°C)")
    ax.set_ylabel("timestamps")
    _sub(ax, "Same instant, different weather")
    _label(ax, "(b)")

    ax = fig.add_subplot(gs[2])
    sh = (
        pd.crosstab(d["month"], d["weather_condition"], normalize="index").reindex(columns=order)
        * 100
    )
    bottom = np.zeros(len(sh))
    cols = [BLUE, SKY, GREEN, ORANGE, VERM, PURPLE, GREY]
    for c, col in zip(order, cols, strict=False):
        ax.bar(
            sh.index, sh[c], bottom=bottom, color=col, width=0.8, label=c.replace("_", " "), lw=0
        )
        bottom += sh[c].to_numpy()
    ax.set_xticks(range(1, 13), list("JFMAMJJASOND"))
    ax.set_ylim(0, 100)
    ax.set_xlabel("month")
    ax.set_ylabel("share of records (%)")
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, fontsize=5.8, columnspacing=0.8
    )
    ax.grid(False)
    _sub(ax, "Conditions ignore the season")
    _label(ax, "(c)")
    return _finish(fig, out_dir, "temperature_vs_weather")


# --------------------------------------------------------------------------- #
# Fig 15 - F7 sub-meter balance
# --------------------------------------------------------------------------- #


def fig_submeter_balance(fx: pd.DataFrame, out_dir=None) -> plt.Figure:
    meta = _hh_meta(fx)
    d = fx[fx["total_energy_kwh"] > 0].copy()
    d["share"] = d[SUB_COLS].sum(axis=1) / d["total_energy_kwh"]
    fig = _fig(W2, 2.7)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.1, 1.35])

    ax = fig.add_subplot(gs[0])
    ax.hist(d["share"], bins=np.linspace(0, 2, 61), color=BLUE, lw=0)
    ax.set_xlim(0, 2)
    ax.axvline(1, color=BLACK, ls="--", lw=0.9)
    ax.axvline(d["share"].median(), color=VERM, lw=1.2)
    ax.text(
        0.97,
        0.95,
        f"median {d['share'].median():.2f}\n5-95 %: {d['share'].quantile(0.05):.2f}-"
        f"{d['share'].quantile(0.95):.2f}\n$>$1: {(d['share'] > 1).mean():.1%}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
    )
    ax.set_xlabel(r"$\sum$ sub-meters / total energy")
    ax.set_ylabel("records")
    _sub(ax, "Not a closed balance")
    _label(ax, "(a)")

    ax = fig.add_subplot(gs[1])
    order = d.groupby(HOUSEHOLD_COL)["share"].median().sort_values().index
    data = [d.loc[d[HOUSEHOLD_COL].eq(h), "share"].to_numpy() for h in order]
    bp = ax.boxplot(
        data,
        vert=False,
        showfliers=False,
        patch_artist=True,
        whis=(5, 95),
        widths=0.65,
        medianprops={"color": BLACK},
    )
    for patch, h in zip(bp["boxes"], order, strict=True):
        patch.set(facecolor=TYPE_COLORS[meta.loc[h, "household_type"]], lw=0.6)
    ax.axvline(1, color=BLACK, ls="--", lw=0.9)
    ax.set_xlim(0, 2.2)
    ax.set_yticks(
        range(1, len(order) + 1),
        [f"{_short(h)[-3:]} {_der_tag(meta, h)}" for h in order],
        fontsize=6,
    )
    ax.set_xlabel("share (box = IQR, whiskers = 5-95 %)")
    _sub(ax, "Per household")
    _label(ax, "(b)")

    ax = fig.add_subplot(gs[2])
    comp = d.groupby(HOUSEHOLD_COL).agg(
        heating=("sub1_heating_kwh", "mean"),
        cooling=("sub2_cooling_kwh", "mean"),
        lighting=("sub3_lighting_kwh", "mean"),
        appliance=("sub4_appliance_kwh", "mean"),
        ev=("ev_load_kw", lambda s: 0.5 * s.mean()),
        total=("total_energy_kwh", "mean"),
    )
    comp["unmetered"] = comp["total"] - comp[
        ["heating", "cooling", "lighting", "appliance", "ev"]
    ].sum(axis=1)
    comp = comp.loc[order[::-1]]
    left = np.zeros(len(comp))
    for c, col in zip(
        ["heating", "cooling", "lighting", "appliance", "ev", "unmetered"],
        [VERM, BLUE, YELLOW, GREEN, PURPLE, "#d0d0d0"],
        strict=True,
    ):
        ax.barh(range(len(comp)), comp[c], left=left, color=col, height=0.75, label=c, lw=0)
        left += comp[c].to_numpy()
    ax.set_yticks(
        range(len(comp)), [f"{_short(h)[-3:]} {_der_tag(meta, h)}" for h in comp.index], fontsize=6
    )
    ax.set_xlabel("mean energy per 30 min (kWh)")
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=5.8, columnspacing=0.8
    )
    _sub(ax, "Where the energy goes")
    _label(ax, "(c)")
    return _finish(fig, out_dir, "submeter_share_boxplot")


# --------------------------------------------------------------------------- #
# Fig 16 - F8 metadata contradictions
# --------------------------------------------------------------------------- #


def fig_metadata(fx: pd.DataFrame, out_dir=None) -> plt.Figure:
    meta = _hh_meta(fx)
    meta["mean_p"] = fx.groupby(HOUSEHOLD_COL)["active_power_kw"].mean()
    fig = _fig(W2, 2.5)
    gs = fig.add_gridspec(1, 2)
    rng = np.random.default_rng(0)

    ax = fig.add_subplot(gs[0])
    for k, t in enumerate(TYPE_ORDER):
        m = meta[meta["household_type"].eq(t)]
        xs = k + rng.uniform(-0.12, 0.12, len(m))
        ax.scatter(
            xs, m["size_m2"], color=TYPE_COLORS[t], s=22, zorder=3, edgecolor="white", lw=0.4
        )
    ax.set_xticks(range(5), [TYPE_LABELS[t].replace(" ", "\n") for t in TYPE_ORDER], fontsize=6.5)
    ax.set_ylabel("declared floor area (m²)")
    big = meta.loc[meta["household_type"].eq("house_large"), "size_m2"].min()
    small = meta.loc[meta["household_type"].eq("apartment_small"), "size_m2"].max()
    ax.annotate(
        f"house, large: {big:.0f} m²",
        xy=(3, big),
        xytext=(1.9, 25),
        fontsize=6.5,
        arrowprops={"arrowstyle": "-", "lw": 0.6},
    )
    ax.annotate(
        f"apartment, small: {small:.0f} m²",
        xy=(0, small),
        xytext=(0.3, 285),
        fontsize=6.5,
        arrowprops={"arrowstyle": "-", "lw": 0.6},
    )
    ax.set_ylim(0, 310)
    _sub(ax, "Declared size contradicts household type")
    _label(ax, "(a)")

    ax = fig.add_subplot(gs[1])
    for t in TYPE_ORDER:
        m = meta[meta["household_type"].eq(t)]
        ax.scatter(
            m["size_m2"],
            m["mean_p"],
            color=TYPE_COLORS[t],
            s=22,
            edgecolor="white",
            lw=0.4,
            zorder=3,
            label=TYPE_LABELS[t],
        )
    rho = meta[["size_m2", "mean_p"]].corr(method="spearman").iloc[0, 1]
    ax.text(
        0.03, 0.95, f"Spearman $\\rho$ = {rho:.2f}", transform=ax.transAxes, va="top", fontsize=7
    )
    ax.set_xlabel("declared floor area (m²)")
    ax.set_ylabel("mean active power (kW)")
    ax.legend(loc="lower right", ncol=2)
    _sub(ax, "Size does not predict load")
    _label(ax, "(b)")
    return _finish(fig, out_dir, "metadata_contradictions")


# --------------------------------------------------------------------------- #
# Fig 17 - F9 flat runs
# --------------------------------------------------------------------------- #


def fig_flat_runs(fx: pd.DataFrame, out_dir=None) -> plt.Figure:
    f = fx.dropna(subset=["flat_run_p"]).copy()
    nxt = f.groupby(HOUSEHOLD_COL)["flat_run_p"].shift(-1)
    is_end = nxt.isna() | (nxt == 1)
    ends = f[is_end]
    lens = ends["flat_run_p"].astype(int)

    fig = _fig(W2, 2.5)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1, 1.4])
    ax = fig.add_subplot(gs[0])
    vc = lens.value_counts().sort_index()
    ax.bar(vc.index, vc.to_numpy(), color=[VERM if i >= 8 else GREEN for i in vc.index], width=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("maximal flat-run length (intervals)")
    ax.set_ylabel("runs")
    ax.text(
        0.97,
        0.95,
        f"max {lens.max()} intervals ({lens.max() / 2:.0f} h)\nruns $\\geq$ 8: {(lens >= 8).sum()}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
    )
    _sub(ax, "Flat runs of active power")
    _label(ax, "(a)")

    ax = fig.add_subplot(gs[1])
    long_p = np.sort(f.loc[f["flat_run_p"] >= 8, "active_power_kw"].to_numpy())
    allp = np.sort(f["active_power_kw"].to_numpy())
    grid = np.logspace(-2.2, 1.4, 300)
    ax.plot(
        grid, np.searchsorted(allp, grid, side="right") / len(allp), color=GREY, label="all records"
    )
    ax.plot(
        grid,
        np.searchsorted(long_p, grid, side="right") / len(long_p),
        color=VERM,
        ls="--",
        label=f"records in runs $\\geq$ 8 (n={len(long_p)})",
    )
    ax.set_xscale("log")
    ax.set_xlabel("active power (kW)")
    ax.set_ylabel("empirical CDF")
    ax.legend(loc="lower right")
    ax.text(
        0.97,
        0.45,
        f"median in long runs:\n{np.median(long_p):.2f} kW",
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=6.5,
    )
    _sub(ax, "Long runs sit at near-zero load")
    _label(ax, "(b)")

    ax = fig.add_subplot(gs[2])
    i_end = ends["flat_run_p"].idxmax()
    hh = f.loc[i_end, HOUSEHOLD_COL]
    sub = fx[fx[HOUSEHOLD_COL].eq(hh)].reset_index()
    pos = int(sub.index[sub["index"] == i_end][0]) if "index" in sub else 0
    n = int(f.loc[i_end, "flat_run_p"])
    w = sub.iloc[max(pos - n - 12, 0) : pos + 13]
    ax.plot(w["ts_local"], w["active_power_kw"], color=BLACK, marker="o", ms=2.2)
    ax.axvspan(
        sub.loc[pos - n + 1, "ts_local"], sub.loc[pos, "ts_local"], color=VERM, alpha=0.15, lw=0
    )
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mpl.dates.HourLocator(interval=4))
    ax.set_xlabel(f"time, {_short(hh)}, {sub.loc[pos, 'ts_local']:%d %b}")
    ax.set_ylabel("active power (kW)")
    _sub(ax, f"Longest natural run ({n} intervals)")
    _label(ax, "(c)")
    return _finish(fig, out_dir, "flat_run_histogram")


# --------------------------------------------------------------------------- #
# Fig 18 - F10 cross-household coherence
# --------------------------------------------------------------------------- #


def fig_cross_household(fx: pd.DataFrame, out_dir=None) -> plt.Figure:
    piv = fx.pivot_table(index=UTC_COL, columns=HOUSEHOLD_COL, values="active_power_kw")
    c30, cday = piv.corr(), piv.resample("1D").mean().corr()
    order = leaves_list(linkage(squareform(1 - cday.to_numpy(), checks=False), "average"))
    names = [_short(c)[-3:] for c in cday.columns[order]]

    fig = _fig(W2, 2.7)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25])
    im = None
    for k, (c, t) in enumerate(((c30, "30-min"), (cday, "daily"))):
        ax = fig.add_subplot(gs[k])
        m = c.to_numpy()[np.ix_(order, order)]
        im = ax.imshow(m, vmin=0, vmax=1, cmap="magma")
        ax.set_xticks(range(len(names)), names, rotation=90, fontsize=5.5)
        ax.set_yticks(range(len(names)), names, fontsize=5.5)
        ax.grid(False)
        _sub(ax, f"Correlation, {t} mean")
        _label(ax, f"({'ab'[k]})")
    fig.colorbar(im, ax=fig.axes[:2], shrink=0.7, pad=0.02, label="Pearson r", location="bottom")

    ax = fig.add_subplot(gs[2])
    wins = [
        ("30min", "30 min"),
        ("1h", "1 h"),
        ("3h", "3 h"),
        ("6h", "6 h"),
        ("12h", "12 h"),
        ("1D", "1 d"),
        ("3D", "3 d"),
        ("7D", "7 d"),
    ]
    mean, lo, hi = [], [], []
    iu = np.triu_indices(piv.shape[1], k=1)
    for w, _ in wins:
        c = (piv if w == "30min" else piv.resample(w).mean()).corr().to_numpy()[iu]
        mean.append(np.nanmean(c)), lo.append(np.nanmin(c)), hi.append(np.nanmax(c))
    xs = np.arange(len(wins))
    ax.fill_between(xs, lo, hi, color=BLUE, alpha=0.2, lw=0, label="min-max over pairs")
    ax.plot(xs, mean, color=BLUE, marker="o", ms=3, label="mean pairwise r")
    ax.set_xticks(xs, [w[1] for w in wins], fontsize=6.5)
    ax.set_xlabel("aggregation window")
    ax.set_ylabel("pairwise correlation")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    _sub(ax, "Coherence emerges with aggregation")
    _label(ax, "(c)")
    return _finish(fig, out_dir, "cross_household_correlation")


# --------------------------------------------------------------------------- #
# Fig 19 - engineered features
# --------------------------------------------------------------------------- #


def fig_engineered_features(
    fx: pd.DataFrame,
    splits: dict,
    out_dir=None,
    hh: str = "HH_00099_AUS",
    week_start: str = "2020-10-12",
) -> plt.Figure:
    from energymind.eda_features import CORR_FEATURES

    fig = _fig(W2, 5.0)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])

    ax = fig.add_subplot(gs[0, 0])
    a = pd.Timestamp(week_start)
    w = fx[
        fx[HOUSEHOLD_COL].eq(hh)
        & (fx["ts_local"] >= a)
        & (fx["ts_local"] < a + pd.Timedelta(days=7))
    ]
    ax.fill_between(
        w["ts_local"],
        w["p_roll_mean_24h"] - 2 * w["p_roll_std_24h"],
        w["p_roll_mean_24h"] + 2 * w["p_roll_std_24h"],
        color=BLUE,
        alpha=0.15,
        lw=0,
        label=r"past-only 24 h mean $\pm 2\sigma$",
    )
    ax.plot(
        w["ts_local"],
        w["how_median"],
        color=GREEN,
        ls="--",
        lw=1,
        label="hour-of-week median (train)",
    )
    ax.plot(w["ts_local"], w["active_power_kw"], color=BLACK, lw=0.9, label="measured $P$")
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%a"))
    ax.xaxis.set_major_locator(mpl.dates.DayLocator())
    ax.set_ylabel("kW")
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=5.8, columnspacing=0.8
    )
    _sub(ax, f"One week, {_short(hh)} (validation window)")
    _label(ax, "(a)")

    ax = fig.add_subplot(gs[0, 1])
    bins = np.linspace(-6, 10, 81)
    for name, (s0, s1) in splits.items():
        z = fx.loc[(fx["ts_local"] >= s0) & (fx["ts_local"] < s1), "p_resid_z"].dropna()
        h, _ = np.histogram(z.clip(-6, 10), bins=bins, density=True)
        ax.step(
            bins[:-1],
            h,
            where="post",
            label=f"{name} ({(z.abs() > 3).mean():.1%} beyond ±3)",
            color={"train": BLUE, "val": ORANGE, "test": VERM}[name],
        )
    ax.set_yscale("log")
    ax.set_xlabel("robust residual $z$ (clipped at -6, 10)")
    ax.set_ylabel("density")
    ax.legend(loc="upper right")
    _sub(ax, "Baseline residual by split")
    _label(ax, "(b)")

    ax = fig.add_subplot(gs[1, 0])
    c = fx[CORR_FEATURES].sample(min(60_000, len(fx)), random_state=0).corr(method="spearman")
    order = leaves_list(linkage(squareform(1 - c.abs().to_numpy(), checks=False), "average"))
    c = c.iloc[order, order]
    im = ax.imshow(c.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
    lab = [CORR_LABELS[k] for k in c.columns]
    ax.set_xticks(range(len(lab)), lab, rotation=60, ha="right", fontsize=5.8)
    ax.set_yticks(range(len(lab)), lab, fontsize=5.8)
    for i in range(len(lab)):
        for j in range(len(lab)):
            ax.text(
                j,
                i,
                f"{c.iloc[i, j]:.1f}".replace("0.", ".").replace("-.", "-."),
                ha="center",
                va="center",
                fontsize=4.6,
                color="white" if abs(c.iloc[i, j]) > 0.6 else BLACK,
            )
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Spearman $\\rho$", pad=0.02, fraction=0.05)
    _sub(ax, "Engineered-feature redundancy")
    _label(ax, "(c)")

    ax = fig.add_subplot(gs[1, 1])
    ns = fx["neigh_share"].dropna()
    n_hh = fx[HOUSEHOLD_COL].nunique()
    ax.hist(ns, bins=(np.arange(n_hh + 2) - 0.5) / n_hh, color=BLUE, lw=0.3, edgecolor="white")
    ax.set_yscale("log")
    ax.axvline(ns.mean(), color=VERM, ls="--", lw=1)
    ax.text(
        0.97,
        0.95,
        f"mean {ns.mean():.2f}\n$>$0.5: {(ns > 0.5).mean():.3%}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
    )
    ax.set_xlabel(r"share of households with $|z|>2$ at the same timestamp")
    ax.set_ylabel("timestamps (household-records)")
    _sub(ax, "Neighbourhood share")
    _label(ax, "(d)")
    return _finish(fig, out_dir, "engineered_features")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

#: figure name -> findings it supports (used by the report and the notebook)
FIGURE_FINDINGS = {
    "dataset_overview": "overview",
    "missing_record_map": "F1/F2 (Step 2)",
    "power_identity_scatter": "F1, F2",
    "physics_residuals": "F1",
    "provided_labels": "F2",
    "leakage_f3": "F3",
    "electrical_parameters": "F4",
    "temperature_vs_weather": "F5",
    "weekly_profile_heatmap": "F6",
    "daily_profiles": "F6",
    "monthly_load_temperature": "F6",
    "submeter_share_boxplot": "F7",
    "metadata_contradictions": "F8",
    "flat_run_histogram": "F9",
    "cross_household_correlation": "F10",
    "temporal_splits": "Step 4",
    "power_distribution_log": "heterogeneity",
    "der_profiles": "DER",
    "engineered_features": "Step 6 (prototype)",
}


def split_bounds(settings) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    t = settings.data.splits.temporal
    return {
        k: (pd.Timestamp(getattr(t, k).start), pd.Timestamp(getattr(t, k).end))
        for k in ("train", "val", "test")
    }


def generate_all(
    fx: pd.DataFrame, raw: pd.DataFrame, labels: pd.DataFrame, settings, out_dir
) -> list[str]:
    """Write every figure (PNG + PDF) and return their names."""
    plt.close("all")
    sp = split_bounds(settings)
    b = settings.data.bounds
    figs = [
        fig_dataset_overview(fx, out_dir),
        fig_missing_record_map(fx, out_dir),
        fig_power_identity(raw, out_dir),
        fig_physics_residuals(fx, settings.data.physics_residual_tolerance, out_dir),
        fig_provided_labels(fx, labels, out_dir),
        fig_leakage(raw, out_dir),
        fig_electrical_parameters(
            fx, {"voltage_v": b.voltage_v, "frequency_hz": b.frequency_hz}, out_dir
        ),
        fig_weather(fx, out_dir),
        fig_weekly_profile(fx, out_dir),
        fig_daily_profiles(fx, out_dir),
        fig_monthly_load_temperature(fx, out_dir),
        fig_submeter_balance(fx, out_dir),
        fig_metadata(fx, out_dir),
        fig_flat_runs(fx, out_dir),
        fig_cross_household(fx, out_dir),
        fig_temporal_splits(fx, sp, out_dir),
        fig_power_distribution(fx, out_dir),
        fig_der_profiles(fx, out_dir),
        fig_engineered_features(fx, sp, out_dir),
    ]
    plt.close("all")
    return list(FIGURE_FINDINGS)[: len(figs)]

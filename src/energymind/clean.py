"""Step 2 - regular grid, provided-label masking, imputation and quality mask.

Starting from the Step 1 artifacts (``measured.parquet`` + ``provided_labels.
parquet``), this module builds the clean baseline that every later stage depends
on:

1. reindex every household onto one shared 30-minute UTC grid and flag the
   missing records (finding: missing *records*, not missing values);
2. mask the provided ``spike``/``dropout`` rows so they cannot pollute the
   baseline (finding F2), keeping a ``provided_anomaly`` marker;
3. impute isolated single-slot gaps and rebuild the dependent channels so the
   electrical identities still hold; longer gaps stay NaN;
4. add physics residuals, plausibility flags and the active-power flat-run
   length;
5. derive the ``quality_mask`` = safe-to-inject zones.

Outputs (under ``data/clean/``): ``clean.parquet`` and ``quality_mask.parquet``.

Run it directly to (re)generate them::

    python -m energymind.clean
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from energymind import io
from energymind.config import Settings, load_settings
from energymind.validate import (
    SUB_COLS,
    bounds_flags,
    flat_run_length,
    physics_residuals,
    recompute_dependents,
)

HOUSEHOLD_COL = io.HOUSEHOLD_COL
UTC_COL = io.UTC_COL

#: Measured physical channels blanked on provided-anomaly rows.
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

#: Channels linearly interpolated across a single-slot gap. The dependent
#: channels (S, I, PF, E) are rebuilt from P/Q/V, never interpolated directly.
INTERP_COLUMNS = [
    "voltage_v",
    "frequency_hz",
    "active_power_kw",
    "reactive_power_kvar",
    "ev_load_kw",
    *SUB_COLS,
]

#: Residual identity columns checked for consistency (r_sub is not an identity).
IDENTITY_RESIDUALS = ["r_S", "r_VI", "r_E", "r_PF"]


# --------------------------------------------------------------------------- #
# Step 2a - regular grid
# --------------------------------------------------------------------------- #


def to_grid(df: pd.DataFrame, cfg: Settings | object) -> pd.DataFrame:
    """Reindex each household onto one shared 30-minute UTC grid.

    The grid spans the global first-to-last timestamp (UTC, not floored to
    midnight). Slots absent from the file become ``is_missing_record = True``.
    """
    data_cfg = cfg.data if isinstance(cfg, Settings) else cfg
    idx = pd.date_range(df[UTC_COL].min(), df[UTC_COL].max(), freq=data_cfg.time.freq)
    parts = []
    for household, group in df.groupby(HOUSEHOLD_COL, sort=True):
        group = group.set_index(UTC_COL).reindex(idx)
        group["is_missing_record"] = group[HOUSEHOLD_COL].isna()
        group[HOUSEHOLD_COL] = household
        parts.append(group.rename_axis(UTC_COL).reset_index())
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------- #
# Step 2b - mask the provided anomalies
# --------------------------------------------------------------------------- #


def mask_provided_labels(df: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Mark provided-anomaly rows and blank their electrical channels.

    Returns a copy with an integer ``provided_anomaly`` column (1 on masked rows)
    and NaNs in :data:`ELECTRICAL_COLUMNS`.
    """
    df = df.copy()
    keys = set(zip(labels[HOUSEHOLD_COL], labels[UTC_COL], strict=True))
    mask = np.fromiter(
        ((h, t) in keys for h, t in zip(df[HOUSEHOLD_COL], df[UTC_COL], strict=True)),
        dtype=bool,
        count=len(df),
    )
    df["provided_anomaly"] = np.int8(0)
    df.loc[mask, "provided_anomaly"] = np.int8(1)
    present = [c for c in ELECTRICAL_COLUMNS if c in df.columns]
    df.loc[mask, present] = np.nan
    return df


# --------------------------------------------------------------------------- #
# Step 2c - impute isolated single-slot gaps
# --------------------------------------------------------------------------- #


def impute_single_slots(df: pd.DataFrame, interp_columns: list[str] | None = None) -> pd.DataFrame:
    """Linearly fill gaps of exactly one slot; leave longer gaps as NaN.

    A slot is a *gap* when it is a missing record or a masked provided anomaly.
    A gap is filled only when both neighbours exist, are valid and are finite.
    The dependent channels are then rebuilt by :func:`recompute_dependents`.
    """
    interp_columns = list(interp_columns or INTERP_COLUMNS)
    df = df.copy()
    df["is_imputed"] = np.int8(0)
    bad = df["is_missing_record"].to_numpy(dtype=bool) | df["provided_anomaly"].eq(1).to_numpy(
        dtype=bool
    )

    for _, group in df.groupby(HOUSEHOLD_COL, sort=False):
        positions = group.index.to_numpy()
        n = len(positions)
        gap = bad[positions]
        if not gap.any():
            continue
        starts = np.flatnonzero(gap & ~np.concatenate(([False], gap[:-1])))
        ends = np.flatnonzero(gap & ~np.concatenate((gap[1:], [False])))
        for s, e in zip(starts, ends, strict=True):
            # Only isolated interior gaps between two valid neighbours.
            if e != s or s == 0 or e == n - 1 or gap[s - 1] or gap[e + 1]:
                continue
            prev, pos, nxt = positions[s - 1], positions[s], positions[e + 1]
            a = df.loc[prev, interp_columns].astype(float).to_numpy()
            b = df.loc[nxt, interp_columns].astype(float).to_numpy()
            if not (np.isfinite(a).all() and np.isfinite(b).all()):
                continue
            fill = (a + b) / 2.0
            df.loc[pos, interp_columns] = fill
            values = dict(zip(interp_columns, fill, strict=True))
            recompute_dependents(
                df,
                [pos],
                values["active_power_kw"],
                values["reactive_power_kvar"],
                values["voltage_v"],
            )
            df.loc[pos, "is_imputed"] = np.int8(1)
    return df


# --------------------------------------------------------------------------- #
# Step 2d - quality mask
# --------------------------------------------------------------------------- #


def quality_mask(df: pd.DataFrame, cfg: Settings | object) -> pd.Series:
    """Safe-to-inject rows: no plausibility flag, not provided-anomaly, not
    imputed, not a missing record, and internally consistent (identities held).
    """
    data_cfg = cfg.data if isinstance(cfg, Settings) else cfg
    tol = data_cfg.physics_residual_tolerance
    flag_columns = [c for c in df.columns if c.startswith("flag_")]
    no_flag = ~df[flag_columns].any(axis=1)
    consistent = (df[IDENTITY_RESIDUALS].abs() <= tol).all(axis=1)
    return (
        no_flag
        & consistent
        & df["provided_anomaly"].eq(0)
        & df["is_imputed"].eq(0)
        & ~df["is_missing_record"].to_numpy(dtype=bool)
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def load_step1(settings: Settings, *, regenerate: bool = False):
    """Return ``(measured, provided_labels)`` from Step 1, generating if needed."""
    cfg = settings.data
    clean_dir = io.resolve(cfg.paths.clean_dir)
    measured_path = clean_dir / "measured.parquet"
    labels_path = clean_dir / "provided_labels.parquet"
    if regenerate or not (measured_path.exists() and labels_path.exists()):
        io.prepare(settings)
    return pd.read_parquet(measured_path), pd.read_parquet(labels_path)


def build_baseline(settings: Settings, *, regenerate_step1: bool = False) -> pd.DataFrame:
    """Assemble the full clean baseline (Step 2) from the Step 1 artifacts."""
    cfg = settings.data
    measured, labels = load_step1(settings, regenerate=regenerate_step1)

    df = to_grid(measured, cfg)
    df = mask_provided_labels(df, labels)
    df = impute_single_slots(df)
    df = pd.concat([df, physics_residuals(df)], axis=1)
    df = pd.concat([df, bounds_flags(df, cfg)], axis=1)
    df["flat_run_p"] = flat_run_length(df["active_power_kw"])
    df["quality_mask"] = quality_mask(df, cfg).astype("int8")
    return df


def write_baseline(df: pd.DataFrame, settings: Settings) -> dict[str, object]:
    """Write ``clean.parquet`` and ``quality_mask.parquet``; return paths."""
    clean_dir = io.resolve(settings.data.paths.clean_dir)
    clean_dir.mkdir(parents=True, exist_ok=True)
    clean_path = clean_dir / "clean.parquet"
    quality_path = clean_dir / "quality_mask.parquet"
    df.to_parquet(clean_path, index=False)
    df[[HOUSEHOLD_COL, UTC_COL, "quality_mask"]].to_parquet(quality_path, index=False)
    return {"clean": clean_path, "quality_mask": quality_path}


def load_baseline(settings: Settings) -> pd.DataFrame:
    """Load a previously built baseline (used by later steps)."""
    return pd.read_parquet(io.resolve(settings.data.paths.clean_dir) / "clean.parquet")


def _summarize(df: pd.DataFrame, cfg: Settings) -> dict[str, object]:
    per_household = df.groupby(HOUSEHOLD_COL).size()
    quality = df["quality_mask"].astype(bool)
    residual_max = df.loc[quality, IDENTITY_RESIDUALS].abs().max().max()
    return {
        "rows": len(df),
        "households": int(df[HOUSEHOLD_COL].nunique()),
        "rows_per_household": int(per_household.iloc[0]),
        "rows_per_household_uniform": bool(per_household.nunique() == 1),
        "missing_records": int(df["is_missing_record"].sum()),
        "provided_anomalies": int(df["provided_anomaly"].sum()),
        "imputed": int(df["is_imputed"].sum()),
        "quality_rows": int(quality.sum()),
        "quality_pct": round(100.0 * quality.mean(), 3),
        "residual_max_on_quality": float(residual_max),
    }


def prepare(settings: Settings | None = None, *, regenerate_step1: bool = False):
    """Run Step 2 end to end and return the baseline plus a summary."""
    settings = settings or load_settings()
    df = build_baseline(settings, regenerate_step1=regenerate_step1)
    paths = write_baseline(df, settings)
    return df, _summarize(df, settings), paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Step 2: clean, mask, impute, quality mask.")
    parser.add_argument("--configs-dir", default=None)
    parser.add_argument("--regenerate-step1", action="store_true")
    args = parser.parse_args(argv)

    settings = load_settings(args.configs_dir) if args.configs_dir else load_settings()
    _, summary, paths = prepare(settings, regenerate_step1=args.regenerate_step1)

    print("Step 2 complete")
    print(f"  rows                : {summary['rows']}")
    print(f"  households          : {summary['households']}")
    print(
        f"  rows/household      : {summary['rows_per_household']} "
        f"(uniform: {summary['rows_per_household_uniform']})"
    )
    print(f"  missing records     : {summary['missing_records']}")
    print(f"  provided anomalies  : {summary['provided_anomalies']}")
    print(f"  imputed slots       : {summary['imputed']}")
    print(f"  quality mask        : {summary['quality_rows']} rows ({summary['quality_pct']}%)")
    print(f"  max |residual| (qc) : {summary['residual_max_on_quality']:.2e}")
    print(f"  wrote               : {paths['clean']}")
    print(f"                        {paths['quality_mask']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

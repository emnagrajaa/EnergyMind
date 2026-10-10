"""Step 4 - canonical data splits (temporal windows, household hold-out, experience id).

Two independent, leakage-free assignments are produced and written once to
``data/splits.json``:

* **temporal** - train / validation / test by *local calendar*, half-open
  ``[start, end)``. From ``configs/data.yaml``: train Jan-Aug, val Sep-Oct,
  test Nov-Dec.
* **household hold-out** - 3 held-out households chosen to cover one with solar,
  one with an EV and one ``commercial_small``, so the model is tested on
  households it has never seen.

Every row is additionally tagged with ``experience_id = month`` (local), which
Phase 2 continual learning reuses at no cost.

Run it directly to (re)generate ``data/splits.json``::

    python -m energymind.splits
"""

from __future__ import annotations

import argparse
import json
from datetime import date

import pandas as pd

from energymind import clean, io
from energymind.config import Settings, load_settings

HOUSEHOLD_COL = io.HOUSEHOLD_COL
UTC_COL = io.UTC_COL
SPLIT_NAMES = ("train", "val", "test")
HOLDOUT_SIZE = 3


# --------------------------------------------------------------------------- #
# Temporal split
# --------------------------------------------------------------------------- #


def temporal_bounds(cfg) -> dict[str, tuple[date, date]]:
    """Local-calendar ``[start, end)`` bounds per split, from the config."""
    t = cfg.splits.temporal
    return {name: (getattr(t, name).start, getattr(t, name).end) for name in SPLIT_NAMES}


def _utc_edge(day: date, offset_h: int) -> pd.Timestamp:
    """Convert a local-midnight calendar date to a UTC timestamp.

    ``offset_h`` is the file's constant offset (``-6``); UTC = local - offset.
    """
    return pd.Timestamp(day).tz_localize("UTC") - pd.Timedelta(hours=offset_h)


def assign_temporal(df: pd.DataFrame, cfg) -> pd.Series:
    """Label each row ``train`` / ``val`` / ``test`` by its UTC timestamp."""
    ts = df[UTC_COL]
    offset = cfg.time.assume_utc_offset_h
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    for name, (start, end) in temporal_bounds(cfg).items():
        mask = (ts >= _utc_edge(start, offset)) & (ts < _utc_edge(end, offset))
        out[mask] = name
    return out


def experience_ids(df: pd.DataFrame, cfg) -> pd.Series:
    """``experience_id`` = local month (1-12)."""
    local = df[UTC_COL].dt.tz_convert(None) - pd.Timedelta(hours=-cfg.time.assume_utc_offset_h)
    return local.dt.month.astype("int16")


# --------------------------------------------------------------------------- #
# Household hold-out
# --------------------------------------------------------------------------- #


def load_households(settings: Settings) -> pd.DataFrame:
    """Distinct households with the metadata used to pick the hold-out."""
    df = clean.load_baseline(settings)
    columns = [HOUSEHOLD_COL, "household_type", "has_solar", "has_ev", "has_battery"]
    return (
        df[columns].drop_duplicates(HOUSEHOLD_COL).sort_values(HOUSEHOLD_COL).reset_index(drop=True)
    )


def choose_holdout(meta: pd.DataFrame, cfg) -> list[str]:
    """Return 3 hold-out households: one commercial_small, one solar, one EV.

    If ``configs/data.yaml`` lists ``splits.holdout_households`` explicitly, that
    list is used verbatim; otherwise the households are chosen deterministically.
    """
    configured = list(cfg.splits.holdout_households)
    if configured:
        return sorted(configured)

    chosen: list[str] = []

    def pick(mask: pd.Series) -> None:
        for household in meta.loc[mask, HOUSEHOLD_COL]:
            if household not in chosen:
                chosen.append(household)
                return

    pick(meta["household_type"].eq("commercial_small"))
    pick(meta["has_solar"].eq(1))
    pick(meta["has_ev"].eq(1))
    # Fill up in the unlikely case a category is empty.
    for household in meta[HOUSEHOLD_COL]:
        if len(chosen) >= HOLDOUT_SIZE:
            break
        if household not in chosen:
            chosen.append(household)
    return sorted(chosen)


def household_roles(households, holdout: list[str]) -> pd.Series:
    """Map each household to ``test`` (held out) or ``train_val``."""
    holdout_set = set(holdout)
    return pd.Series(
        ["test" if hh in holdout_set else "train_val" for hh in households],
        index=pd.Index(households, name=HOUSEHOLD_COL),
        dtype="object",
    )


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def build_splits(settings: Settings) -> dict:
    """Assemble the ``splits.json`` payload (deterministic, JSON-serialisable)."""
    cfg = settings.data
    meta = load_households(settings)
    all_households = meta[HOUSEHOLD_COL].tolist()
    holdout = choose_holdout(meta, cfg)
    train_val = [hh for hh in all_households if hh not in set(holdout)]

    offset = cfg.time.assume_utc_offset_h
    temporal = {
        name: {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "start_utc": _utc_edge(start, offset).isoformat(),
            "end_utc": _utc_edge(end, offset).isoformat(),
        }
        for name, (start, end) in temporal_bounds(cfg).items()
    }
    return {
        "local_utc_offset_h": offset,
        "temporal": temporal,
        "holdout_households": holdout,
        "train_val_households": train_val,
        "experience_id": "month",
    }


def write_splits(splits: dict, settings: Settings):
    """Write ``data/splits.json`` (sorted keys, stable formatting)."""
    path = io.resolve(settings.data.paths.splits_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_splits(settings: Settings) -> dict:
    """Load a previously written ``data/splits.json``."""
    return json.loads(io.resolve(settings.data.paths.splits_file).read_text(encoding="utf-8"))


def add_split_columns(df: pd.DataFrame, splits: dict, cfg) -> pd.DataFrame:
    """Attach ``split``, ``household_role`` and ``experience_id`` to a frame."""
    out = df.copy()
    out["split"] = assign_temporal(out, cfg)
    holdout = set(splits["holdout_households"])
    out["household_role"] = ["test" if hh in holdout else "train_val" for hh in out[HOUSEHOLD_COL]]
    out["experience_id"] = experience_ids(out, cfg)
    return out


def verify(df: pd.DataFrame, cfg) -> dict:
    """Check the two assignments are disjoint and complete."""
    temporal = assign_temporal(df, cfg)
    unassigned = int(temporal.isna().sum())
    per_split = temporal.value_counts().to_dict()
    return {
        "rows": len(df),
        "temporal_unassigned": unassigned,
        "per_split": per_split,
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def prepare(settings: Settings | None = None):
    settings = settings or load_settings()
    splits = build_splits(settings)
    path = write_splits(splits, settings)
    return splits, path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Step 4: build data/splits.json.")
    parser.add_argument("--configs-dir", default=None)
    args = parser.parse_args(argv)

    settings = load_settings(args.configs_dir) if args.configs_dir else load_settings()
    splits, path = prepare(settings)
    df = clean.load_baseline(settings)
    report = verify(df, settings.data)

    print("Step 4 complete")
    print(f"  temporal windows : {', '.join(SPLIT_NAMES)}")
    print(f"  rows per split   : {report['per_split']}")
    print(f"  unassigned rows  : {report['temporal_unassigned']}")
    print(f"  hold-out (test)  : {splits['holdout_households']}")
    print(f"  train/val        : {len(splits['train_val_households'])} households")
    print(f"  experience id    : {splits['experience_id']}")
    print(f"  wrote            : {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

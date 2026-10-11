"""Step 1 - load the raw CSV, drop derived columns, split out provided labels.

The input file is a clean synthetic single-city panel (13 households, 30-minute
interval, 2020) with 104 columns. 64 of them start at the ``hour`` column and are
*pre-computed* features (calendar, sin/cos, rolling, lag, ratio, z-score, flags).
Some of those include the current value (``power_2h_mean``, ``power_zscore``), so
using them would leak the target and they would not move when we inject an
anomaly. They are therefore dropped here and recomputed past-only later
(blueprint finding F3).

This module produces the two Step 1 artifacts under ``data/clean/``:

* ``measured.parquet`` - the measured/metadata columns plus a UTC timestamp;
* ``provided_labels.parquet`` - the 698 provided ``spike``/``dropout`` rows, kept
  separately as a sanity test set (finding F2), never as the benchmark.

Run it directly to (re)generate them::

    python -m energymind.io
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import pandas as pd

from energymind.config import REPO_ROOT, DataConfig, Settings, load_settings

#: Column identifying a household (stable across the dataset).
HOUSEHOLD_COL = "household_id"
#: UTC timestamp column added by this module.
UTC_COL = "ts_utc"
#: Header of the checksum manifest.
MANIFEST_COLUMNS = ["file", "sha256", "size_bytes", "rows", "columns"]

_CHUNK = 1 << 20


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def resolve(path: str | Path) -> Path:
    """Resolve a possibly-relative path against the repository root."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def sha256_file(path: str | Path) -> str:
    """Streaming SHA-256 of a file (does not load it into memory)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_env(path: str | Path | None = None) -> dict[str, str]:
    """Load ``.env`` into ``os.environ`` (existing variables are never overwritten).

    A tiny parser so the project needs no extra dependency; returns the values.
    """
    env_path = resolve(path or ".env")
    if not env_path.exists():
        return {}
    values: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        values[key] = value
        os.environ.setdefault(key, value)
    return values


# --------------------------------------------------------------------------- #
# Column selection
# --------------------------------------------------------------------------- #


def derived_columns(all_columns: list[str], cfg: DataConfig) -> list[str]:
    """Columns dropped as pre-computed features (finding F3).

    Positionally this is every column from ``derived_start`` onward (64 columns
    in this file). ``tariff_rate`` sits inside that block but is a genuine input
    the generator placed there, so it is re-added to the measured set via
    ``add_back`` and is *not* dropped -- leaving 63 columns actually removed.
    """
    start = all_columns.index(cfg.columns.derived_start)
    keep = set(cfg.columns.add_back)
    return [c for c in all_columns[start:] if c not in keep]


def select_measured_columns(all_columns: list[str], cfg: DataConfig) -> list[str]:
    """Measured + metadata columns, in file order, with labels removed.

    Everything before ``derived_start`` is measured/metadata; from there on the
    columns are derived. The provided label columns and ``drop_also`` are
    removed, and ``add_back`` columns (inputs the generator miscategorised) are
    re-appended.
    """
    start = all_columns.index(cfg.columns.derived_start)
    before = list(all_columns[:start])
    drop = set(cfg.columns.drop_also) | set(cfg.columns.label_columns)
    measured = [c for c in before if c not in drop]
    for col in cfg.columns.add_back:
        if col not in measured:
            measured.append(col)
    return measured


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_raw(csv_path: str | Path) -> pd.DataFrame:
    """Read the raw CSV exactly as it is (never mutate the file)."""
    path = resolve(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"raw CSV not found: {path}\n"
            "Place Austin_TX_US_2020.csv in data/raw/ (see DATA_LICENSE.md)."
        )
    return pd.read_csv(path)


def add_utc(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Attach a timezone-aware UTC timestamp.

    The timestamp column is naive local time and the file carries a constant
    ``timezone_offset_h`` with no DST, so ``ts_utc = local - offset``.
    """
    offset = pd.to_timedelta(df[cfg.time.timezone_offset_column], unit="h")
    ts = pd.to_datetime(df[cfg.time.timestamp_column]) - offset
    return df.assign(**{UTC_COL: ts.dt.tz_localize("UTC")})


def split_provided_labels(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Extract the provided anomaly rows as a standalone table.

    These are single-interval spikes/dropouts that break the electrical
    identities (finding F2); they are a sanity test set, not the benchmark.
    """
    flag_col, type_col = cfg.columns.label_columns
    return df.loc[
        df[flag_col] == 1,
        [HOUSEHOLD_COL, UTC_COL, type_col],
    ].reset_index(drop=True)


def load_measured(csv_path: str | Path, cfg: DataConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(measured, provided_labels)`` for the configured CSV."""
    df = load_raw(csv_path)
    df = add_utc(df, cfg)
    columns = select_measured_columns(list(df.columns), cfg)
    labels = split_provided_labels(df, cfg)
    measured = df[[*columns, UTC_COL]].copy()
    return measured, labels


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #


def write_manifest(
    csv_path: str | Path,
    manifest_path: str | Path,
    *,
    rows: int,
    columns: int,
) -> Path:
    """Write ``file, sha256, size_bytes, rows, columns`` for the raw CSV."""
    csv_path = resolve(csv_path)
    manifest_path = resolve(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "file": csv_path.name,
                "sha256": sha256_file(csv_path),
                "size_bytes": csv_path.stat().st_size,
                "rows": rows,
                "columns": columns,
            }
        ],
        columns=MANIFEST_COLUMNS,
    ).to_csv(manifest_path, index=False)
    return manifest_path


def verify_manifest(csv_path: str | Path, manifest_path: str | Path) -> dict[str, object]:
    """Check the raw CSV against the recorded SHA-256 and size."""
    csv_path = resolve(csv_path)
    manifest_path = resolve(manifest_path)
    if not manifest_path.exists():
        return {"verified": False, "reason": "manifest missing", "path": str(manifest_path)}
    if not csv_path.exists():
        return {"verified": False, "reason": "raw CSV missing", "path": str(csv_path)}
    row = pd.read_csv(manifest_path).iloc[0]
    actual = sha256_file(csv_path)
    expected = str(row["sha256"])
    size_ok = csv_path.stat().st_size == int(row["size_bytes"])
    return {
        "verified": actual == expected and size_ok,
        "sha256": actual,
        "expected": expected,
        "size_ok": size_ok,
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def prepare(settings: Settings | None = None) -> dict[str, object]:
    """Run Step 1: write ``measured.parquet`` and ``provided_labels.parquet``."""
    settings = settings or load_settings()
    cfg = settings.data

    raw_csv = resolve(cfg.paths.raw_csv)
    manifest = verify_manifest(cfg.paths.raw_csv, cfg.paths.manifest)

    measured, labels = load_measured(raw_csv, cfg)

    clean_dir = resolve(cfg.paths.clean_dir)
    clean_dir.mkdir(parents=True, exist_ok=True)
    measured_path = clean_dir / "measured.parquet"
    labels_path = clean_dir / "provided_labels.parquet"
    measured.to_parquet(measured_path, index=False)
    labels.to_parquet(labels_path, index=False)

    all_columns = list(pd.read_csv(raw_csv, nrows=0).columns)
    derived = derived_columns(all_columns, cfg)
    return {
        "raw_csv": raw_csv,
        "manifest_verified": manifest,
        "measured_path": measured_path,
        "labels_path": labels_path,
        "rows": len(measured),
        "households": int(measured[HOUSEHOLD_COL].nunique()),
        "measured_columns": measured.shape[1],
        "provided_labels": len(labels),
        "derived_dropped": derived,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Step 1: load, split columns, convert to UTC.")
    parser.add_argument("--configs-dir", default=None, help="directory of YAML configs")
    args = parser.parse_args(argv)

    settings = load_settings(args.configs_dir) if args.configs_dir else load_settings()
    result = prepare(settings)

    verified = result["manifest_verified"]
    status = "ok" if isinstance(verified, dict) and verified.get("verified") else "NOT VERIFIED"
    print("Step 1 complete")
    print(f"  raw csv           : {result['raw_csv']}")
    print(f"  manifest          : {status}")
    print(f"  rows              : {result['rows']}")
    print(f"  households        : {result['households']}")
    print(f"  measured columns  : {result['measured_columns']} (incl. ts_utc)")
    print(f"  provided labels   : {result['provided_labels']}")
    print(f"  derived dropped   : {len(result['derived_dropped'])}")  # type: ignore[arg-type]
    print(f"  wrote             : {result['measured_path']}")
    print(f"                      {result['labels_path']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

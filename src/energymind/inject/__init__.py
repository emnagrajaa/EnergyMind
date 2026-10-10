"""Step 5 - seeded, physics-consistent anomaly injection (blueprint Section 7).

15 anomaly types across 3 origins, each in a *naive* variant (single-channel
edit, the identities break) and a *consistent* variant (root edit + dependent
channels rebuilt). Injection happens only on ``quality_mask = 1`` rows, never
overlaps, and is fully determined by ``(prevalence, seed)``.

Outputs (``data/injected/``): ``labels.parquet`` (committed) plus, per run, an
``injected_p{p}_s{seed}.parquet`` rebuilt from the seed (gitignored).

    python -m energymind.inject --prevalence 0.05 --seed 1
"""

from __future__ import annotations

import argparse
import uuid

import numpy as np
import pandas as pd

from energymind import clean, io
from energymind import splits as splits_mod
from energymind.config import Settings, load_settings
from energymind.inject import adversarial as _adv
from energymind.inject import data_quality as _dq
from energymind.inject import physical as _phys
from energymind.inject.base import (
    UTC_COL,
    Grid,
    Injection,
    build_context,
    draw_start,
    mark,
    parse_duration,
    window_ok,
)

#: type name -> (function, kind)
REGISTRY: dict[str, tuple] = {
    "additive_bias": (_dq.additive_bias, "single"),
    "gain_error": (_dq.gain_error, "single"),
    "voltage_drift": (_dq.voltage_drift, "single"),
    "stuck_at": (_dq.stuck_at, "single"),
    "dropout": (_dq.dropout, "single"),
    "dead_sensor": (_dq.dead_sensor, "single"),
    "noise_burst": (_dq.noise_burst, "single"),
    "unit_error": (_dq.unit_error, "single"),
    "sub_meter_bias": (_dq.sub_meter_bias, "single"),
    "under_reporting": (_adv.under_reporting, "single"),
    "replay": (_adv.replay, "single"),
    "mimicry": (_adv.mimicry, "single"),
    "coherent_peak": (_phys.coherent_peak, "physical"),
    "neighborhood_outage": (_phys.neighborhood_outage, "physical"),
    "voltage_sag": (_phys.voltage_sag, "physical"),
}

SPLIT_CODE = {"train": 0, "val": 1, "test": 2}
ALL_SPLITS = (0, 1, 2)
LABEL_COLUMNS = [
    "anomaly_id",
    "household_id",
    "t_start",
    "t_end",
    "type",
    "origin",
    "severity",
    "variant",
    "scope",
    "params",
    "seed",
    "injector_version",
    "provenance",
]


def sample_params(spec_params: dict, rng) -> dict:
    """Draw one value per parameter (lists are sampled, scalars kept)."""
    out: dict = {}
    for key, val in spec_params.items():
        if isinstance(val, (list, tuple)):
            choice = rng.choice(val)
            out[key] = choice.item() if hasattr(choice, "item") else choice
        else:
            out[key] = val
    return out


def _positions(grid: Grid, hh: str, start: int, length: int) -> np.ndarray:
    return grid.rows[hh][start : start + length]


def _physical_targets(grid: Grid, name: str, params: dict, rng) -> list[str]:
    households = grid.households
    total = len(households)
    if name == "coherent_peak":
        share = float(params.get("min_household_share", 0.4))
        n_target = int(rng.integers(int(np.ceil(share * total)), total + 1))
    else:
        n_target = total
    return [str(hh) for hh in rng.choice(households, size=n_target, replace=False)]


def _common_start(grid: Grid, targets: list[str], length: int, allowed: dict, rng):
    n = grid.n
    if length > n:
        return None
    valid = np.ones(n - length + 1, dtype=bool)
    for hh in targets:
        starts = window_ok(grid.quality[hh], grid.occupied[hh], length, allowed.get(hh))
        row = np.zeros(n - length + 1, dtype=bool)
        row[starts] = True
        valid &= row
    starts = np.flatnonzero(valid)
    if len(starts) == 0:
        return None
    return int(rng.choice(starts))


def inject_all(
    df: pd.DataFrame,
    settings: Settings,
    prevalence: float,
    seed: int,
    *,
    types: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(injected_frame, labels_frame)`` for one ``(prevalence, seed)``."""
    cfg = settings.injection
    grid = Grid.from_frame(df)
    ctx = build_context(grid.df, settings)

    code = splits_mod.assign_temporal(grid.df, settings.data).map(SPLIT_CODE).to_numpy()
    withheld = set(cfg.withhold_from_train)
    names = [t for t in (types or list(cfg.types)) if t in cfg.types]

    rng = np.random.default_rng(seed)
    target_total = int(round(prevalence * len(grid.df)))
    per_type = target_total / max(len(names), 1)

    labels: list[Injection] = []
    counter = 0

    for name in names:
        fn, kind = REGISTRY[name]
        spec = cfg.types[name]
        lo, hi = (
            (parse_duration(spec.length.min), parse_duration(spec.length.max))
            if spec.length
            else (1, 1)
        )
        allowed_splits = [2] if name in withheld else list(ALL_SPLITS)

        def allowed_mask(hh: str, _splits: tuple = tuple(allowed_splits)) -> np.ndarray:
            return np.isin(code[grid.rows[hh]], _splits)

        cumulative = 0.0
        attempts = 0
        budget = max(200, int(per_type) * 25 + 200)

        while cumulative < per_type and attempts < budget:
            attempts += 1
            length = int(rng.integers(lo, hi + 1))
            params = sample_params(spec.params, rng)

            if kind == "single":
                hh = str(rng.choice(grid.households))
                start = draw_start(
                    grid.quality[hh], grid.occupied[hh], length, rng, allowed_mask(hh)
                )
                if start is None:
                    continue
                # A replay copies t-lag; that source must be untouched baseline,
                # otherwise a previously injected anomaly would be copied (and its
                # broken identities with it).
                if name == "replay":
                    lag = parse_duration(str(params["lag"]))
                    s0 = start - lag
                    if s0 < 0:
                        continue
                    src = slice(s0, s0 + length)
                    if grid.occupied[hh][src].any() or not grid.quality[hh][src].all():
                        continue
                pos = _positions(grid, hh, start, length)
                variant = str(rng.choice(spec.variants))
                info = fn(grid.df, pos, params, variant, rng, ctx, hh)
                mark(grid.occupied[hh], start, length)
                scope, household_id, cum = "single", hh, float(length)
                first_row = pos[0]
            else:
                variant = "consistent"
                targets = _physical_targets(grid, name, params, rng)
                allowed = {hh: allowed_mask(hh) for hh in targets}
                start = _common_start(grid, targets, length, allowed, rng)
                if start is None:
                    continue
                pairs = [(hh, _positions(grid, hh, start, length)) for hh in targets]
                info = fn(grid.df, pairs, params, variant, rng, ctx)
                for hh in targets:
                    mark(grid.occupied[hh], start, length)
                scope, household_id, cum = (
                    "neighborhood",
                    ",".join(sorted(targets)),
                    float(length * len(targets)),
                )
                first_row = pairs[0][1][0]

            counter += 1
            t_start = grid.df.loc[first_row, UTC_COL]
            t_end = t_start + pd.Timedelta(minutes=30 * length)
            effects = {k: v for k, v in info.items() if k not in ("severity", "scope")}
            labels.append(
                Injection(
                    anomaly_id=str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"energymind:{seed}:{prevalence}:{name}:{counter}",
                        )
                    ),
                    household_id=household_id,
                    t_start=t_start,
                    t_end=t_end,
                    type=name,
                    origin=spec.origin,
                    severity=info["severity"],
                    variant=variant,
                    scope=scope,
                    params={**params, **effects},
                    seed=seed,
                )
            )
            cumulative += cum

    rows = [x.as_row() for x in labels]
    labels_df = pd.DataFrame(rows, columns=LABEL_COLUMNS)
    return grid.df, labels_df


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def run(
    settings: Settings | None = None,
    *,
    prevalences: list[float] | None = None,
    seeds: list[int] | None = None,
    write_series: bool = True,
) -> dict:
    """Inject for every ``(prevalence, seed)`` and write labels (and series)."""
    settings = settings or load_settings()
    prevalences = prevalences or list(settings.injection.prevalence)
    seeds = seeds or list(settings.models.evaluation.seeds)

    df = clean.load_baseline(settings)
    out_dir = io.resolve(settings.data.paths.injected_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, object] = {}
    primary_labels = None
    for prevalence in prevalences:
        for seed in seeds:
            injected, labels = inject_all(df, settings, prevalence, seed)
            tag = f"p{prevalence}_s{seed}"
            labels_path = out_dir / f"labels_{tag}.parquet"
            labels.to_parquet(labels_path, index=False)
            if write_series:
                injected.to_parquet(out_dir / f"injected_{tag}.parquet", index=False)
            if primary_labels is None:
                primary_labels = labels
            written[tag] = {"labels": labels_path, "events": len(labels)}

    if primary_labels is not None:
        primary_labels.to_parquet(out_dir / "labels.parquet", index=False)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Step 5: anomaly injection.")
    parser.add_argument("--configs-dir", default=None)
    parser.add_argument("--prevalence", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-series", action="store_true", help="write labels only")
    args = parser.parse_args(argv)

    settings = load_settings(args.configs_dir) if args.configs_dir else load_settings()
    prevalences = [args.prevalence] if args.prevalence is not None else None
    seeds = [args.seed] if args.seed is not None else None
    written = run(settings, prevalences=prevalences, seeds=seeds, write_series=not args.no_series)

    print("Step 5 complete")
    for tag, info in written.items():
        print(f"  {tag}: {info['events']} events -> {info['labels']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

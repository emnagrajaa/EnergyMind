"""Step 8 - benchmark orchestration.

For each ``(prevalence, seed)`` it builds the injected feature table, fits every
detector on the training rows, chooses the alert threshold on validation at the
fixed false-alarm budget, then evaluates on the temporal test split and on the
household hold-out. The supervised classifier is trained on train events and
scored on test events. Every result is appended to ``results/runs.jsonl``.

    python -m energymind.benchmark --prevalence 0.05 --seeds 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from energymind import clean, evaluate, features, inject, io, nilm
from energymind.config import Settings, load_settings
from energymind.detect.ensemble import EnsembleDetector
from energymind.detect.iforest import IsolationForestDetector
from energymind.detect.lstm_ae import LSTMAutoencoder
from energymind.detect.matrix_profile import MatrixProfileDetector
from energymind.detect.pyod_models import ECODDetector
from energymind.detect.robust_z import RobustZ
from energymind.detect.rules import Rules
from energymind.detect.supervised import SupervisedDetector
from energymind.splits import assign_temporal

DETECTOR_FACTORIES = {
    "robust_z": lambda s: RobustZ(),
    "rules": lambda s: Rules(),
    "iforest": lambda s: IsolationForestDetector(
        n_estimators=int(s.models.detectors["iforest"].params.get("n_estimators", 200))
    ),
    "ecod": lambda s: ECODDetector(),
    "matrix_profile": lambda s: MatrixProfileDetector(),
    "ensemble": lambda s: EnsembleDetector(),
    "lstm_ae": lambda s: LSTMAutoencoder(
        window=int(s.models.detectors["lstm_ae"].params.get("window", 48)),
        epochs=int(s.models.detectors["lstm_ae"].params.get("epochs", 10)),
    ),
}
#: Detectors that need the injected labels at fit time.
LABELLED_DETECTORS = {"supervised"}
DEFAULT_DETECTORS = [
    "robust_z",
    "rules",
    "iforest",
    "ecod",
    "matrix_profile",
    "ensemble",
    "lstm_ae",
    "supervised",
]


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


def config_hash(settings: Settings) -> str:
    blob = json.dumps(
        {
            "data": settings.data.model_dump(mode="json"),
            "injection": settings.injection.model_dump(mode="json"),
            "models": settings.models.model_dump(mode="json"),
            "policy": settings.policy.model_dump(mode="json"),
            "agents": settings.agents.model_dump(mode="json"),
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def append_run(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _split_frames(frame: pd.DataFrame, cfg) -> dict[str, pd.DataFrame]:
    labels = assign_temporal(frame, cfg)
    return {name: frame[labels == name] for name in ("train", "val", "test")}


def _score_detector(
    detector, frame: pd.DataFrame, events: pd.DataFrame, settings: Settings
) -> tuple[pd.Series, dict]:
    """Fit on train; threshold on val; return global / per-household / oracle results."""
    parts = _split_frames(frame, settings.data)
    detector.fit(parts["train"])
    scores = detector.score(frame)

    budget = settings.models.evaluation.false_alarm_budget_per_household_week
    tol = settings.models.evaluation.tolerance_intervals
    val, test = parts["val"], parts["test"]

    def metrics(subset, threshold):
        return evaluate.evaluate(subset, scores.loc[subset.index], events, threshold, budget, tol)[
            0
        ]

    holdout = set(
        json.loads(io.resolve(settings.data.paths.splits_file).read_text())["holdout_households"]
    )
    hold = frame[frame[io.HOUSEHOLD_COL].isin(holdout)]

    # global: event-F1-optimal threshold within the false-alarm budget
    thr = evaluate.select_threshold(val, scores.loc[val.index], events, budget, tol)
    out = {
        "global": {
            "val": metrics(val, thr),
            "test": metrics(test, thr),
            "holdout": metrics(hold, thr),
            "threshold": thr,
        }
    }
    # per-household thresholds
    thr_ph = evaluate.select_threshold_per_household(
        val, scores.loc[val.index], events, budget, tol
    )
    out["perhousehold"] = {"test": metrics(test, thr_ph)}
    # oracle ceiling on the test split
    thr_oracle = evaluate.oracle_threshold(test, scores.loc[test.index], events, tol)
    out["oracle"] = {"test": metrics(test, thr_oracle)}
    return scores, out


def _train_classifiers(
    frame: pd.DataFrame, events: pd.DataFrame, scores: dict[str, pd.Series], settings: Settings
) -> dict:
    from energymind.classify import (
        EventClassifier,
        assign_event_split,
        build_event_dataset,
        classifier_metrics,
    )

    x, meta = build_event_dataset(frame, events, scores)
    meta = meta.assign(split=assign_event_split(meta, settings, frame))
    train = meta["split"] == "train"
    val = meta["split"] == "val"
    test = meta["split"] == "test"
    if train.sum() < 5 or test.sum() < 1:
        return {}

    out: dict = {}
    best = None
    for name in ("lightgbm", "random_forest"):
        model = EventClassifier.fit(
            x[train], meta.loc[train, "type"], meta.loc[train, "origin"], name
        )
        val_metrics = classifier_metrics(
            meta.loc[val, "type"], meta.loc[val, "origin"], model.predict(x[val])
        )
        if best is None or val_metrics["type_macro_f1"] > best[1]["type_macro_f1"]:
            best = (model, val_metrics, name)
        out[name] = {"val": val_metrics}
    model, _, chosen = best
    test_pred = model.predict(x[test])
    test_metrics = classifier_metrics(meta.loc[test, "type"], meta.loc[test, "origin"], test_pred)
    out["chosen"] = chosen
    out["test"] = test_metrics
    out["n_events"] = {"train": int(train.sum()), "val": int(val.sum()), "test": int(test.sum())}
    return out


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def run(
    settings: Settings | None = None,
    *,
    prevalences: list[float] | None = None,
    seeds: list[int] | None = None,
    detectors: list[str] | None = None,
    include_classifier: bool = True,
    runs_path: Path | None = None,
) -> pd.DataFrame:
    """Run the benchmark grid and append results to ``results/runs.jsonl``."""
    settings = settings or load_settings()
    prevalences = prevalences or list(settings.models.evaluation.prevalences)
    seeds = seeds or list(settings.models.evaluation.seeds)
    names = detectors or DEFAULT_DETECTORS
    runs_path = runs_path or io.resolve("results/runs.jsonl")

    clean_df = clean.load_baseline(settings)
    engineered = features.engineer_features(clean_df, features.train_end_of(settings))
    nilm_model, nilm_report = nilm.fit_and_evaluate(engineered, settings)

    commit = git_commit()
    cfg_hash = config_hash(settings)
    records = []
    rows = []

    for prevalence in prevalences:
        for seed in seeds:
            injected, labels = inject.inject_all(clean_df, settings, prevalence, seed)
            frame = features.build_features(injected, settings, nilm_model=nilm_model)
            events = evaluate.expand_events(labels)

            scores: dict[str, pd.Series] = {}
            for name in names:
                if name in LABELLED_DETECTORS:
                    detector = SupervisedDetector(
                        events=events,
                        tol_intervals=settings.models.evaluation.tolerance_intervals,
                    )
                else:
                    detector = DETECTOR_FACTORIES[name](settings)
                scores[name], result = _score_detector(detector, frame, events, settings)
                variants = {
                    name: result["global"],
                    f"{name}+ph": {"test": result["perhousehold"]["test"]},
                    f"{name}.oracle": {"test": result["oracle"]["test"]},
                }
                for variant, splits in variants.items():
                    for split, metrics in splits.items():
                        if split == "threshold":
                            continue
                        records.append(
                            {
                                "task": "detection",
                                "model": variant,
                                "thresholding": "per_household"
                                if variant.endswith("+ph")
                                else "oracle"
                                if variant.endswith(".oracle")
                                else "global_event",
                                "prevalence": prevalence,
                                "seed": seed,
                                "split": split,
                                "git_commit": commit,
                                "config_hash": cfg_hash,
                                "ts": datetime.now(UTC).isoformat(),
                                **metrics,
                            }
                        )
                    if "test" in splits:
                        rows.append(
                            {
                                "task": "detection",
                                "model": variant,
                                "prevalence": prevalence,
                                "seed": seed,
                                **{
                                    f"test_{k}": v
                                    for k, v in splits["test"].items()
                                    if not isinstance(v, dict)
                                },
                            }
                        )

            if include_classifier and scores:
                clf = _train_classifiers(frame, events, scores, settings)
                if clf:
                    records.append(
                        {
                            "task": "classification",
                            "model": clf["chosen"],
                            "prevalence": prevalence,
                            "seed": seed,
                            "split": "test",
                            "git_commit": commit,
                            "config_hash": cfg_hash,
                            "ts": datetime.now(UTC).isoformat(),
                            **clf["test"],
                            "n_events": clf["n_events"],
                        }
                    )
                    rows.append(
                        {
                            "task": "classification",
                            "model": clf["chosen"],
                            "prevalence": prevalence,
                            "seed": seed,
                            **clf["test"],
                        }
                    )

    for record in records:
        append_run(runs_path, record)
    _ = np  # keep numpy import for parity with metric dtypes
    _ = nilm_report
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Step 8: detection + classification benchmark.")
    parser.add_argument("--configs-dir", default=None)
    parser.add_argument("--prevalence", type=float, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--detectors", nargs="+", default=None)
    parser.add_argument("--no-classifier", action="store_true")
    args = parser.parse_args(argv)

    settings = load_settings(args.configs_dir) if args.configs_dir else load_settings()
    results = run(
        settings,
        prevalences=[args.prevalence] if args.prevalence is not None else None,
        seeds=args.seeds,
        detectors=args.detectors,
        include_classifier=not args.no_classifier,
    )
    print("Step 8 complete")
    detection = results[results["task"] == "detection"]
    if not detection.empty:
        summary = detection.groupby("model")["test_event_f1"].agg(["mean", "std"])
        print(summary.round(4).to_string())
    classification = results[results["task"] == "classification"]
    if not classification.empty:
        print(
            classification[["model", "type_macro_f1", "origin_accuracy", "origin_ece"]]
            .round(4)
            .to_string(index=False)
        )
    print(f"  runs -> {io.resolve('results/runs.jsonl')}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

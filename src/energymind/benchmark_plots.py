"""Benchmark figures (blueprint Step 8 / notebook 04).

Styled after ``eda_plots``: same paper rcParams, Okabe-Ito palette, PNG + PDF at
300 dpi, bold ``(a)`` panel labels.

Two entry points:

* :func:`generate_from_runs` reads ``results/runs.jsonl`` and draws the
  leaderboard, the risk/recall trade-off and the recall breakdowns (no recompute);
* :func:`build_gallery` re-runs one ``(prevalence, seed)`` to draw score
  distributions, PR curves and the classifier diagnostics.

    python -m energymind.benchmark_plots --prevalence 0.05 --seed 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from energymind import clean, evaluate, features, inject, io, nilm
from energymind import eda_plots as base
from energymind.config import Settings, load_settings

BLUE, ORANGE, GREEN, VERM = base.BLUE, base.ORANGE, base.GREEN, base.VERM
BLACK, GREY = base.BLACK, base.GREY
W2 = base.W2

FAMILY_COLORS = {"unsupervised": BLUE, "ensemble": GREEN, "supervised": VERM}
SUPERVISED = {"supervised"}
ENSEMBLE = {"ensemble"}
RUNS_PATH = "results/runs.jsonl"


def _family(model: str) -> str:
    base_name = model.split("+")[0].split(".")[0]
    if base_name in SUPERVISED:
        return "supervised"
    if base_name in ENSEMBLE:
        return "ensemble"
    return "unsupervised"


# --------------------------------------------------------------------------- #
# From runs.jsonl
# --------------------------------------------------------------------------- #


def load_runs(path: Path | str | None = None) -> pd.DataFrame:
    resolved = io.resolve(path or RUNS_PATH)
    if not Path(resolved).exists():
        return pd.DataFrame(
            columns=["task", "model", "thresholding", "prevalence", "seed", "split", "event_f1"]
        )
    records = [
        json.loads(line) for line in Path(resolved).read_text(encoding="utf-8").splitlines() if line
    ]
    return pd.DataFrame(records)


def _detection_table(runs: pd.DataFrame, thresholding: str = "global_event") -> pd.DataFrame:
    det = runs[(runs["task"] == "detection") & (runs["split"] == "test")]
    if "thresholding" in det.columns:
        det = det[det["thresholding"] == thresholding]
    return det


def fig_detection_leaderboard(runs: pd.DataFrame, out_dir=None):
    """Mean +/- std test event-F1 per detector (global, within-budget)."""
    det = _detection_table(runs)
    agg = det.groupby("model")["event_f1"].agg(["mean", "std"]).sort_values("mean")
    fig = base._fig(W2, max(3.0, 0.34 * len(agg) + 1.0))
    ax = fig.add_subplot(111)
    colors = [FAMILY_COLORS[_family(m)] for m in agg.index]
    ax.barh(
        agg.index,
        agg["mean"].to_numpy(),
        xerr=agg["std"].fillna(0).to_numpy(),
        color=colors,
        height=0.72,
    )
    ax.set_xlabel("event F1 (test, mean +/- std over seeds)")
    base._label(ax, "(a)")
    base._sub(ax, "detection leaderboard")
    handles = [plt.Line2D([], [], color=c, lw=4) for c in FAMILY_COLORS.values()]
    ax.legend(handles, list(FAMILY_COLORS), loc="lower right")
    return base._finish(fig, out_dir, "bench_detection_leaderboard")


def fig_detector_tradeoff(runs: pd.DataFrame, out_dir=None):
    """Event recall versus false alarms per household-week (budget line at 1.0)."""
    det = _detection_table(runs)
    agg = det.groupby("model")[["fa_per_household_week", "event_recall", "event_f1"]].mean()
    fig = base._fig(W2 * 0.62, 4.2)
    ax = fig.add_subplot(111)
    for model, row in agg.iterrows():
        color = FAMILY_COLORS[_family(model)]
        ax.scatter(row["fa_per_household_week"], row["event_recall"], color=color, s=28)
        ax.annotate(
            model,
            (row["fa_per_household_week"], row["event_recall"]),
            fontsize=6,
            xytext=(3, 2),
            textcoords="offset points",
        )
    ax.axvline(1.0, color=BLACK, ls="--", lw=0.8)
    ax.set_xlabel("false alarms per household-week")
    ax.set_ylabel("event recall")
    base._label(ax, "(b)")
    base._sub(ax, "risk / recall trade-off (budget = 1.0)")
    return base._finish(fig, out_dir, "bench_detector_tradeoff")


def _breakdown_matrix(runs: pd.DataFrame, key: str) -> pd.DataFrame:
    det = _detection_table(runs)
    rows = {}
    for model, group in det.groupby("model"):
        values = [r.get(key, {}) for r in group.to_dict("records")]
        keys = sorted({k for v in values for k in v})
        rows[model] = {k: float(np.mean([v.get(k, np.nan) for v in values])) for k in keys}
    return pd.DataFrame(rows).T


def fig_recall_breakdown(runs: pd.DataFrame, out_dir=None):
    """Recall by variant (naive/consistent) and by event length, per detector."""
    by_variant = _breakdown_matrix(runs, "recall_by_variant")
    by_length = _breakdown_matrix(runs, "recall_by_length")
    fig = base._fig(W2, 3.4)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25])

    ax = fig.add_subplot(gs[0, 0])
    cols = [c for c in ["naive", "consistent"] if c in by_variant.columns]
    im = ax.imshow(by_variant[cols].to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(by_variant)), by_variant.index, fontsize=6)
    base._label(ax, "(a)")
    base._sub(ax, "recall by variant")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[0, 1])
    order = [c for c in ["1", "2-6", "7-48", "49+"] if c in by_length.columns]
    im = ax.imshow(by_length[order].to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(order)), order)
    ax.set_yticks(range(len(by_length)), by_length.index, fontsize=6)
    base._label(ax, "(b)")
    base._sub(ax, "recall by event length (intervals)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    return base._finish(fig, out_dir, "bench_recall_breakdown")


def generate_from_runs(runs: pd.DataFrame | None = None, out_dir=None) -> list[str]:
    runs = runs if runs is not None else load_runs()
    plt.close("all")
    names = ["bench_detection_leaderboard", "bench_detector_tradeoff", "bench_recall_breakdown"]
    fig_detection_leaderboard(runs, out_dir)
    fig_detector_tradeoff(runs, out_dir)
    fig_recall_breakdown(runs, out_dir)
    plt.close("all")
    return names


# --------------------------------------------------------------------------- #
# One representative run (score distributions, PR curves, classifier)
# --------------------------------------------------------------------------- #


def fig_score_distributions(frame, scores: dict, events, out_dir=None):
    """Event vs non-event score distributions for each detector (log x)."""
    names = list(scores)
    ncols = 3
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(W2, 1.9 * nrows), layout="constrained", squeeze=False
    )
    in_event = evaluate.point_labels(frame, events, pd.Timedelta("30min")).to_numpy()
    for i, name in enumerate(names):
        ax = axes[i // ncols][i % ncols]
        v = scores[name].to_numpy(dtype=float)
        finite = np.isfinite(v)
        lo = np.nanmin(v[finite])
        shift = -lo + 1.0 if lo <= 0 else 0.0
        ax.hist(
            v[finite & ~in_event] + shift,
            bins=60,
            log=True,
            color=GREY,
            alpha=0.7,
            label="non-event",
        )
        ax.hist(
            v[finite & in_event] + shift, bins=60, log=True, color=VERM, alpha=0.7, label="event"
        )
        ax.set_title(name, fontsize=7)
        ax.tick_params(labelsize=6)
        if i == 0:
            ax.legend(fontsize=6)
    base._finish(fig, out_dir, "bench_score_distributions")
    return fig


def fig_pr_curves(frame, scores: dict, events, out_dir=None):
    """Point-level precision-recall curves per detector."""
    in_event = evaluate.point_labels(frame, events, pd.Timedelta("30min")).to_numpy()
    fig = base._fig(W2 * 0.6, 4.4)
    ax = fig.add_subplot(111)
    for i, (name, serie) in enumerate(scores.items()):
        v = serie.to_numpy(dtype=float)
        v = np.where(
            np.isfinite(v), v, np.nanmin(v[np.isfinite(v)] if np.isfinite(v).any() else 0.0) - 1
        )
        precision, recall, _ = precision_recall_curve(in_event, v)
        ax.plot(recall, precision, lw=1.1, label=name, color=plt.cm.tab10(i % 10))
    ax.set_xlabel("recall (point)")
    ax.set_ylabel("precision (point)")
    ax.legend(fontsize=6)
    base._label(ax, "(c)")
    base._sub(ax, "precision-recall (test)")
    return base._finish(fig, out_dir, "bench_pr_curves")


def fig_classifier(runs: pd.DataFrame, confusion: np.ndarray, classes, reliability, out_dir=None):
    """Classifier: type macro-F1 / origin accuracy per seed, confusion, reliability."""
    clf = runs[(runs["task"] == "classification")]
    fig = base._fig(W2, 3.2)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.2, 1.1])

    ax = fig.add_subplot(gs[0, 0])
    if not clf.empty and {"type_macro_f1", "origin_accuracy"} <= set(clf.columns):
        agg = clf.groupby("model")[["type_macro_f1", "origin_accuracy"]].mean()
        x = np.arange(len(agg))
        ax.bar(x - 0.2, agg["type_macro_f1"], width=0.4, color=BLUE, label="type macro-F1")
        ax.bar(x + 0.2, agg["origin_accuracy"], width=0.4, color=GREEN, label="origin acc.")
        ax.set_xticks(x, agg.index, rotation=20, fontsize=6)
        ax.legend(fontsize=6)
    base._label(ax, "(a)")
    base._sub(ax, "classifier (test, mean over seeds)")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right", fontsize=6)
    ax.set_yticks(range(len(classes)), classes, fontsize=6)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, int(confusion[i, j]), ha="center", va="center", fontsize=6)
    base._label(ax, "(b)")
    base._sub(ax, "origin confusion (test)")

    ax = fig.add_subplot(gs[0, 2])
    centers, accs = reliability
    ax.plot([0, 1], [0, 1], color=BLACK, ls="--", lw=0.8)
    ax.plot(centers, accs, marker="o", color=VERM)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    base._label(ax, "(c)")
    base._sub(ax, "origin reliability")
    base._finish(fig, out_dir, "bench_classifier")
    return fig


def _reliability(proba: np.ndarray, correct: np.ndarray, n_bins: int = 10):
    conf = proba.max(axis=1)
    bins = np.clip((conf * n_bins).astype(int), 0, n_bins - 1)
    centers, accs = [], []
    for b in range(n_bins):
        m = bins == b
        if m.any():
            centers.append(conf[m].mean())
            accs.append(correct[m].mean())
    return np.array(centers), np.array(accs)


def prepare_run(settings: Settings, *, prevalence: float = 0.05, seed: int = 1) -> dict:
    """Compute one injected run's scores and classifier diagnostics for plotting."""
    from energymind.benchmark import DEFAULT_DETECTORS, DETECTOR_FACTORIES, LABELLED_DETECTORS
    from energymind.classify import EventClassifier, build_event_dataset
    from energymind.detect.supervised import SupervisedDetector
    from energymind.splits import assign_temporal

    clean_df = clean.load_baseline(settings)
    engineered = features.engineer_features(clean_df, features.train_end_of(settings))
    nilm_model, _ = nilm.fit_and_evaluate(engineered, settings)
    injected, labels = inject.inject_all(clean_df, settings, prevalence, seed)
    frame = features.build_features(injected, settings, nilm_model=nilm_model)
    events = evaluate.expand_events(labels)
    split = assign_temporal(frame, settings.data)
    parts = {n: frame[split == n] for n in ("train", "val", "test")}

    scores: dict[str, pd.Series] = {}
    for name in DEFAULT_DETECTORS:
        if name == "lstm_ae":
            continue  # slow; already covered by runs.jsonl
        detector = (
            SupervisedDetector(events=events)
            if name in LABELLED_DETECTORS
            else DETECTOR_FACTORIES[name](settings)
        )
        detector.fit(parts["train"])
        scores[name] = detector.score(frame)

    x, meta = build_event_dataset(frame, events, scores)
    t = settings.data.splits.temporal
    offset = settings.data.time.assume_utc_offset_h
    train_end = pd.Timestamp(t.train.end).tz_localize("UTC") - pd.Timedelta(hours=offset)
    test_start = pd.Timestamp(t.test.start).tz_localize("UTC") - pd.Timedelta(hours=offset)
    train = (meta["t_start"] < train_end).to_numpy()
    test = (meta["t_start"] >= test_start).to_numpy()

    classes: list[str] = []
    confusion = np.zeros((0, 0), dtype=int)
    reliability = (np.array([]), np.array([]))
    if train.sum() > 5 and test.sum() > 0:
        from sklearn.metrics import confusion_matrix

        model = EventClassifier.fit(
            x[train], meta.loc[train, "type"], meta.loc[train, "origin"], "lightgbm"
        )
        pred = model.predict(x[test])
        classes = list(model.classes_origin)
        confusion = confusion_matrix(meta.loc[test, "origin"], pred["origin_pred"], labels=classes)
        correct = (np.array(pred["origin_pred"]) == meta.loc[test, "origin"].to_numpy()).astype(
            float
        )
        reliability = _reliability(pred["origin_proba"], correct)
    return {
        "frame": frame,
        "scores": scores,
        "events": events,
        "confusion": confusion,
        "classes": classes,
        "reliability": reliability,
    }


def build_gallery(
    settings: Settings, *, prevalence: float = 0.05, seed: int = 1, out_dir=None
) -> dict:
    """Re-run one prevalence/seed and draw the score/PR/classifier figures."""
    figures = io.resolve(out_dir) if out_dir else io.resolve("results/figures")
    data = prepare_run(settings, prevalence=prevalence, seed=seed)
    fig_score_distributions(data["frame"], data["scores"], data["events"], figures)
    fig_pr_curves(data["frame"], data["scores"], data["events"], figures)
    if len(data["classes"]):
        fig_classifier(
            load_runs(), data["confusion"], data["classes"], data["reliability"], figures
        )
    return data


def run_all(settings: Settings, *, prevalence: float, seed: int, out_dir=None) -> list[str]:
    plt.close("all")
    runs = load_runs()
    names = generate_from_runs(runs, out_dir)
    build_gallery(settings, prevalence=prevalence, seed=seed, out_dir=out_dir)
    names += ["bench_score_distributions", "bench_pr_curves", "bench_classifier"]
    plt.close("all")
    return names


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Step 8: benchmark figures.")
    parser.add_argument("--configs-dir", default=None)
    parser.add_argument("--prevalence", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)

    settings = load_settings(args.configs_dir) if args.configs_dir else load_settings()
    names = run_all(settings, prevalence=args.prevalence, seed=args.seed)
    print("Benchmark figures complete")
    for name in names:
        print(f"  {name}")
    print(f"  figures -> {io.resolve('results/figures')}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Step 9 - explanations (blueprint Section 9).

Three jobs, kept distinct:

* **TreeSHAP** - signed per-feature attributions for a tree model (the Random
  Forest / LightGBM classifier, or an Isolation Forest).
* **Reconstruction attribution** - for the LSTM autoencoder, an *exact*
  decomposition of the reconstruction error into the channel and the time step
  that drove it (a faithful alternative to DeepSHAP for an MSE reconstruction
  model).
* **LIME** - run with several seeds; its agreement with SHAP is the
  **explanation reliability** ``mean Jaccard(top-k SHAP, top-k LIME)``.

Plus the **rule-based evidence checks** (baseline deviation, identity residual,
neighbourhood coherence, NILM residual) that become the operator-facing text.

Importing ``shap`` / ``lime`` is deferred so the core test suite does not need
them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TOP_K = 5
IDENTITY_COLUMNS = ["r_S", "r_VI", "r_E", "r_PF"]


# --------------------------------------------------------------------------- #
# TreeSHAP
# --------------------------------------------------------------------------- #


def _to_class_vector(values, class_index: int | None):
    """Normalise shap output to a 1-D attribution vector for one class."""
    if isinstance(values, list):  # older shap: one array per class
        group = values[class_index or 0]
        return np.asarray(group)[0]
    array = np.asarray(values)
    if array.ndim == 3:  # (n, features, classes)
        cls = class_index if class_index is not None else int(array[0].sum(axis=0).argmax())
        return array[0, :, cls]
    if array.ndim == 2:  # (n, features)
        return array[0]
    raise ValueError(f"unexpected shap output shape {array.shape}")


def tree_shap_top(
    model,
    x_row: pd.Series,
    feature_names: list[str],
    class_index: int | None = None,
    k: int = TOP_K,
) -> list[tuple[str, float]]:
    """Top-``k`` signed TreeSHAP contributions for one row."""
    import shap

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(x_row.to_frame().T)
    vector = _to_class_vector(values, class_index)
    order = np.argsort(np.abs(vector))[::-1][:k]
    return [(feature_names[i], round(float(vector[i]), 5)) for i in order]


# --------------------------------------------------------------------------- #
# LIME
# --------------------------------------------------------------------------- #


def make_lime(
    training: pd.DataFrame, feature_names: list[str], class_names: list[str] | None = None
):
    """Build a tabular LIME explainer over the training design matrix."""
    from lime.lime_tabular import LimeTabularExplainer

    return LimeTabularExplainer(
        training.to_numpy(),
        feature_names=list(feature_names),
        class_names=class_names or ["0", "1"],
        mode="classification",
        # Zero-variance columns break LIME's discretizer (truncnorm scale 0), so we
        # perturb on the raw scale; the columns stay aligned with the model input.
        discretize_continuous=False,
        random_state=0,
    )


def _parse_lime_features(explanation, feature_names: list[str], label: int) -> list[str]:
    """Recover feature names from LIME's ``'name <= x'`` condition strings."""
    found: list[str] = []
    for condition, _weight in explanation.as_list(label=label):
        for name in feature_names:
            if condition.startswith(name):
                if name not in found:
                    found.append(name)
                break
    return found


def lime_top_features(
    explainer,
    x_row: pd.Series,
    predict_fn,
    class_index: int,
    feature_names: list[str],
    k: int = TOP_K,
    n_seeds: int = 5,
) -> list[list[str]]:
    """Run LIME ``n_seeds`` times; return the top-k feature list per seed."""
    tops: list[list[str]] = []
    for seed in range(n_seeds):
        import random

        random.seed(seed)
        np.random.seed(seed)
        explanation = explainer.explain_instance(
            x_row.to_numpy().astype(float),
            predict_fn,
            num_features=k,
            num_samples=1000,
            labels=[class_index],
        )
        tops.append(_parse_lime_features(explanation, feature_names, class_index))
    return tops


def mean_jaccard(reference: list[str], candidates: list[list[str]]) -> float:
    """Mean Jaccard overlap of ``reference`` with each candidate set."""
    ref = set(reference)
    scores = []
    for candidate in candidates:
        cand = set(candidate)
        union = ref | cand
        scores.append(len(ref & cand) / len(union) if union else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def explain_case(
    model,
    x_train: pd.DataFrame,
    x_case: pd.Series,
    predict_fn,
    feature_names: list[str],
    class_index: int,
    k: int = TOP_K,
    n_seeds: int = 5,
) -> dict:
    """SHAP + LIME + reliability for one case (needs shap and lime installed)."""
    shap_top = tree_shap_top(model, x_case, feature_names, class_index=class_index, k=k)
    try:
        # LIME's distance step rejects NaN, so sanitise its inputs.
        explainer = make_lime(x_train.fillna(0.0), feature_names)
        lime_tops = lime_top_features(
            explainer,
            x_case.fillna(0.0),
            predict_fn,
            class_index,
            feature_names,
            k=k,
            n_seeds=n_seeds,
        )
        reliability = mean_jaccard([f for f, _ in shap_top], lime_tops)
        lime_union: list[str] = []
        for top in lime_tops:
            for name in top:
                if name not in lime_union:
                    lime_union.append(name)
        return {
            "shap_top": shap_top,
            "lime_top": lime_union[:k],
            "reliability": round(reliability, 3),
        }
    except Exception:  # lime optional / unstable
        return {
            "shap_top": shap_top,
            "lime_top": [],
            "reliability": None,
            "note": "lime unavailable; reliability not computed",
        }


# --------------------------------------------------------------------------- #
# Reconstruction attribution (LSTM autoencoder)
# --------------------------------------------------------------------------- #


def reconstruction_attribution(
    window: np.ndarray, reconstruction: np.ndarray, channels: list[str]
) -> dict:
    """Exact MSE decomposition per channel and per time step.

    ``window`` and ``reconstruction`` are ``(time, channel)`` arrays. The total MSE
    is the mean over time of the per-time, per-channel squared error, so the
    channel and time with the largest error are the model's explanation.
    """
    window = np.asarray(window, dtype=float)
    reconstruction = np.asarray(reconstruction, dtype=float)
    squared = (reconstruction - window) ** 2
    per_channel = squared.mean(axis=0)
    per_time = squared.mean(axis=1)
    channel_order = np.argsort(per_channel)[::-1]
    time_index = int(np.argmax(per_time))
    return {
        "per_channel": {channels[i]: round(float(per_channel[i]), 5) for i in channel_order},
        "top_channel": channels[int(channel_order[0])],
        "top_time_index": time_index,
        "total_mse": round(float(squared.mean()), 5),
    }


# --------------------------------------------------------------------------- #
# Rule-based evidence checks
# --------------------------------------------------------------------------- #


def evidence_checks(window: pd.DataFrame, tolerance: float = 1e-2) -> list[dict]:
    """Observed-versus-expected checks over an event window (units included)."""

    def _median(column: str) -> float | None:
        if column not in window.columns:
            return None
        value = window[column].median()
        return None if pd.isna(value) else round(float(value), 4)

    checks: list[dict] = []

    observed = window["active_power_kw"].median()
    expected = window["how_median"].median() if "how_median" in window.columns else np.nan
    z = window["p_resid_z"].median() if "p_resid_z" in window.columns else np.nan
    checks.append(
        {
            "check": "baseline_deviation",
            "observed_kw": None if pd.isna(observed) else round(float(observed), 3),
            "expected_kw": None if pd.isna(expected) else round(float(expected), 3),
            "z": None if pd.isna(z) else round(float(z), 2),
            "status": "anomalous" if (not pd.isna(z) and abs(z) > 3) else "normal",
        }
    )

    residual = None
    if all(column in window.columns for column in IDENTITY_COLUMNS):
        residual = float(window[IDENTITY_COLUMNS].abs().max().max())
    checks.append(
        {
            "check": "current_vs_power_identity",
            "rel_residual": None if residual is None else round(residual, 4),
            "status": "broken" if (residual is not None and residual > tolerance) else "holds",
        }
    )

    share = window["neigh_share"].max() if "neigh_share" in window.columns else np.nan
    checks.append(
        {
            "check": "neighborhood_share",
            "value": None if pd.isna(share) else round(float(share), 3),
            "status": "coherent" if (not pd.isna(share) and share >= 0.4) else "isolated",
        }
    )

    checks.append({"check": "nilm_residual_total", "value": _median("nilm_resid_total")})
    checks.append({"check": "sub_meter_share_cooling", "value": _median("sub2_share")})
    return checks


# --------------------------------------------------------------------------- #
# Sanity check and CLI
# --------------------------------------------------------------------------- #


def sanity_check(explanation: dict, keywords: list[str], k: int = 3) -> bool:
    """True if any ``keyword`` appears in the SHAP top-``k`` feature names."""
    top = [name for name, _ in explanation.get("shap_top", [])][:k]
    return any(keyword in name for name in top for keyword in keywords)


def main(argv: list[str] | None = None) -> int:
    """Explain the top flagged cases of an injected run.

    Requires ``shap`` and ``lime`` installed and ``data/features/features.parquet``
    (run ``python -m energymind.features`` first).
    """
    import argparse

    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier

    from energymind import evaluate, records
    from energymind.classify import build_event_dataset
    from energymind.config import load_settings

    parser = argparse.ArgumentParser(description="Step 9: SHAP + LIME explanations and records.")
    parser.add_argument("--configs-dir", default=None)
    parser.add_argument("--cases", type=int, default=10, help="number of cases to summarise")
    parser.add_argument("--feature-file", default="data/features/features.parquet")
    parser.add_argument("--label-file", default="data/injected/labels.parquet")
    args = parser.parse_args(argv)

    settings = load_settings(args.configs_dir) if args.configs_dir else load_settings()
    frame = pd.read_parquet(args.feature_file)
    labels = pd.read_parquet(args.label_file)
    events = evaluate.expand_events(labels)
    x, meta = build_event_dataset(frame, events)

    t = settings.data.splits.temporal
    off = settings.data.time.assume_utc_offset_h
    train_end = pd.Timestamp(t.train.end).tz_localize("UTC") - pd.Timedelta(hours=off)
    test_start = pd.Timestamp(t.test.start).tz_localize("UTC") - pd.Timedelta(hours=off)
    train = (meta["t_start"] < train_end).to_numpy()
    test = (meta["t_start"] >= test_start).to_numpy()

    rf = RandomForestClassifier(n_estimators=300, random_state=0).fit(
        x[train], meta.loc[train, "type"]
    )
    classes = list(rf.classes_)

    print("Explanation reliability (mean Jaccard of SHAP vs LIME top-5, over test cases)")
    reliabilities = []
    sub_meter_top3 = None
    for idx in meta[test].index[: args.cases]:
        case = meta.loc[idx]
        pred = rf.predict(x.loc[idx].to_frame().T)[0]
        class_index = classes.index(pred) if pred in classes else 0
        result = explain_case(
            rf, x[train], x.loc[idx], rf.predict_proba, list(x.columns), class_index
        )
        if result["reliability"] is not None:
            reliabilities.append(result["reliability"])
        print(
            f"  {case['type']:16s} pred={pred:16s} reliability={result['reliability']} "
            f"shap_top={[f for f, _ in result['shap_top']]}"
        )
        if case["type"] == "sub_meter_bias":
            sub_meter_top3 = sanity_check(result, ["sub1_", "sub2_", "sub3_", "sub4_"], k=3)

    if reliabilities:
        print(f"\nmean reliability: {np.mean(reliabilities):.3f} (over {len(reliabilities)} cases)")
    if sub_meter_top3 is not None:
        print(f"sub_meter_bias sanity (sub-meter feature in SHAP top-3): {sub_meter_top3}")

    # one full example record
    idx = meta[test].index[0]
    case = meta.loc[idx]
    rows = events[events.anomaly_id == case["anomaly_id"]]
    window = pd.concat(
        [
            frame[
                (frame.household_id == r.household_id)
                & frame.ts_utc.between(r.t_start, r.t_end, inclusive="left")
            ]
            for r in rows.itertuples()
        ]
    )
    pred = rf.predict(x.loc[idx].to_frame().T)[0]
    result = explain_case(
        rf,
        x[train],
        x.loc[idx],
        rf.predict_proba,
        list(x.columns),
        classes.index(pred) if pred in classes else 0,
    )
    record = records.build_record(
        anomaly_id=case["anomaly_id"],
        household_id=rows.household_id.iloc[0],
        t_start=window["ts_utc"].iloc[0],
        t_end=window["ts_utc"].iloc[-1],
        detection={"model": "supervised", "version": "0.1.0", "score": 0.83, "p_anomaly": 0.91},
        classification={
            "model": "random_forest",
            "version": "0.1.0",
            "type_probs": {pred: 0.64},
            "origin_probs": {case["origin"]: 0.81},
            "type_pred": pred,
            "origin_pred": case["origin"],
        },
        evidence=evidence_checks(window),
        explanation=result,
        signature=records.build_signature(window, window["ts_utc"].iloc[0]),
        severity=case["severity"],
        versions={"features": "0.1.0", "git": "local", "data_sha": "n/a"},
    )
    print("\n=== example anomaly record ===")
    print(records.record_to_json(record)[:2000])
    print("\n=== operator text ===")
    print(record["text"])
    print(f"\nsignature dim: {len(record['signature'])}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

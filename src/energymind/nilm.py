"""Step 7 - NILM disaggregation and its residual feature (blueprint Section 6).

The four sub-meters already exist in the dataset, so the NILM model is not needed
to *create* labels. It exists for two reasons:

* **deployment realism** - most meters have no sub-metering, so a model that maps
  the aggregate to the four categories is what a real node would run; and
* **a useful anomaly feature** - the residual ``measured sub-meter - NILM
  estimate`` exposes a biased sub-meter (an injection) as a disagreement between
  the sub-meter and what the aggregate implies.

Everything is fitted on the **clean training window and training households**
only. Predictions carry ``provenance = inferred`` and are **never** used as labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from energymind import io
from energymind.validate import SUB_COLS

HOUSEHOLD_COL = io.HOUSEHOLD_COL
NILM_TARGETS = list(SUB_COLS)
#: Aggregate context -> sub-meter categories (all columns from engineer_features).
NILM_FEATURES = [
    "active_power_kw",
    "reactive_power_kvar",
    "hour_local",
    "dow",
    "temperature_c",
    "p_roll_mean_2h",
    "occupancy_probability",
]
PROVENANCE = "inferred"


@dataclass
class NilmModel:
    """One gradient-boosted regressor per sub-meter category."""

    models: dict[str, HistGradientBoostingRegressor]
    features: list[str]
    targets: list[str] = field(default_factory=lambda: list(NILM_TARGETS))
    provenance: str = PROVENANCE


def _holdout_households(settings) -> set[str]:
    from energymind import splits

    path = io.resolve(settings.data.paths.splits_file)
    if path.exists():
        import json

        return set(json.loads(path.read_text(encoding="utf-8"))["holdout_households"])
    return set(splits.build_splits(settings)["holdout_households"])


def _design(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return frame[features].astype(float)


def fit_nilm(
    frame: pd.DataFrame,
    settings,
    *,
    features: list[str] | None = None,
    max_iter: int = 200,
    random_state: int = 0,
) -> NilmModel:
    """Fit the NILM on the clean training window and training households.

    ``frame`` must already carry the engineered columns (``engineer_features``):
    ``hour_local``, ``dow``, ``p_roll_mean_2h`` and the sub-meter targets.
    """
    features = features or NILM_FEATURES
    train_end = pd.Timestamp(settings.data.splits.temporal.train.end)
    holdout = _holdout_households(settings)
    train = frame[(frame["ts_local"] < train_end) & (~frame[HOUSEHOLD_COL].isin(holdout))]

    models: dict[str, HistGradientBoostingRegressor] = {}
    for target in NILM_TARGETS:
        rows = train[features + [target]].dropna(subset=[target])
        model = HistGradientBoostingRegressor(
            max_iter=max_iter, random_state=random_state, early_stopping=False
        )
        model.fit(_design(rows, features), rows[target].to_numpy())
        models[target] = model
    return NilmModel(models=models, features=features)


def predict_nilm(model: NilmModel, frame: pd.DataFrame) -> pd.DataFrame:
    """Predict the four sub-meter categories (columns ``nilm_sub1..4``)."""
    design = _design(frame, model.features)
    return pd.DataFrame(
        {f"nilm_{target}": model.models[target].predict(design) for target in model.targets},
        index=frame.index,
    )


def evaluate_nilm(model: NilmModel, frame: pd.DataFrame, holdout: set[str]) -> dict:
    """Mean absolute error per category and in total, on the hold-out households."""
    test = frame[frame[HOUSEHOLD_COL].isin(holdout)]
    pred = predict_nilm(model, test)
    mae: dict[str, float] = {}
    for target in model.targets:
        mask = test[target].notna()
        mae[target] = float(
            np.mean(np.abs(test.loc[mask, target] - pred.loc[mask, f"nilm_{target}"]))
        )
    total_true = test[model.targets].sum(axis=1)
    total_pred = pred[[f"nilm_{t}" for t in model.targets]].sum(axis=1)
    mask = total_true.notna() & total_pred.notna()
    mae_total = float(np.mean(np.abs(total_true[mask] - total_pred[mask])))
    return {"mae": mae, "mae_total": mae_total, "n_test": int(len(test))}


def fit_and_evaluate(frame: pd.DataFrame, settings) -> tuple[NilmModel, dict]:
    """Fit on the clean training split and evaluate on the hold-out households."""
    model = fit_nilm(frame, settings)
    report = evaluate_nilm(model, frame, _holdout_households(settings))
    return model, report


def add_nilm(frame: pd.DataFrame, model: NilmModel) -> pd.DataFrame:
    """Attach the NILM estimates and residuals (``provenance = inferred``)."""
    out = pd.concat([frame, predict_nilm(model, frame)], axis=1)
    measured_total = out[model.targets].sum(axis=1)
    predicted_total = out[[f"nilm_{t}" for t in model.targets]].sum(axis=1)
    for target in model.targets:
        out[f"nilm_resid_{target}"] = out[target] - out[f"nilm_{target}"]
    out["nilm_resid_total"] = measured_total - predicted_total
    return out

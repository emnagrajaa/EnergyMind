"""M8 - learned confidence ``f`` for a decision (blueprint Section 12).

The confidence is a *calibrated probability that the final decision is correct*:

    f = sigmoid( b0 + bA*logit(A) + bL*L + bH*logit(H) + bG*G + bQ*Q + bS*S )

with the six per-case inputs:

* **A** - local accuracy: isotonic-calibrated probability that the model is right
  at this confidence (fitted on validation);
* **L** - convergence: 1 if converged in round 1, ``1-(r-1)/r_max`` if converged at
  round ``r``, 0 if not converged;
* **H** - similarity-weighted outcome of the top-k *verified* historical cases,
  shrunk toward a prior;
* **G** - share of the Validator's physical/context checks that support the
  proposed origin;
* **Q** - data completeness of the window;
* **S** - "seen before" indicator (the agents were skipped).

The beta weights are learned from validation data, never hand-set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

FEATURE_NAMES = ["logit_A", "L", "logit_H", "G", "Q", "S"]
EPS = 1e-3
#: Shrinkage prior and weight for the historical term H.
HISTORY_PRIOR = 0.5
HISTORY_SHRINK = 2.0


# --------------------------------------------------------------------------- #
# Per-case signals (raw, before fitting)
# --------------------------------------------------------------------------- #


@dataclass
class Signal:
    """Raw per-case inputs feeding the confidence features."""

    model_conf: float  # model probability of its own prediction (for A)
    converged: bool
    rounds_used: int
    r_max: int = 3
    proposed_origin: str | None = None
    histories: list = field(default_factory=list)  # list of (similarity, origin) or Match-like
    checks: list = field(default_factory=list)  # evidence checks (dicts with 'check'/'status')
    completeness: float = 1.0
    seen_before: bool = False


def _clip_logit(p: float) -> float:
    p = min(max(float(p), EPS), 1.0 - EPS)
    return float(np.log(p / (1.0 - p)))


# --------------------------------------------------------------------------- #
# Feature terms
# --------------------------------------------------------------------------- #


def convergence_L(converged: bool, rounds_used: int, r_max: int) -> float:
    """1 if converged in round 1, down to 0 if not converged."""
    if not converged:
        return 0.0
    r = max(1, int(rounds_used))
    return max(0.0, 1.0 - (r - 1) / max(1, r_max))


def _origin_of(item) -> str | None:
    if isinstance(item, tuple) and len(item) >= 2:
        return item[1]
    return getattr(getattr(item, "case", item), "origin_pred", None) or getattr(
        item, "origin", None
    )


def _similarity_of(item) -> float | None:
    if isinstance(item, tuple) and item:
        return float(item[0])
    return getattr(item, "similarity", None)


def history_H(
    histories: list,
    proposed_origin: str | None,
    prior: float = HISTORY_PRIOR,
    m: float = HISTORY_SHRINK,
) -> float:
    """Similarity-weighted outcome of verified historical cases, shrunk to a prior."""
    num = 0.0
    den = 0.0
    for item in histories or []:
        sim = _similarity_of(item)
        if sim is None:
            sim = 0.5
        outcome = 1.0 if _origin_of(item) == proposed_origin else 0.0
        num += float(sim) * outcome
        den += float(sim)
    return (num + m * prior) / (den + m)


#: Which physical/context checks support each origin (weight = validation precision, kept at 1).
_SUPPORT = {
    "data_quality": {
        "current_vs_power_identity": "broken",
        "baseline_deviation": "anomalous",
        "sensor_zero_reading": True,
        "nilm_residual": "mismatch",
    },
    "physical_event": {"neighborhood_share": "coherent"},
    "adversarial": {"neighborhood_share": "isolated"},
}


def consistency_G(checks: list, proposed_origin: str | None) -> float:
    """Share of visible checks that support the proposed origin (0..1)."""
    if not checks:
        return 0.5
    rules = _SUPPORT.get(str(proposed_origin), {})
    supported = 0
    considered = 0
    for check in checks:
        name = str(check.get("check", ""))
        status = check.get("status")
        considered += 1
        if name in rules:
            expected = rules[name]
            if (
                expected is True
                or status == expected
                or (expected == "mismatch" and status in ("mismatch", "broken"))
            ):
                supported += 1
        # an origin-specific check that is present and passes counts as support
        elif (
            proposed_origin == "physical_event"
            and name == "neighborhood_share"
            and status == "coherent"
        ):
            supported += 1
    return supported / considered if considered else 0.5


def feature_row(signal: Signal, a_from_conf: float) -> dict:
    """One feature row from a signal and the calibrated A value."""
    return {
        "logit_A": _clip_logit(a_from_conf),
        "L": convergence_L(signal.converged, signal.rounds_used, signal.r_max),
        "logit_H": _clip_logit(history_H(signal.histories, signal.proposed_origin)),
        "G": consistency_G(signal.checks, signal.proposed_origin),
        "Q": float(min(max(signal.completeness, 0.0), 1.0)),
        "S": 1.0 if signal.seen_before else 0.0,
    }


# --------------------------------------------------------------------------- #
# Local-accuracy calibration (A)
# --------------------------------------------------------------------------- #


class AccuracyCalibrator:
    """Isotonic map from model confidence to the probability the model is correct."""

    def __init__(self, prior: float = 0.6) -> None:
        self.prior = prior
        self._iso: IsotonicRegression | None = None

    def fit(self, model_conf: np.ndarray, correct: np.ndarray) -> AccuracyCalibrator:
        conf = np.asarray(model_conf, dtype=float)
        y = np.asarray(correct, dtype=float)
        if len(np.unique(y)) >= 2:
            self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(conf, y)
        return self

    def transform(self, model_conf: float) -> float:
        if self._iso is None:
            return self.prior
        return float(self._iso.predict([float(model_conf)])[0])


# --------------------------------------------------------------------------- #
# Confidence model
# --------------------------------------------------------------------------- #


class ConfidenceModel:
    """Logistic regression over the six confidence features."""

    def __init__(self, a_prior: float = 0.6) -> None:
        self.a_prior = a_prior
        self.calibrator = AccuracyCalibrator(prior=a_prior)
        self.model: LogisticRegression | None = None

    def fit(self, signals: list[Signal], y_correct: np.ndarray) -> ConfidenceModel:
        y = np.asarray(y_correct, dtype=float)
        self.calibrator = AccuracyCalibrator(prior=self.a_prior).fit(
            [s.model_conf for s in signals], y
        )
        x = build_matrix(signals, self.calibrator)
        self.model = LogisticRegression(max_iter=1000).fit(x, y)
        return self

    def predict(self, signals: list[Signal]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("ConfidenceModel.fit must be called first")
        x = build_matrix(signals, self.calibrator)
        return self.model.predict_proba(x)[:, 1]

    @property
    def coefficients(self) -> dict[str, float]:
        if self.model is None:
            return {}
        coefs = [self.model.intercept_[0], *self.model.coef_[0].tolist()]
        return dict(zip(["b0", *FEATURE_NAMES], coefs, strict=False))


def build_matrix(signals: list[Signal], calibrator: AccuracyCalibrator) -> np.ndarray:
    rows = [feature_row(s, calibrator.transform(s.model_conf)) for s in signals]
    return pd.DataFrame(rows, columns=FEATURE_NAMES).to_numpy(dtype=float)


# --------------------------------------------------------------------------- #
# Calibration report
# --------------------------------------------------------------------------- #


def reliability_curve(y_true, probabilities, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    centers, accs = [], []
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            centers.append(p[mask].mean())
            accs.append(y[mask].mean())
    return np.array(centers), np.array(accs)


def expected_calibration_error(y_true, probabilities, n_bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(ece)

"""M8 - learned confidence ``f`` for a decision (blueprint Section 12).

The confidence is a *calibrated probability that the final decision is correct*:

    f = sigmoid( b0 + bA*logit(A) + bL*L_eff + bH*logit(H) + bG*G + bQ*Q + bD*D )

with the six per-case inputs:

* **A** - local accuracy: isotonic-calibrated probability of the **proposed origin**
  (fitted on a split that is not used to fit beta);
* **L_eff** - ``(1 - D) * L`` where ``L`` = convergence: 1 if converged in round 1,
  ``1-(r-1)/r_max`` if converged at round ``r``, 0 if not converged. On the reuse path
  ``L_eff`` is 0 because deliberation did not happen;
* **H** - similarity-weighted share of the retrieved, causal, verified precedents whose
  ``origin_pred == proposal_origin``, shrunk toward the validation base rate ``p0``:
  ``(sum(s*o) + m*p0) / (sum(s) + m)`` with ``m = 2``; with no precedents ``H = p0``;
* **G** - share of the Validator's physical/context checks that support the proposed
  origin;
* **Q** - data completeness of the window;
* **D** - deliberation skipped: 1 on the "seen before" reuse path, else 0.

The beta weights are learned with L2-regularised logistic regression, never hand-set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

FEATURE_NAMES = ["logit_A", "L_eff", "logit_H", "G", "Q", "D"]
EPS = 1e-3
#: Shrinkage weight and default prior for the historical term H.
HISTORY_PRIOR = 0.5
HISTORY_SHRINK = 2.0


# --------------------------------------------------------------------------- #
# Per-case signals (raw, before fitting)
# --------------------------------------------------------------------------- #


@dataclass
class Signal:
    """Raw per-case inputs feeding the confidence features."""

    model_conf: float  # model probability of its own (max) prediction
    converged: bool
    rounds_used: int
    r_max: int = 3
    proposed_origin: str | None = None
    histories: list = field(default_factory=list)  # list of (similarity, origin) or Match-like
    checks: list = field(default_factory=list)  # evidence checks (dicts with 'check'/'status')
    completeness: float = 1.0
    #: True on the "seen before" reuse path (deliberation skipped); feature ``D``.
    deliberation_skipped: bool = False
    #: Classifier probability of the *proposed origin* (for A). ``None`` -> fall back to
    #: ``model_conf``.
    proposal_prob: float | None = None
    #: Prior ``p0`` for H (the validation base rate); set by the pipeline.
    base_rate: float = HISTORY_PRIOR


def _signal_prob(signal: Signal) -> float:
    """The classifier probability used for A: the proposed origin's, else the max."""
    return signal.model_conf if signal.proposal_prob is None else signal.proposal_prob


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


def feature_row(signal: Signal, a_from_conf: float, *, base_rate: float | None = None) -> dict:
    """One feature row from a signal and the calibrated A value.

    ``L_eff = (1 - D) * L`` so L is 0 when deliberation was skipped; ``H`` shrinks
    toward ``base_rate`` (defaults to the signal's prior).
    """
    prior = signal.base_rate if base_rate is None else base_rate
    skipped = 1.0 if signal.deliberation_skipped else 0.0
    return {
        "logit_A": _clip_logit(a_from_conf),
        "L_eff": (1.0 - skipped)
        * convergence_L(signal.converged, signal.rounds_used, signal.r_max),
        "logit_H": _clip_logit(history_H(signal.histories, signal.proposed_origin, prior=prior)),
        "G": consistency_G(signal.checks, signal.proposed_origin),
        "Q": float(min(max(signal.completeness, 0.0), 1.0)),
        "D": skipped,
    }


# --------------------------------------------------------------------------- #
# Local-accuracy calibration (A)
# --------------------------------------------------------------------------- #


class AccuracyCalibrator:
    """Isotonic map from the classifier probability of the proposed origin to P(correct)."""

    def __init__(self, prior: float = 0.6) -> None:
        self.prior = prior
        self._iso: IsotonicRegression | None = None

    def fit(self, proposal_prob: np.ndarray, correct: np.ndarray) -> AccuracyCalibrator:
        conf = np.asarray(proposal_prob, dtype=float)
        y = np.asarray(correct, dtype=float)
        if len(np.unique(y)) >= 2:
            self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(conf, y)
        return self

    def transform(self, proposal_prob: float) -> float:
        if self._iso is None:
            return self.prior
        return float(self._iso.predict([float(proposal_prob)])[0])


# --------------------------------------------------------------------------- #
# Confidence model
# --------------------------------------------------------------------------- #


class ConfidenceModel:
    """L2-regularised logistic regression over the six confidence features."""

    def __init__(self, a_prior: float = 0.6, C: float = 1.0) -> None:
        self.a_prior = a_prior
        self.C = C
        self.calibrator = AccuracyCalibrator(prior=a_prior)
        self.model: LogisticRegression | None = None
        self._constant: float | None = None

    def fit(
        self,
        signals: list[Signal],
        y_correct: np.ndarray,
        *,
        a_signals: list[Signal] | None = None,
        a_y: np.ndarray | None = None,
    ) -> ConfidenceModel:
        """Fit A (isotonic) and beta (logistic).

        ``a_signals`` / ``a_y`` (default: the fitting set) calibrate **A**; passing a
        separate split keeps the isotonic map off the data used to fit beta.
        """
        y = np.asarray(y_correct, dtype=float)
        cal_signals = signals if a_signals is None else a_signals
        cal_y = y if a_y is None else np.asarray(a_y, dtype=float)
        self.calibrator = AccuracyCalibrator(prior=self.a_prior).fit(
            [_signal_prob(s) for s in cal_signals], cal_y
        )
        x = build_matrix(signals, self.calibrator)
        if len(np.unique(y)) < 2:
            # single-class fitting split: a constant model (f = the base rate)
            self.model = None
            self._constant = float(y.mean()) if len(y) else float(self.a_prior)
            return self
        self._constant = None
        self.model = LogisticRegression(C=self.C, max_iter=1000).fit(x, y)
        return self

    def predict(self, signals: list[Signal]) -> np.ndarray:
        if self._constant is not None:
            return np.full(len(signals), self._constant, dtype=float)
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

    def coefficient_cis(
        self,
        signals: list[Signal],
        y_correct: np.ndarray,
        *,
        n_boot: int = 1000,
        seed: int = 0,
        alpha: float = 0.05,
    ) -> dict[str, list[float]]:
        """Bootstrap ``1-alpha`` CIs for beta (resampling the fitting set)."""
        if self.model is None:
            return {}
        x = build_matrix(signals, self.calibrator)
        y = np.asarray(y_correct, dtype=float)
        if len(y) == 0:
            return {}
        names = ["b0", *FEATURE_NAMES]
        rng = np.random.default_rng(seed)
        draws = np.full((n_boot, len(names)), np.nan, dtype=float)
        for b in range(n_boot):
            idx = rng.integers(0, len(y), len(y))
            if len(np.unique(y[idx])) < 2:
                continue
            fit = LogisticRegression(C=self.C, max_iter=1000).fit(x[idx], y[idx])
            draws[b] = [fit.intercept_[0], *fit.coef_[0]]
        lo = np.nanpercentile(draws, 100.0 * alpha / 2.0, axis=0)
        hi = np.nanpercentile(draws, 100.0 * (1.0 - alpha / 2.0), axis=0)
        return {
            name: [round(float(lo[i]), 4), round(float(hi[i]), 4)] for i, name in enumerate(names)
        }


def build_matrix(signals: list[Signal], calibrator: AccuracyCalibrator) -> np.ndarray:
    rows = [
        feature_row(s, calibrator.transform(_signal_prob(s)), base_rate=s.base_rate)
        for s in signals
    ]
    return pd.DataFrame(rows, columns=FEATURE_NAMES).to_numpy(dtype=float)


def variance_inflation_factors(signals: list[Signal], calibrator: AccuracyCalibrator) -> dict:
    """VIF per feature (``> 5`` signals problematic collinearity between features)."""
    if not signals:
        return {}
    x = build_matrix(signals, calibrator)
    out: dict[str, float] = {}
    for j, name in enumerate(FEATURE_NAMES):
        others = np.delete(x, j, axis=1)
        design = np.column_stack([np.ones(len(others)), others])
        target = x[:, j]
        coef, *_ = np.linalg.lstsq(design, target, rcond=None)
        residual = target - design @ coef
        ss_res = float((residual**2).sum())
        ss_tot = float(((target - target.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        out[name] = round(1.0 / (1.0 - r2), 4) if r2 < 1.0 else float("inf")
    return out


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


# --------------------------------------------------------------------------- #
# Risk-coverage curve (is f a better ranking signal than the raw model prob?)
# --------------------------------------------------------------------------- #


def risk_coverage(cases: list[dict], score_key: str = "f") -> dict:
    """Coverage vs error among the most-confident cases, for a score.

    Returns ``{"coverage": [...], "risk": [...]}`` where ``risk`` is the error
    among the top-``coverage`` fraction of cases by ``score_key``.
    """
    if not cases:
        return {"coverage": [], "risk": []}
    order = sorted(cases, key=lambda c: c.get(score_key, 0.0), reverse=True)
    correct = np.array([bool(c["correct"]) for c in order], dtype=float)
    coverage, risk = [], []
    for k in range(1, len(correct) + 1):
        coverage.append(k / len(correct))
        risk.append(1.0 - correct[:k].mean())
    return {"coverage": coverage, "risk": risk}


def risk_coverage_summary(cases: list[dict]) -> dict:
    """Compare the area under the risk-coverage curve for ``f`` vs ``model_conf``."""

    def area(points: dict) -> float:
        if not points["coverage"]:
            return 0.0
        return float(np.trapezoid(points["risk"], points["coverage"]))

    return {
        "auc_risk_f": round(area(risk_coverage(cases, "f")), 4),
        "auc_risk_model_conf": round(area(risk_coverage(cases, "model_conf")), 4),
        "f_curve": risk_coverage(cases, "f"),
        "model_conf_curve": risk_coverage(cases, "model_conf"),
    }

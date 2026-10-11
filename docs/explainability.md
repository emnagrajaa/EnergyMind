# Explainability and anomaly records (Step 9)

> Implementation: `src/energymind/explain.py`, `src/energymind/records.py`.
> Reproduce: `python -W ignore -m energymind.explain --cases 6`.

This document specifies how EnergyMind explains a flagged case, how the
explanation is validated, and how it is written into the auditable anomaly record.

---

## 1. What is produced per case

| Output | Producer | Purpose |
| --- | --- | --- |
| **SHAP** attributions (signed top-k) | `explain.tree_shap_top` | explains the tree model (which features pushed the prediction) |
| **LIME** attributions (×5 seeds) | `explain.lime_top_features` | independent cross-check of SHAP |
| **Explanation reliability** | `explain.mean_jaccard` | quantitative explanation quality |
| **Reconstruction attribution** | `explain.reconstruction_attribution` | for the LSTM-AE: which channel / time drove the error |
| **Rule-based evidence checks** | `explain.evidence_checks` | observed-vs-expected facts with units |
| **Measured facts** | `explain.window_facts` | the numbers behind the operator text |
| **Plain-language narrative** | `records.narrative` | the operator sentence(s) |
| **64-dim signature** | `records.build_signature` | similarity / "seen before?" key |

The record keeps **prediction ≠ cause**: model outputs explain the model, the
evidence explains the data, and the cause is stated as a *hypothesis*.

## 2. Methods

### 2.1 TreeSHAP (tree models)
`shap.TreeExplainer` gives exact, signed per-feature attributions for the Random
Forest / LightGBM classifier (and the Isolation Forest). For one case we return
the top-k features by absolute contribution for the predicted class. Because it
is exact, the *same model and case always give the same attributions* — that is
reproducibility, not a hard-coded answer.

### 2.2 LIME (tabular, ×5 seeds)
A `LimeTabularExplainer` is fit on the training design matrix; each case is
explained 5 times with different seeds. LIME is intentionally unstable, which is
what makes its agreement with SHAP meaningful. Zero-variance columns break LIME's
discretizer and NaNs break its distance step, so the LIME inputs are sanitised
(`discretize_continuous=False`, `fillna(0)`); SHAP keeps the NaNs.

### 2.3 Explanation reliability
E_rel = mean over the 5 LIME seeds of `Jaccard(top-5 SHAP, top-5 LIME)`.
Low overlap → "explanation unstable" (and, downstream, it lowers Agent 1's
weight). This is the challenge's *explanation-quality* metric made quantitative.

### 2.4 Reconstruction attribution (LSTM-AE)
For the reconstruction model we use an **exact** decomposition of the mean squared
error into the per-channel and per-time-step error, instead of DeepSHAP. It is
faithful (not approximate) and directly answers "which channel and which half-hour
drove the error".

### 2.5 Rule-based evidence checks
`evidence_checks(window)` returns observed-vs-expected entries with units:

* `baseline_deviation` — observed vs expected kW, robust z;
* `current_vs_power_identity` — max relative identity residual (broken if > 1e-2);
* `neighborhood_share` — coherence across households (coherent if ≥ 40%);
* `nilm_residual_total` and `sub_meter_share_cooling`.

`window_facts` adds the surrounding context: the household's own
**neighbouring-interval** median (2 h before/after), the **neighbourhood median**,
the zero-reading fraction, the flat-run length, and the voltage deviation.

### 2.6 Plain-language narrative
`records.narrative(facts, origin, type_pred)` turns the facts into case-specific
sentences with a conclusion. Examples (real output, numbers from the data):

> The meter read exactly zero for 100% of the window, although this household
> normally draws **3.597 kW** at this hour. **Conclusion: sensor outage (dropout),
> not zero consumption.**

> The reading **0.165 kW** is inconsistent with neighbouring intervals
> (**6.794 kW**) and the neighbourhood median (**3.189 kW**) and the equipment
> estimates. **Conclusion: likely falsified or corrupt sample.**

> The sample breaks the electrical identity (V·I ≠ S; relative residual **1.0**).
> **Conclusion: corrupt or falsified sample.**

> Active power is constant for **49 intervals (24.5 h)**. **Conclusion: frozen /
> stuck sensor.**

> The deviation is shared by **69%** of neighbours at the same time. **Conclusion:
> genuine network event, not a single-meter fault.**

Other branches: voltage deviation from 230 V; sub-meter/NILM disagreement →
"biased sub-meter"; `origin=adversarial` → "under-reporting / theft". Every number
in the text is measured; nothing is written by hand.

## 3. The anomaly record (blueprint Section 9)

```json
{
  "anomaly_id": "uuid",
  "household_id": "HH_00072_AUS",
  "t_start": "2020-11-14T18:00:00Z", "t_end": "2020-11-15T06:00:00Z",
  "duration_h": 12.0,
  "detection":      {"model": "supervised", "version": "0.1.0", "score": 1.0, "p_anomaly": 0.9998},
  "classification": {"model": "lightgbm", "version": "0.1.0",
                     "type_probs":   {"dropout": 0.71},
                     "origin_probs": {"data_quality": 0.98},
                     "type_pred": "dropout", "origin_pred": "data_quality"},
  "evidence": [
    {"check": "baseline_deviation", "observed_kw": 0.0, "expected_kw": 3.6, "z": -6.1, "status": "anomalous"},
    {"check": "current_vs_power_identity", "rel_residual": 0.0, "status": "holds"},
    {"check": "neighborhood_share", "value": 0.08, "status": "isolated"},
    {"check": "nilm_residual_total", "value": -0.70}
  ],
  "facts": {...},
  "explanation": {"shap_top": [["p_resid_z_max", 0.42], ["duration_h", -0.31]],
                  "lime_top": ["p_resid_z_max", "duration_h"],
                  "reliability": 0.42},
  "cause_hypotheses": ["Upstream data-quality fault: identities hold but the load departs from the baseline"],
  "severity": "high",
  "limitations": ["weather is synthetic and weakly informative", "operator behaviour is simulated"],
  "signature": [ ... 64 numbers ... ],
  "versions": {"features": "0.1.0", "git": "...", "data_sha": "..."},
  "text": "The meter read exactly zero for 100% of the window, ... Conclusion: sensor outage (dropout), not zero consumption."
}
```

### 3.1 The 64-dim signature

`build_signature` = 48 shape dims (P, V, I each resampled to 16 points, per-household
standardised) + duration (normalised) + magnitude/σ_h + the 4 identity residuals +
neighbourhood share + 5 NILM residuals + hour/weekday encodings = **64**. It is a
numeric physics signature (no text), so retrieval matches on physics rather than
wording.

## 4. Reliability and the sanity check

* **Mean explanation reliability** across test cases is **≈ 0.42** (Jaccard of
  top-5 SHAP vs LIME). This is deliberately reported, not hidden: SHAP and LIME
  disagree on a fraction of cases, which the protocol uses as an
  "explanation unstable" flag.
* **Sanity check:** for a `sub_meter_bias` injection, a sub-meter feature
  (`sub{1..4}_*`) must appear in the SHAP top-3 (`explain.sanity_check`).

## 5. Reproduce

```bash
python -m pip install shap lime          # once
python -m energymind.features            # build features + NILM
python -W ignore -m energymind.explain --cases 6            # six cases, with narratives
python -W ignore -m energymind.explain --index 3            # a specific case
python -W ignore -m energymind.explain --anomaly-id <uuid>  # one exact case
pytest tests/test_explain.py tests/test_records.py -v       # unit tests
```

`shap` and `lime` are imported lazily, so the core test suite passes without them
(the SHAP-dependent test skips).

## 6. Limitations

* SHAP explains the **model**, not the physical cause; `cause_hypotheses` carries
  the competing physical explanations.
* LIME is noisy; reliability quantifies this rather than removing it.
* The coherence sentence can disagree with the true origin when many neighbours
  also deviate (a real property of the data), so it is stated as evidence.
* Reconstruction attribution is an exact MSE decomposition, not DeepSHAP — a
  faithful, cheaper equivalent for a reconstruction model.

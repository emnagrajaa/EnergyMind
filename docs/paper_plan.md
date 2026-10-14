# Paper plan — EnergyMind Phase 1 proposal

> Derives from `docs/project_plan.md` §18 and the built artifacts (`docs/data_report.md`,
> `docs/anomaly_generation.md`, `docs/benchmark.md`, `docs/explainability.md`,
> `docs/memory_audit.md`, `docs/decision_protocol.md`). Output goes to `docs/proposal/`.

**Working title.** *A Trust Layer for Smart-Meter Data: Physics-Consistent Anomaly
Injection, Two-Agent Deliberation and Learned Confidence for Human–AI Escalation,
Toward Federated Continual Learning.*

**Style.** Registered-report: problem, data findings, methods and hypotheses now;
every result marked *planned* or *preliminary*.
**Venue / length.** IEEE ISGT or SmartGridComm (6–8 pages, two columns), or a
journal (Energy & AI / IEEE Access). Figures below are IEEE double-column width.

**Two-column figure/table numbering convention:** Fig. 1 … are data/architecture;
Tab. 1 … are results.

---

## 1. Abstract (≈200 words, no results)

Problem: automated grid decisions consume smart-meter data that can be faulty,
biased or falsified. Approach: a *trust layer* that detects and classifies
anomalies, explains them, scores its own confidence, and routes to a human or acts
autonomously — validated on a physics-consistent injected benchmark. Contribution
sentence + Phase-2 pointer (FCL).

## 2. Introduction

* Tunisian context: seasonal peaks, renewables growth, smart-meter rollout, the
  trust question (who may act on this reading?).
* The gap: detection ≠ decision. Papers report detection scores on easy synthetic
  anomalies and stop.
* Research question **RQ**: *Does a learned confidence over model calibration,
  agent convergence, verified history and physics consistency route decisions
  better than the model's own probability?*
* Contributions (5): data audit; physics-consistent injection benchmark; two-agent
  protocol with citation checking; learned confidence + tiered routing with
  simulated over/under-trust; FCL design.
* Figure: **Fig. 1** the trust layer schematic (NEW — export the Mermaid from
  `README.md` / `project_plan.md` §4 to PDF).

## 3. Related work

* Energy anomaly detection (statistical, ML, deep: SARIMA, Isolation Forest, LSTM-AE)
  and the TSB-AD evaluation critique (no point-adjusted F1, VUS-PR).
* Federated / continual learning for grids (Shrestha/Nowroozi LSTM-AE + FedAvg;
  FCL survey 2502.07059; drift-aware VAE 2602.12976).
* Explainable anomaly detection in energy (Cai et al., multi-head transformer + SHAP).
* Model calibration & selective prediction (Guo 2017; Geifman & El-Yaniv 2017).
* Human–automation trust (Parasuraman & Riley 1997; Bansal 2021) and multi-agent
  debate (Du 2023).
* **Table: Tab. 1** related-work positioning (method × {detection, classification,
  explanation, confidence, routing, federated}).

## 4. Problem formulation

* Notation: household `h`, interval `t`, channels `P,Q,V,I,S,PF,E`, sub-meters.
* Labels: `type`, `origin ∈ {data_quality, adversarial, physical_event}`,
  `severity`, `variant ∈ {naive, consistent}`.
* Action set `A` (8 actions) with risk + reversibility; the routing objective.
* Confidence `f`; the tier function; the never-automate list.
* No figure; **Tab. 2** the action set with risk/reversibility.

## 5. Data (findings F1–F10)

* The Austin 2020 panel: 13 households, 30-min, 2024 rows… state shape/coverage and
  the honest synthetic framing.
* Findings F1–F10 with the pipeline consequences.
* Splits (temporal + 3-household hold-out) and the `quality_mask`.
* Figures (existing, from `docs/data_report.md`):
  * **Fig. 2** `dataset_overview.png`
  * **Fig. 3** `power_identity_scatter.png` (F1/F2)
  * **Fig. 4** `physics_residuals.png` (F1)
  * **Fig. 5** `leakage_f3.png` (F3)
  * **Fig. 6** `provided_labels.png` (F2)
  * **Fig. 7** `monthly_load_temperature.png` + `daily_profiles.png` (F6)
  * **Fig. 8** `submeter_share_boxplot.png` (F7)
  * **Fig. 9** `cross_household_correlation.png` (F10)
  * **Fig. 10** `flat_run_histogram.png` (F9) + `temperature_vs_weather.png` (F5)
  * **Fig. 11** `weekly_profile_heatmap.png` (O1) + `der_profiles.png` (O2)
  * Supplementary: `missing_record_map.png`, `electrical_parameters.png`,
    `metadata_contradictions.png`, `temporal_splits.png`, `power_distribution_log.png`,
    `engineered_features.png`
* **Tab. 3** findings F1–F10 with consequence.

## 6. Anomaly simulation

* Why inject; the identity constraint; naive vs consistent (H1).
* The 15 types, formulas, ranges, seeds, prevalence, withholding, non-overlap.
* Multi-household physical events (coherence).
* Figures (existing):
  * **Fig. 12** `injection_overview.png`
  * **Fig. 13** `injection_type_gallery.png`
  * **Fig. 14** `injection_residual_gallery.png` (naive vs consistent)
  * **Fig. 15** `injection_variant_gain_error.png` (one detailed comparison)
* **Tab. 4** the 15 types × origin × variants × ranges (from `anomaly_generation.md`).

## 7. Architecture and decision protocol

* Data treatment → domestic features/NILM → ground truth → injection → past-only
  features → benchmark → XAI → memory → agents → confidence → tiers → audit.
* The two agents (evidence separation), closed action set, citation checks,
  bounded deliberation (`r_max=3`).
* Confidence `f`; tiered routing; memory & audit (Qdrant + hash chain).
* Figures (NEW — schematics to export):
  * **Fig. 16** end-to-end architecture (from the Mermaid diagram).
  * **Fig. 17** two-agent deliberation loop (blind round 1 → peer evidence →
    convergence).
  * **Fig. 18** routing/tier diagram + never-automate.
* **Tab. 5** agent evidence boundaries (Analyst vs Validator read/don't-read).
* **Tab. 6** the six confidence terms A/L/H/G/Q/S.

## 8. Experimental design

* Detection protocol: event unit, ±1 tolerance, no point-adjusted F1, threshold at
  1 false-alarm/household-week, temporal test + household hold-out, 3 seeds.
* Classification: per-event features + detector scores, LightGBM (calibrated).
* Agent ablations (rule-based vs LLM; same vs different families; ± history).
* Decision: calibration (ECE), risk–coverage vs raw model probability, simulated
  operators.
* **Fig. 19** evaluation pipeline schematic (NEW).
* **Tab. 7** metrics list with definitions.

## 9. Results (preliminary)

* **9.1 Benchmark (detection).**
  * **Fig. 20** `bench_detection_leaderboard.png` (event F1, mean±std).
  * **Fig. 21** `bench_recall_breakdown.png` (naive vs consistent; by length).
  * **Fig. 22** `bench_pr_curves.png` + `bench_score_distributions.png`.
  * **Fig. 23** `bench_detector_tradeoff.png` (recall vs FA budget).
  * **Tab. 8** the leaderboard (all detectors × metrics) — from `runs.jsonl`.
* **9.2 Classification.**
  * **Fig. 24** `bench_classifier.png` (macro-F1, origin confusion, reliability).
  * **Tab. 9** LightGBM vs Random Forest (macro-F1, origin acc, ECE).
* **9.3 Explainability.**
  * **Fig. 25** explanation reliability distribution (SHAP vs LIME Jaccard) — NEW
    plot from `explain.explain_case`.
  * **Tab. 10** example anomaly record with evidence + narrative.
* **9.4 Decision quality (H3).**
  * **Fig. 26** risk–coverage curve: `f` vs raw model probability — NEW plot from
    `confidence.risk_coverage_summary`.
  * **Fig. 27** reliability diagram + ECE — NEW plot from
    `confidence.reliability_curve`.
  * **Fig. 28** tier distribution + human-review summary (auto vs escalated to the
    human) — NEW (bar chart of `human_review` / `tier_counts`).
  * **Tab. 11** learned confidence coefficients β.
  * **Tab. 12** human-review summary (escalated, approved, rejected, pending).
* **9.5 Agent ablation.**
  * **Tab. 13** rule-based vs LLM rows (`results/m7_ablation.json`): converged %,
    rounds, fallback rate, action agreement.

> All 9.x results are marked **preliminary** until seeds and the test set are frozen.

## 10. Discussion

* H1 (consistent ≥ naive difficulty), H2 (agent agreement informative), H3 (f beats
  raw probability — risk–coverage), H4 (FCL, Phase 2).
* What the learned β say (e.g. history and convergence dominate; `G` null here).
* Where it fails (consistent short events; rule-based non-convergence → tier 3).

## 11. Governance and human oversight

* Three tiers, never-automate list, mandatory override reasons.
* Auditability: hash-chained log, verified-only memory, model/prompt hashes.
* **Fig. 29** operator review card mock-up (NEW — from `docs/decision_protocol.md`).

## 12. Limitations

* Synthetic data, weak weather–load coupling, 13 households; simulated operators;
  Austin calendar vs Tunisian deployment; preliminary results.

## 13. Conclusion and future work

* Phase 2: FCL (Avalanche + FedAvg), user study, business scope
  (Postgres/compute/operator-time cost).

## References

The blueprint's verified references (Shrestha/Nowroozi; FCL survey; Kim et al.;
Wu & Keogh; Guo; Geifman & El-Yaniv; SHAP; LIME; Parasuraman & Riley; Bansal; FedAvg;
EWC; Du) + fill the two blocked ScienceDirect entries.

## Appendix

* Data schema (104 columns), hyper-parameters, seeds, the 15 injection formulas,
  reproducibility commands.

---

## Figure inventory

**Existing (33 PNGs, `results/figures/`):** 19 EDA + 8 injection + 6 benchmark.
**Still needed (to create):**

| # | Figure | Source / how |
| --- | --- | --- |
| 1 | Architecture schematic | export the Mermaid (`project_plan.md` §4) |
| 16–18 | Deliberation loop, routing diagram, review card | schematics (draw/export) |
| 19 | Evaluation pipeline | schematic |
| 25 | Explanation reliability histogram | plot from `explain.explain_case` over N cases |
| 26 | Risk–coverage curve (f vs model prob) | plot `confidence.risk_coverage_summary` |
| 27 | Reliability diagram + ECE | plot `confidence.reliability_curve` |
| 28 | Tier distribution + human review summary | bar chart from `pipeline.run` output |
| — | M8 figures are printed by `notebooks/05_decision_demo.ipynb` today; promote to saved PNGs |

Regenerate all existing figures:

```bash
python -m energymind.eda            # data_report figures
python -m energymind.inject_plots   # injection gallery
python -m energymind.benchmark_plots --prevalence 0.05 --seed 1
```

## Table inventory

Tab. 1 related-work matrix · Tab. 2 action set · Tab. 3 findings F1–F10 ·
Tab. 4 injection types · Tab. 5 agent evidence boundaries · Tab. 6 confidence terms ·
Tab. 7 metrics · Tab. 8 benchmark leaderboard · Tab. 9 classifier ·
Tab. 10 example anomaly record · Tab. 11 confidence β · Tab. 12 operator trust ·
Tab. 13 rule-based vs LLM ablation.

## Ownership (2–3 people)

* **Data + injection + paper data sections** — §5, §6.
* **Modelling + XAI** — §8, §9.1–9.3 (benchmark, classification, explanation).
* **Decision layer** — §7, §9.4–9.5, §11 (agents, confidence, routing, HIL).
* Proposal assembly and figures/LaTeX — shared.

# EnergyMind Tunisia — Phase 1 Execution Plan

> Working execution plan derived from the Phase 1 Research Plan & Proposal Blueprint and `README.md`.
> **Challenge:** TSYP14 · **Phase 1 deadline:** 15 Oct 2026 · **Today:** 10 Oct 2026.
> **Team size:** 2–3 people · **Agents:** LLM-backed (primary), rule-based (fallback + ablation) · **Dataset:** supplied by the team in `data/raw/`.

This file turns the blueprint (§15 "What to code, step by step") into an ordered, dependency-aware checklist with owners and definitions of done. It does **not** replace the blueprint — section references (§n) point back to it.

---

## 0. Locked decisions

| Decision | Choice |
| --- | --- |
| Raw dataset | The team places `Austin_TX_US_2020.csv` (193 MB) in `data/raw/`. Nothing runs until it is present. |
| Two-agent backend | **LLM-backed is required** for Phase 1: Agent 1 = Analyst (model family A), Agent 2 = Validator (model family B). Rule-based versions stay in the code as timeout/invalid-output fallback **and** as the paper's ablation baseline. |
| Team | 2–3 people → three workstreams, ownership in §8. |
| Task runner | Windows host has **no `make`**. Provide a PowerShell task runner (`tasks.ps1`) mirroring the `Makefile` targets. |
| PDF | `docs/…(Architects of the Energy Noosphere…).pdf` is source material only; the blueprint markdown is treated as authoritative. |

---

## 1. What we are building

A **trust layer between raw smart-meter data and any automated action**: physics-consistent anomaly injection → a benchmark of detectors/classifiers → SHAP+LIME explanations → a "seen before?" vector memory → a two-agent deliberation → a learned, calibrated confidence `f` → tiered human-in-the-loop routing → an append-only audit log. Phase 2 (federated continual learning) is **design only**.

**Important reframing for this plan:** the blueprint's Phase-1 bar is Steps 1–6 (data → injection → features) plus a *minimal* prototype of Steps 7–12. Because LLM agents are now required, Steps 9–12 get more weight than "minimal", so we start the decision-layer track early on **mocked records** (§15 ownership note).

---

## 2. Environment & blockers

| # | Status | Note |
| --- | --- | --- |
| Python 3.11.9 | ✅ present | Meets the 3.11+ requirement. |
| Docker 29.8.0 | ✅ present | Needed for `pgvector/pgvector:pg16`. |
| `make` | ❌ missing on Windows | Use `tasks.ps1` (and keep `Makefile` for CI/Linux). |
| Dataset | ⛔ missing | Blockers Steps 1–8. **Team action.** |
| PDF | ⚠️ unreadable by the AI agent | Non-blocking; markdown plan is authoritative. |

**Immediate team action:** copy `Austin_TX_US_2020.csv` into `data/raw/`, then we record its SHA-256 in `data/raw/MANIFEST.csv`.

---

## 3. Deliverables checklist (from blueprint §19)

**Proposal**
- [ ] Scope (13-household node), objectives, RQ, H1–H4
- [ ] Data findings F1–F10 with plots + honest framing of the synthetic source
- [ ] Anomaly-simulation plan (§7): formulas, both variants, seeds
- [ ] Design per goal: G1 benchmark, G2 SHAP/LIME + confidence + log, G3 agents + tiers + HIL, G4 FCL
- [ ] Audit-log decision (Postgres + pgvector) with rationale
- [ ] Results marked planned/preliminary; business-scope paragraph

**Repository**
- [ ] Steps 1–6 code, tests, notebooks 01–03
- [ ] `docs/data_report.md`, `docs/anomaly_generation.md`, `DATA_LICENSE.md`
- [ ] Benchmark notebook with ≥ robust z, rules, IF on one seed (preliminary)
- [ ] `docker-compose.yml` + `sql/schema.sql`; decision demo on ~20 test cases (Steps 9–12)
- [ ] CI green

---

## 4. Milestones (dependency-ordered)

Legend: **Owner** = track in §8. **Dep** = prerequisites. **Done when** = acceptance test.

### M0 — Foundations *(Data track)*
| Task | Files | Done when |
| --- | --- | --- |
| Packaging | `pyproject.toml`, `requirements.lock` | `pip install -e .` succeeds |
| Config schemas | `configs/{data,injection,models,policy,agents}.yaml`, `src/energymind/config.py` (pydantic) | All YAML loads + validates |
| Task runner | `Makefile`, `tasks.ps1` | `data/inject/bench/demo/test` runnable on Windows + Linux |
| CI | `.github/workflows/ci.yml` | ruff + pytest on `tests/fixtures` pass, no data/Postgres needed |
| Legal/env | `LICENSE`, `DATA_LICENSE.md`, `.env.example` | Present |

### M1 — Data layer *(Data track · blueprint Steps 1–4)*
- **Step 1** `io.py` — checksum CSV → `raw.parquet`; drop the **64 derived columns**; parse time as UTC−6 (no DST); keep 698 provided labels separately. Done when `measured.parquet` (37 cols) has no rolling/lag/zscore/flag column and `provided_labels.parquet` has 698 rows.
- **Step 2** `clean.py` + `validate.py` — 30-min grid (17,520 rows/HH, 8 missing records), physics residuals `r_S, r_VI, r_E, r_PF, r_sub`, bounds flags, flat-run length, provided-label masking, single-slot imputation, **quality_mask**. Done when all `|r_*| < 1e-2` on clean rows (measured max ≈ 0.0072).
- **Step 3** `notebooks/01_inspect.ipynb`, `02_eda.ipynb` → `docs/data_report.md` (plots ↔ F1–F10).
- **Step 4** `splits.py` — temporal (train Jan–Aug / val Sep–Oct / test Nov–Dec), 3 household hold-outs (one solar, one EV, one commercial_small), `experience_id = month`. Done when no forbidden overlap.

### M2 — Anomaly injection *(Injection track · Step 5)*
- `inject/base.py` (`Injection`, `recompute_dependents`), then one function per type in `data_quality.py` (9), `adversarial.py` (3), `physical.py` (3). **15 types × 2 variants.**
- `REGISTRY` + `inject_all(clean, cfg, seed)`: inject only where `quality_mask = 1`, one anomaly per (HH, interval), seeds per split, prevalence 1/5/10%, replay+mimicry withheld from train/val.
- `notebooks/03_injection_gallery.ipynb`, `tests/test_injection.py`.
- **Done when:** same seed → identical SHA-256; consistent variants keep `|r_*| < 1e-2`; naive variants break them (blueprint verified a 12 h bias on HH_00072: consistent residual 0, naive `r_S` 0.16, `r_E` 0.19).

### M3 — Features & NILM *(Injection → Data · Steps 6–7)*
- `features.py` — **past-only** rolling windows (`shift(1)` first) over P/I/V (2 h/24 h/7 d), residual vs hour-of-week median (fit on train), physics residuals, **neighbourhood share**, sub-meter shares, flat-run length.
- `nilm.py` — multi-output LightGBM/HistGB aggregate → sub1–sub4, `provenance=inferred`; report MAE on hold-out households; residual used as a feature.V
- **Done when:** `tests/test_features.py` proves a change at `t` alters no past-only feature at `t`.

### M4 — Benchmark *(Modelling track · Step 8)*
- `detect/base.py` (`fit`, `score`) + `robust_z`, `rules`, `sarima`, `iforest`, `pyod_models` (ECOD/COPOD/HBOS), `matrix_profile` (discords **+ replay motifs/AB-join**), `lstm_ae`, `usad`, `tranad`, `moment_zs` (optional), `ensemble`.
- `classify.py` (RF/LightGBM type+origin, calibrated), `evaluate.py` (event P/R/F1 at fixed false-alarm budget, PR-AUC, VUS-PR, false alarms/HH-week, latency, per type×severity×variant recall, macro-F1, origin confusion, ECE, `runs.jsonl` writer).
- `notebooks/04_benchmark.ipynb` generates the results table from `runs.jsonl`. **No point-adjusted F1.** 3 seeds, mean ± std; temporal test + household hold-out.
- **Done when:** ≥4 detectors scored on 1 seed, table auto-generated.

### M5 — Explanations & records *(Decision track · Step 9)*
- `explain.py` — TreeSHAP (RF/IF), Kernel/DeepSHAP (LSTM-AE), LIME×5 seeds, `reliability = mean Jaccard(top5_SHAP, top5_LIME)`, rule-based evidence checks.
- `records.py` — anomaly record (§9 schema), **64-dim signature**, template `render_text` (numbers only from evidence).

### M6 — Memory & audit log *(Decision track · Step 10)*
- `sql/schema.sql` — `anomaly_case` (vector(64) + HNSW) and `audit_event` (hash chain, `REVOKE UPDATE, DELETE`).
- `docker-compose.yml` (pgvector/pg16), `store.py` (`insert_case`, `append_event` computing `row_hash = sha256(prev_hash ‖ canonical_json(payload))`, `nearest_verified`), `retrieval.py` (`Retriever` interface + `LocalPgRetriever`, filtered k-NN).

### M7 — Two-agent loop — **LLM-backed** *(Decision track · Step 11)*
- `agents/base.py` (Proposal dataclass, Agent protocol, closed action set + risk/reversibility), `prompts/analyst_system.md`, `prompts/validator_system.md` (versioned + hashed into the log).
- `analyst.py`, `validator.py` — **rule-based fallbacks** and ablation baselines.
- `llm_agent.py` — **primary path.** Ollama (default) / OpenAI-compatible client; Pydantic `ProposalOut` JSON-schema via `format=`; `temperature 0`, fixed seed; `validate_citations()` rejects uncited evidence; on timeout/invalid output → rule-based fallback.
- `deliberation.py` — blind round 1, ≤ `R_max = 3`, agents may revise only citing new evidence, non-convergence → both proposals to tier 3.
- `configs/agents.yaml` — backend, per-agent model, timeout, seed.
- `tests/test_deliberation.py` — uncited evidence rejected; `R_max` respected.
- **Done when:** demo runs both agents end-to-end on real records, every message written as an `audit_event`, and rule-based vs LLM ablation rows exist.

**Model pairing (blueprint §11 hardware table):**
| Hardware | Agent 1 (Analyst) | Agent 2 (Validator) |
| --- | --- | --- |
| 8 GB GPU / 16 GB RAM | `qwen3.5:9b` | `granite4.2:8b` / `phi4-mini` |
| 16 GB GPU / 32 GB | `gpt-oss:20b` | `qwen3.5:9b` |
| 24 GB GPU | `qwen3.5:27b` | `gemma4:26b` |
| CPU only | `qwen3.5:4b` | `phi4-mini` (keep `R_max = 2`) |

### M8 — Confidence, routing, HIL *(Decision track · Step 12)*
- `confidence.py` — features A (local accuracy, isotonic), L (convergence), H (verified-history, shrinkage toward prior), G (weighted Validator checks), Q (completeness), S (seen-before indicator) → logistic `f` fit on validation; reliability diagram + ECE on test.
- `policy.py` — 3 tiers (τ chosen on validation to ε error target) + never-automate list.
- `review.py` — the human-in-the-loop card (approve / reject / request more information) sent to the real user for every escalated case; risk–coverage curve vs raw model probability lives in `confidence.py`.
- `pipeline.py` — end-to-end: case → seen-before? → agents → f → tier → log. `notebooks/05_decision_demo.ipynb` (~20 cases, audit trail shown).

### M9 — Docs & proposal *(Paper track)*
- `docs/{data_report,anomaly_generation,model_card,decision_protocol,fcl_design}.md`, `docs/proposal/` (registered-report-style paper, results marked planned/preliminary), README finalise, CI green.

---

## 5. Critical path vs prototype

- **Must ship (deadline):** M0–M3 + notebook 03 + `data_report.md` + `anomaly_generation.md` + preliminary M4 (≥ 3 detectors, 1 seed) + M6 schema/docker + M7/M8 demo on ~20 cases + CI.
- **Stretch:** USAD/TranAD/MOMENT rows, full 3-seed benchmark, NILM hold-out metrics, simulated-operator study at full prevalence.
- **Design only:** Phase 2 FCL (`docs/fcl_design.md`).

---

## 6. Suggested cadence (5 days to 15 Oct)

| Day | Data (D) | Injection (I) | Modelling + Decision (M) | Paper |
| --- | --- | --- | --- | --- |
| **Fri 10** | Obtain dataset; M0 configs; Steps 1–2 + tests | Inspect schema; injector API + 3 types | pgvector schema + docker; agents interface | §§1–2, 4 |
| **Sat 11** | Step 3 EDA + `data_report.md`; Step 4 splits | Remaining Phase-1 types + tests | Rule-based agents on mocked records | §5 from EDA |
| **Sun 12** | Support features | Gallery notebook; prevalence runs | `features.py`; confidence + routing on mocks | §§6–7 |
| **Mon 13** | NILM (Step 7) | `anomaly_generation.md` | Benchmark ≥4 models; LSTM-AE if time; SHAP | §8, figures |
| **Tue 14** | Freeze data + checksums; freeze seeds | — | LLM agents on real records; demo notebook; simulated operators | Full draft, limitations |
| **Wed 15** | README, CI, licence note | — | — | Proofread + submit |

---

## 7. Risks & mitigations

| Risk | Mitigation |
| --- | --- |
| Dataset arrives late | Start M0 + all schema/interface code against the documented 104-col schema; fixtures = 2 HH × 14 days. |
| No GPU for LLM agents | Use `qwen3.5:4b` + `phi4-mini` on CPU with `R_max = 2`; keep rule-based fallback as the guaranteed path. |
| Derived-column leakage | `test_io.py` asserts no rolling/lag/zscore/flag column survives; `test_features.py` asserts past-only. |
| Benchmark overstates performance | Report naive **and** consistent variants separately; forbid point-adjusted F1. |
| Unverified feedback leaks into `f` | Retrieval is `status = 'verified'` only; H shrinkage toward prior. |
| Thresholds tuned on test | All τ and the ε error target fixed on validation only. |

---

## 8. Ownership 

| Track | Steps / milestones | Notes |
| --- | --- | --- |
| **A — Data & Injection** | M0–M2 (Steps 1–5), `data_report.md`, `anomaly_generation.md` | Critical path; unblocks everyone. |
| **B — Modelling & XAI** | M3–M5 (Steps 6–9): features, NILM, benchmark, SHAP+LIME, records | Owns the results table and figures. |
| **C — Decision layer** | M6–M8 (Steps 10–12): Postgres/pgvector, agents, confidence, routing, HIL | Starts on **mocked records** immediately, so it is never blocked on the dataset. |
| **Paper** | M9 + §18 proposal | Shared across the team from day 1; tracks populate their sections. |

---

## 9. Open questions (team decisions before thresholds are frozen)

- [ ] Source + licence of `Austin_TX_US_2020.csv` — decides what the repo may contain (`DATA_LICENSE.md`).
- [ ] Does the original `Data.zip` contain other cities/years? (Phase 2 FCL nodes.)
- [ ] Fix the **error target ε** (e.g. 2%) and the **false-alarm budget** per household-week.
- [ ] Confirm the three hold-out households (suggested: one solar, one EV, one commercial_small).
- [ ] Postgres via local Docker or a hosted free-tier instance for the demo?
- [ ] Which two LLM model families are available on the team's hardware?

---

## 10. Definition of done — Phase 1

1. `pip install -e .` and the task runner rebuild `clean.parquet` + labels from `data/raw/` deterministically (checksums match).
2. Tests green: `test_io`, `test_physics`, `test_injection`, `test_features`, `test_splits`, `test_deliberation`, `test_confidence`.
3. Notebooks 01–05 reproduce: inspection, EDA (F1–F10), injection gallery, benchmark table from `runs.jsonl`, decision demo with audit trail.
4. Preliminary benchmark (≥ robust z, rules, IF) with event metrics and no point-adjusted F1.
5. Postgres + pgvector schema applied; append-only hash-chained log populated by the demo.
6. Docs complete (`data_report`, `anomaly_generation`, `model_card`, `decision_protocol`, `fcl_design`) + proposal draft; CI green.

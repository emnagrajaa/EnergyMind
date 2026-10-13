# Decision protocol: agents, confidence, tiers and human oversight

> Implementation: `src/energymind/agents/` (M7), `confidence.py`, `policy.py`,
> `simulate_hil.py`, `pipeline.py` (M8). Config: `configs/policy.yaml`,
> `configs/agents.yaml`. Reproduce: `python -m energymind.pipeline --demo`.

This document specifies how EnergyMind turns a flagged anomaly into an auditable
decision: two independent agents deliberate, a learned confidence scores the
decision, a tier decides who acts, and everything is written to an append-only
log.

```
anomaly record ──► "seen before?" ──► two-agent deliberation ──► confidence f
 (M9)                (M6 Qdrant)        (M7 Analyst+Validator)     (M8)
                                              │                        │
                                              └────────► tiered routing ─┴──► audit log
                                                         (autonomous | approval | human-led)
```

---

## 1. The two agents (M7)

Two agents with **different evidence** so that agreement is informative.

| | Agent 1 — Analyst | Agent 2 — Validator |
| --- | --- | --- |
| Reads | detector score/probability, classifier `type`/`origin` probabilities, SHAP/LIME, model severity | electrical-identity residuals, baseline deviation, neighbourhood coherence, NILM residuals, sensor diagnostics, **verified historical precedents** |
| Does **not** read | physics, neighbours, history | model probabilities, SHAP/LIME |
| Question | what does the model say, and what should be done? | is that consistent with physics, the neighbourhood and history? |
| Backend | LLM (`configs/agents.yaml`, family A) | LLM (family B, different model) |
| Fallback | `RuleBasedAnalyst` | `RuleBasedValidator` |

Rules enforced by the contract layer (`agents/base.py`):

* a **closed action set** and a closed **decision-origin** set (`data_quality`,
  `physical_event`, `adversarial`, `unknown`) — agents cannot invent either;
* every proposal must cite **≥1 valid evidence id** from its own `AgentView`;
  uncited or unknown ids are rejected (`validate_citations`);
* **round 1 is blind**; later rounds disclose **only the evidence ids the peer
  cited**, on frozen previous-round snapshots;
* an agent may change its action/origin **only by citing new evidence**
  (`validate_revision`); a change without new evidence is rejected and logged;
* the LLM's own `support` value is self-assessed and is **never** the system
  confidence.

**The loop** (`Deliberator`): bounded by `r_max = 3`; convergence = **same action
and same origin**; non-convergence is preserved (never averaged). Every message
(`deliberation_started`, `analyst_proposal`, `validator_proposal`,
`peer_evidence_disclosed`, `revision_rejected`, `convergence`/`non_convergence`,
`deliberation_completed`) is an `audit_event` in the hash-chained log.

## 2. Closed action set and risk

From `configs/policy.yaml`; agents choose only from this set.

| Action | Risk | Reversible | Typical tier |
| --- | --- | --- | --- |
| `no_action` | low | yes | 1 |
| `flag_data_suspect` | low | yes | 1–2 |
| `request_meter_reread` | low | yes | 1–2 |
| `quarantine_and_estimate` | medium | yes | 2 |
| `open_maintenance_ticket` | medium | yes | 2 |
| `notify_customer` | medium | no | 2 |
| `investigate_tampering` | high | no | 3 |
| `coordinate_with_grid_operator` | high | no | 3 |

**Never automated** (the scenario acts on data, not equipment):
`remote_disconnect, switching, setpoint_change, load_control,
tampering_accusation, field_crew_dispatch`.

## 3. Confidence score `f` (M8)

`f` is the **calibrated probability that the final decision is correct** — not the
model's raw probability:

```
f = sigmoid( b0 + bA·logit(A) + bL·L + bH·logit(H) + bG·G + bQ·Q + bS·S )
```

| Term | Meaning |
| --- | --- |
| **A** | local accuracy — isotonic calibration of the model confidence → P(model correct) |
| **L** | convergence: 1 if converged in round 1, `1-(r-1)/r_max` if converged at `r`, 0 if not |
| **H** | similarity-weighted outcome of the top-k **verified** precedents, shrunk toward a prior |
| **G** | share of the Validator's physical/context checks that support the proposed origin |
| **Q** | data completeness of the window |
| **S** | "seen before" indicator (agents skipped) |

The weights **β are learned on the validation split** (logistic regression), never
hand-set. Last run (`results/m8_decision.json`):

```
b0=-0.43  logit_A=0.05  L=0.59  logit_H=1.58  G=0.00  Q=-0.39  S=0.59
test ECE = 0.041        # reliability of f on the test split
```

The fitted β are themselves a result: here **H (history) and L (convergence)
carry the information**, while `G` carried none (the rule-based checks were
uninformative). Calibration is verified with a reliability diagram and ECE.

## 4. Tiered routing (M8)

`confidence.py` produces `f`; `policy.py` routes with **both `f` and the action's
risk** (a confident wrong `investigate_tampering` costs more than a confident
wrong `flag_data_suspect`).

| Tier | Condition (thresholds chosen on validation) |
| --- | --- |
| **1 Autonomous** | `f ≥ τ_auto` **and** low-risk & reversible **and** converged **and** origin = `data_quality` |
| **2 Human approval** | `f ≥ τ_review`, or any medium-risk action |
| **3 Human-led** | low `f`, non-converged, unseen type (novelty), any high-risk action, `adversarial`/`unknown` origin, or any never-automate action |

`τ_auto` is the **lowest** threshold for which tier-1 decisions on validation meet
the error target `ε = 0.02`; `τ_review` is the lowest `f` at which decisions are
better than chance. Placeholders in the config are replaced by these fitted values
at run time (`choose_thresholds`). Last run: `τ_auto = 0.972`, `τ_review = 0.032`.

**The main evidence that `f` adds value** is the **risk–coverage curve** against
the raw model probability. Last run (lower area = better):

```
area under risk-coverage:  f = 0.015   vs   model_conf = 0.316
```

## 5. Human in the loop (M8)

The cases that routing escalates (tier 2/3) reach a simulated operator, and trust
is measured:

| Indicator | Definition | Direction |
| --- | --- | --- |
| Automation-bias rate | P(operator approves \| proposal wrong) | down |
| Unjustified-rejection rate | P(operator rejects \| proposal right) | down |
| Appropriate reliance | P(final decision correct) | up |
| Unnecessary escalation | share of escalated cases where the AI was right & confident | down |
| Operator load | alerts per operator-hour bound | bounded |

Three simulated operators — **rubber-stamper** (approves 95%), **skeptic**
(rejects 40% at random), **calibrated** (approves with probability `f`) — under
three strategies: `always_auto`, `single_threshold`, `tiered_f`. Example
(`results/m8_decision.json`, 20 test cases):

| strategy | operator | coverage | final error | automation bias | reliance |
| --- | --- | --- | --- | --- | --- |
| always_auto | calibrated | 1.00 | 0.75 | — | 0.25 |
| tiered_f | rubber-stamper | 0.00 | 0.75 | 0.75 | 0.25 |
| tiered_f | calibrated | 0.00 | **0.15** | 0.00 | **0.85** |

Tiered routing + a calibrated operator cut the final error sharply; a
rubber-stamper on the same routing shows the automation-bias cost.

## 6. Memory and the audit trail (M6)

* **Precedents**: only `status = 'verified'` cases are retrieved (filtered k-NN on
  the 64-dim physics signature, same `type`/`origin`). A case never matches itself.
* **Write-back**: `MemoryStore.record_decision` merges `status`, `decision_action`,
  `confidence` and `tier` into the Qdrant payload **and** appends the audit event.
  A decision consistent with the outcome → `verified` (reusable); a wrong one →
  `rejected` (never retrieved).
* **Audit log**: append-only, hash-chained (`row_hash = SHA-256(prev_hash ‖
  canonical_json(row))`); SQLite triggers block `UPDATE`/`DELETE` (Postgres uses
  `REVOKE`). `verify_chain()` recomputes every hash.

## 7. End-to-end pipeline (Step 12)

`src/energymind/pipeline.py`:

```
case → retrieve verified precedents → seen-before gate?
        yes → reuse the historical decision (agents skipped), S=1
        no  → two-agent deliberation (history as Validator evidence)
     → confidence f (learned)  → tier (τ on validation)  → write-back + audit log
```

Outputs a JSON report (`results/m8_decision.json`) with thresholds, β
coefficients, ECE, tier counts, the risk–coverage comparison, the simulated
operator table and the audit event list. `notebooks/05_decision_demo.ipynb` shows
~20 cases with the audit trail.

## 8. Reproduce

```bash
python -W ignore -m energymind.pipeline --demo --n-val 40 --n-test 20   # offline memory
python -W ignore -m energymind.pipeline --qdrant --n-val 40 --n-test 20  # live Qdrant
python -m pytest tests/test_confidence.py tests/test_policy.py tests/test_simulate_hil.py -v
python -m pytest tests/test_deliberation.py -v -k "r_max or without_new_evidence"
```

**Tests**: contract + deliberation (`tests/test_agent_base.py`, `test_analyst.py`,
`test_validator.py`, `test_llm_agent.py`, `test_deliberation.py` — ~195 tests),
confidence/policy/HIL (`test_confidence.py`, `test_policy.py`,
`test_simulate_hil.py`), memory (`test_store.py`, `test_retrieval.py`),
pipeline (`test_agent_pipeline.py`). The full suite runs without a database or an
LLM (fallbacks); Qdrant/shap/lightgbm-dependent tests skip when absent.

## 9. Limitations

* **Simulated operators** — no user study in Phase 1; behaviour is modelled.
* **Rule-based fallbacks** rarely converge on hard consistent anomalies, so in the
  demo most cases land in tier 3 and the stored `decision_action` is often null.
  With converging LLM agents (or seeded decisions) tiers 1/2 populate.
* **Thresholds** are only as good as the validation split; they are re-fitted per
  run and must be frozen (with seeds) before the final results table.
* The `support` field is agent self-assessment; the calibrated confidence is `f`,
  computed downstream — never mixed with the LLM's own number.

**Phase 2**: the same `f` and tiers gate federated continual-learning decisions;
peer-retrieved precedents are forced to tier ≥ 2 at first use (`fcl_design.md`).

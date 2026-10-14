# Decision protocol: agents, confidence, tiers and human oversight

> Implementation: `src/energymind/agents/` (M7), `confidence.py`, `policy.py`,
> `review.py`, `pipeline.py` (M8). Config: `configs/policy.yaml`,
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

**The loop** (`Deliberator`): bounded by `r_max = 3` (set once in
`configs/agents.yaml`; `configs/policy.yaml` intentionally does not duplicate it);
convergence requires the **same origin and the same action risk class** (`low` /
`medium` / `high` from `configs/policy.yaml`). When the agents agree on the origin but
propose different, equally risky actions, the agreed action is the **lower-risk** of
the two (ties: reversible first, then action name). This keeps genuine disagreement
visible — differing origins or differing risk classes do **not** converge —
instead of forcing agreement by aligning the agents' action maps. Non-convergence is
preserved (never averaged); the Analyst's last proposal then becomes the proposed
decision (see §6). The output JSON records `convergence_rate` and
`convergence_rate_by_origin`. Every message
(`deliberation_started`, `analyst_proposal`, `validator_proposal`,
`peer_evidence_disclosed`, `revision_rejected`, `convergence`/`non_convergence`,
`deliberation_completed`) is an `audit_event` in the hash-chained log.

The two agents are **injected, not hard-coded**. The pipeline uses the rule-based agents
by default; to run the same pipeline with the cloud/LLM agents, point
`configs/agents.yaml` at the backend (`backend: openai_compatible`, `base_url`,
`api_key_env`) and pass the agents in::

```python
from energymind.agents.llm_agent import create_analyst_agent, create_validator_agent

report = pipeline.run(
    settings, n_val=40, n_test=20,
    analyst=create_analyst_agent(settings),
    validator=create_validator_agent(settings),
)
```

With `backend: ollama` the same call uses the local models; if the backend is
unreachable the agents fall back to the rule-based ones, so the pipeline never blocks.

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
f = sigmoid( b0 + bA·logit(A) + bL·L_eff + bH·logit(H) + bG·G + bQ·Q + bD·D )
```

| Term | Meaning |
| --- | --- |
| **A** | local accuracy — isotonic calibration of the classifier probability **of the proposed origin** → P(correct). The isotonic map is fitted on a split that is **not** used to fit β |
| **L_eff** | `(1 − D)·L` with `L` = convergence: 1 if converged in round 1, `1-(r-1)/r_max` if converged at `r`, 0 if not. On the reuse path `L_eff = 0` (no deliberation) |
| **H** | similarity-weighted share of the retrieved, **causal**, verified precedents whose `origin_pred == proposal_origin`, shrunk toward the validation base rate `p0`: `(Σ s·o + m·p0) / (Σ s + m)`, `m = 2`; with no precedents `H = p0` |
| **G** | share of the Validator's physical/context checks that support the proposed origin |
| **Q** | data completeness of the window |
| **D** | deliberation skipped: 1 on the "seen before" reuse path, else 0 |

The weights **β are learned with L2-regularised logistic regression (`C=1.0`) on the
validation split**, never hand-set. Last run (`results/m8_decision.json`, 40 val / 20 test):

```
b0=1.30  logit_A=-0.51  L_eff=0.22  logit_H=0.97  G=0.06  Q=0.93  D=0.00
test ECE = 0.318        # reliability of f on the test split
```

The report also carries bootstrap 95% CIs for β (1,000 resamples of the fitting set)
and the **variance-inflation factors** of the features (a VIF > 5 warns of
collinearity; the log notes it). The fitted β are themselves a result: here **H
(history) and Q (completeness) carry the information**, while `G` is weak. Calibration
is verified with a reliability diagram and ECE.

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

The human in the loop is the **real user / human reviewer** the system escalates to —
not a separate "operator" component, and distinct from the grid operator that the
`coordinate_with_grid_operator` action contacts. The flow is:

```
anomaly → Analyst → Validator → proposed decision + confidence f → tier
  tier 1 (autonomous)   → act
  tier 2 (uncertain)    → the human approves or rejects the agents' proposal
  tier 3 (high-risk)    → the human enters their own final action + origin
        → final decision stored (Qdrant payload + audit log)
```

**Tier 2 — uncertain** (`review.py::review_card`, `tier=2`): the human sees the
proposal and may only approve or reject it:

```
The system detected a possible data-quality anomaly.

Proposed action: Quarantine and estimate
Confidence: 61.5%
Reason: Strong data-quality failure indication (data_quality=0.82) with high severity.
Evidence: baseline deviation: anomalous; current vs power identity: broken

Do you approve the proposed decision?
[Approve] [Reject]
```

`Approve` accepts the proposal (reusable iff the outcome confirms it); `Reject` takes no
action and does not reuse the proposal.

**Tier 3 — high-risk** (`tier=3`): the human does **not** approve the agents' proposal;
they enter their own action and origin. The proposal is shown for reference only:

```
The system detected a possible possible tampering anomaly.
TIER 3 (human-led, high-risk): decide the final action yourself.
The agents' proposal below is shown FOR REFERENCE ONLY and is NOT approved.

Proposed action (reference only): Investigate tampering
Predicted origin: possible tampering
Confidence: 61.5% (risk: high)
Reason: ...
Evidence: ...

Enter your own final action and origin.
```

The entered action must be in the policy's allowed action set (`configs/policy.yaml`,
including an explicit `no_action` when present); the origin must be one of the closed
origin set. Invalid input is re-prompted (terminal) or rejected (programmatic). The
human's action/origin are written to the append-only log as a `human_custom_decision`
event. A tier-3 decision is reusable only when it matches the observed outcome — the
agents' proposal is **never** reused merely because the human answered.

Every escalated test case's card is written to the report under `review_cards`, and the
resolution counts under `human_review` (`requested` / `approved` / `rejected` /
`custom` / `more_info_requested` / `pending_human` / `auto`).

**Only after the human answers is the final decision stored** (`record_decision` →
Qdrant payload + append-only audit event). A case with no answer stays
`pending_human` (status `proposed`) — the pipeline never invents a human response and
never silently approves/rejects/reuses.

Batch runs: `--human-answer approve|reject` answers every **tier-2** card; add
`--human-action <action> --human-origin <origin>` for **tier-3** cards. Per-case answers
go through `human_decisions={anomaly_id: value}` (tier 2: a string; tier 3: a
`{"action", "origin"}` mapping). For **real terminal review**, pass
`--interactive-human-review` (or `interactive=True`): tier-2 cards prompt
`1` Approve / `2` Reject, tier-3 cards prompt for the human's own action and origin, and
invalid input is re-prompted — no answer is ever defaulted or simulated.

`f` and the tier table above still decide **which** cases reach the human; they do not
model *how* the human responds. Trust metrics that need real users (automation-bias
rate, appropriate reliance, reviewer load) are a Phase-2 user study, not a simulation.

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
     → confidence f (learned)  → tier (τ on validation)
        tier 1 → autonomous: act and store the decision
        tier 2/3 → human review card → approve / reject / request more information
     → store the final decision (Qdrant payload) + append-only audit log
```

Outputs a JSON report (`results/m8_decision.json`) with thresholds, β
coefficients, ECE, tier counts, the risk–coverage comparison, the human-review
summary (`human_review`) with the cards sent to the human (`review_cards`), and the
audit event list. `notebooks/05_decision_demo.ipynb` shows ~20 cases with the audit
trail.

## 8. Reproduce

```bash
python -W ignore -m energymind.pipeline --n-val 40 --n-test 20                    # wait for the human
python -W ignore -m energymind.pipeline --n-val 40 --n-test 20 --human-answer approve
python -W ignore -m energymind.pipeline --n-val 40 --n-test 20 \
    --human-answer approve --human-action flag_data_suspect --human-origin data_quality
python -W ignore -m energymind.pipeline --n-val 40 --n-test 20 --interactive-human-review  # real prompts
python -W ignore -m energymind.pipeline --qdrant --reset --n-val 40 --n-test 20   # live Qdrant
python -m pytest tests/test_confidence.py tests/test_policy.py tests/test_review.py -v
python -m pytest tests/test_deliberation.py -v -k "r_max or without_new_evidence"
```

**Tests**: contract + deliberation (`tests/test_agent_base.py`, `test_analyst.py`,
`test_validator.py`, `test_llm_agent.py`, `test_deliberation.py` — ~195 tests),
confidence/policy (`test_confidence.py`, `test_policy.py`), human review
(`test_review.py`), memory (`test_store.py`, `test_retrieval.py`),
pipeline (`test_agent_pipeline.py`). The full suite runs without a database or an
LLM (fallbacks); the LLM agents are only used when `configs/agents.yaml` points at a
reachable backend. Qdrant/shap/lightgbm-dependent tests skip when absent.

## 9. Limitations

* **Human review is not yet a UI** — the card is rendered into the report and the
  human's answer is supplied through `human_decisions` / `--human-answer`; wiring it to
  a real screen (and measuring trust with users) is Phase 2.
* **Rule-based fallbacks** rarely converge on hard consistent anomalies, so in the
  demo most cases land in tier 3 and the stored `decision_action` is often null.
  With converging LLM agents (or seeded decisions) tiers 1/2 populate.
* **Thresholds** are only as good as the validation split; they are re-fitted per
  run and must be frozen (with seeds) before the final results table.
* The `support` field is agent self-assessment; the calibrated confidence is `f`,
  computed downstream — never mixed with the LLM's own number.

**Phase 2**: the same `f` and tiers gate federated continual-learning decisions;
peer-retrieved precedents are forced to tier ≥ 2 at first use (`fcl_design.md`).

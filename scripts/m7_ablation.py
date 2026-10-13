"""Rule-based vs LLM ablation for the M7 two-agent deliberation.

Runs the same **real** anomaly records through both agent back-ends and writes one
row per (case, mode) so the paper can compare:

    rule : RuleBasedAnalyst + RuleBasedValidator (deterministic, offline)
    llm  : the configured LLM agents (Ollama), falling back per call if unreachable

Rows are written to ``results/m7_ablation.json`` (and printed as a table). Every
deliberation message is also written to a fresh hash-chained audit log, and the
event count per case is recorded.

    python scripts/m7_ablation.py --modes rule                 # offline rows
    python scripts/m7_ablation.py --modes rule llm --endpoint https://<ollama>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from energymind import evaluate, io  # noqa: E402
from energymind.agent_pipeline import _build_record, run_case  # noqa: E402
from energymind.agents import (  # noqa: E402
    RuleBasedAnalyst,
    RuleBasedValidator,
    create_analyst_agent,
    create_validator_agent,
)
from energymind.agents.llm_agent import OllamaBackend, OpenAICompatibleBackend  # noqa: E402
from energymind.config import load_settings  # noqa: E402
from energymind.detect.rules import Rules  # noqa: E402
from energymind.retrieval import InMemoryRetriever  # noqa: E402
from energymind.store import AuditLog, MemoryStore  # noqa: E402


def _agents(mode: str, args, settings):
    if mode == "rule":
        return RuleBasedAnalyst(), RuleBasedValidator()
    if mode != "llm":
        raise ValueError(f"unknown mode: {mode}")
    if args.backend == "ollama":
        backend = OllamaBackend(host=args.endpoint)
    else:
        import os

        backend = OpenAICompatibleBackend(
            base_url=args.endpoint or settings.agents.openai_compatible.base_url,
            api_key=os.environ.get(settings.agents.openai_compatible.api_key_env, ""),
        )
    analyst = create_analyst_agent(settings=settings, backend=backend, model=args.analyst_model)
    validator = create_validator_agent(
        settings=settings, backend=backend, model=args.validator_model
    )
    if args.timeout is not None:
        analyst.timeout_s = float(args.timeout)
        validator.timeout_s = float(args.timeout)
    return analyst, validator


def main() -> int:
    parser = argparse.ArgumentParser(description="M7 rule-based vs LLM ablation.")
    parser.add_argument("--modes", nargs="+", default=["rule"], choices=["rule", "llm"])
    parser.add_argument("--cases", type=int, default=8)
    parser.add_argument("--r-max", type=int, default=3)
    parser.add_argument("--backend", choices=["ollama", "openai_compatible"], default="ollama")
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--timeout", type=float, default=None, help="per-call timeout override (s)")
    parser.add_argument("--analyst-model", default="qwen2.5:3b")
    parser.add_argument("--validator-model", default="granite4.2:8b")
    parser.add_argument("--feature-file", default="data/features/features.parquet")
    parser.add_argument("--label-file", default="data/injected/labels.parquet")
    parser.add_argument("--output-json", default="results/m7_ablation.json")
    args = parser.parse_args()

    io.load_env()
    settings = load_settings()
    frame = pd.read_parquet(io.resolve(args.feature_file))
    labels = pd.read_parquet(io.resolve(args.label_file))
    events = evaluate.expand_events(labels)

    t = settings.data.splits.temporal
    off = settings.data.time.assume_utc_offset_h
    test_start = pd.Timestamp(t.test.start).tz_localize("UTC") - pd.Timedelta(hours=off)
    test_ids = events.loc[events["t_start"] >= test_start, "anomaly_id"].unique()[: args.cases]

    detector_scores = Rules().fit(frame).score(frame)
    records = {cid: _build_record(frame, events, cid, detector_scores) for cid in test_ids}

    rows: list[dict] = []
    print(
        f"\n{'case':<14} | {'mode':<5} | {'conv':<5} | {'rnd':<3} | {'fallback':<8} | "
        f"{'analyst':<24} | {'validator':<24} | {'audit':<5} | latency"
    )
    print("-" * 130)
    for mode in args.modes:
        analyst, validator = _agents(mode, args, settings)
        for cid in test_ids:
            # Empty memory: isolate the agents from the "seen before" path.
            store = MemoryStore(InMemoryRetriever(), AuditLog(sqlite_path=":memory:"))
            t0 = time.perf_counter()
            outcome = run_case(
                records[cid],
                store,
                settings=settings,
                r_max=args.r_max,
                analyst=analyst,
                validator=validator,
            )
            latency = round(time.perf_counter() - t0, 2)
            d = outcome.decision
            result = outcome.deliberation
            fallback = bool(result and (result.analyst_fallback or result.validator_fallback))
            row = {
                "case_id": str(cid),
                "true_type": records[cid]["type_pred"],
                "true_origin": records[cid]["origin_pred"],
                "mode": mode,
                "path": outcome.path,
                "converged": d["converged"],
                "rounds_used": result.rounds_used if result else None,
                "agreed_action": d["action"],
                "agreed_origin": d["origin"],
                "analyst_action": d["analyst"].get("action"),
                "analyst_origin": d["analyst"].get("origin"),
                "validator_action": d["validator"].get("action"),
                "validator_origin": d["validator"].get("origin"),
                "fallback_used": fallback,
                "audit_events": len(store.audit.events()),
                "audit_valid": store.audit.verify_chain()["valid"],
                "latency_s": latency,
            }
            rows.append(row)
            print(
                f"{str(cid)[:12]:<14} | {mode:<5} | {str(d['converged']):<5} | "
                f"{str(row['rounds_used']):<3} | {str(fallback):<8} | "
                f"{str(row['analyst_action']):<24} | {str(row['validator_action']):<24} | "
                f"{row['audit_events']:<5} | {latency}s"
            )

    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "modes": args.modes,
        "cases": len(test_ids),
        "r_max": args.r_max,
        "rows": rows,
        "note": "M7 rule-based vs LLM ablation (LLM rows fall back if the endpoint is unreachable)",
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {out}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Smoke validation script for real LLM-backed M7 deliberation.

Executes real LLM inference across Analyst and Validator, verifies evidence
isolation, citation validation, multi-round deliberation, and audit hash-chain integrity.
Never prints or logs API keys or credentials.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from energymind.agents.deliberation import Deliberator
from energymind.agents.llm_agent import (
    OllamaBackend,
    OpenAICompatibleBackend,
    create_analyst_agent,
    create_validator_agent,
)
from energymind.config import load_settings
from energymind.store import AuditLog

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("smoke_m7_llm")

# Representative anomaly cases for smoke validation
SAMPLE_CASES: list[dict[str, Any]] = [
    {
        "anomaly_id": "CASE_DQ_001",
        "household_id": "HH_00072",
        "t_start": "2020-06-01T12:00:00Z",
        "t_end": "2020-06-01T14:00:00Z",
        "duration_h": 2.0,
        "detection": {"model": "isolation_forest", "p_anomaly": 0.94, "score": 2.15},
        "classification": {
            "model": "lightgbm",
            "type_probs": {"dead_sensor": 0.85, "dropout": 0.10, "drift": 0.05},
            "origin_probs": {"data_quality": 0.90, "physical_event": 0.08, "adversarial": 0.02},
        },
        "explanation": {
            "shap_top": [["active_power_kw", 0.35421], ["voltage_v", -0.12345]],
            "lime_top": [["active_power_kw", 0.33211], ["current_a", 0.15432]],
            "reliability": 0.78,
        },
        "severity": "high",
        "evidence": [
            {"check": "current_vs_power_identity", "rel_residual": 0.0012, "status": "holds"},
            {
                "check": "baseline_deviation",
                "observed_kw": 0.12,
                "expected_kw": 1.45,
                "z": -2.85,
                "status": "anomalous",
            },
            {"check": "neighborhood_share", "share": 0.03, "status": "isolated"},
        ],
    },
    {
        "anomaly_id": "CASE_PHYS_002",
        "household_id": "HH_00104",
        "t_start": "2020-07-15T18:00:00Z",
        "t_end": "2020-07-15T20:00:00Z",
        "duration_h": 2.0,
        "detection": {"model": "isolation_forest", "p_anomaly": 0.91, "score": 1.95},
        "classification": {
            "model": "lightgbm",
            "type_probs": {"extreme_spike": 0.80, "load_shift": 0.15, "drift": 0.05},
            "origin_probs": {"physical_event": 0.88, "data_quality": 0.08, "adversarial": 0.04},
        },
        "explanation": {
            "shap_top": [["active_power_kw", 0.52100], ["current_a", 0.31200]],
            "lime_top": [["active_power_kw", 0.49000], ["reactive_power_kvar", 0.22000]],
            "reliability": 0.85,
        },
        "severity": "high",
        "evidence": [
            {"check": "current_vs_power_identity", "rel_residual": 0.0005, "status": "holds"},
            {
                "check": "baseline_deviation",
                "observed_kw": 5.80,
                "expected_kw": 1.20,
                "z": 4.10,
                "status": "anomalous",
            },
            {"check": "neighborhood_share", "share": 0.65, "status": "coherent"},
        ],
    },
    {
        "anomaly_id": "CASE_AMBIG_003",
        "household_id": "HH_00031",
        "t_start": "2020-08-10T02:00:00Z",
        "t_end": "2020-08-10T03:00:00Z",
        "duration_h": 1.0,
        "detection": {"model": "isolation_forest", "p_anomaly": 0.55, "score": 0.82},
        "classification": {
            "model": "lightgbm",
            "type_probs": {"dropout": 0.40, "drift": 0.35, "dead_sensor": 0.25},
            "origin_probs": {"data_quality": 0.50, "physical_event": 0.35, "adversarial": 0.15},
        },
        "explanation": {
            "shap_top": [["active_power_kw", 0.15000], ["voltage_v", 0.08000]],
            "lime_top": [["active_power_kw", 0.12000], ["current_a", 0.05000]],
            "reliability": 0.45,
        },
        "severity": "low",
        "evidence": [
            {"check": "current_vs_power_identity", "rel_residual": 0.0080, "status": "holds"},
            {
                "check": "baseline_deviation",
                "observed_kw": 0.45,
                "expected_kw": 0.60,
                "z": -0.85,
                "status": "normal",
            },
            {"check": "neighborhood_share", "share": 0.08, "status": "isolated"},
        ],
    },
]


def run_smoke_test(
    backend_type: str,
    analyst_model: str,
    validator_model: str,
    endpoint: str | None = None,
    r_max: int = 2,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Execute end-to-end smoke validation of M7 pipeline."""
    settings = load_settings()

    # Configure backend
    if backend_type == "ollama":
        host = endpoint or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        logger.info("Using Ollama backend at %s", host)
        analyst_backend = OllamaBackend(host=host)
        validator_backend = OllamaBackend(host=host)
    elif backend_type == "openai_compatible":
        base_url = (
            endpoint
            or os.environ.get("OPENAI_BASE_URL")
            or settings.agents.openai_compatible.base_url
        )
        api_key = os.environ.get(settings.agents.openai_compatible.api_key_env, "")
        logger.info("Using OpenAI-compatible backend at %s", base_url)
        analyst_backend = OpenAICompatibleBackend(base_url=base_url, api_key=api_key)
        validator_backend = OpenAICompatibleBackend(base_url=base_url, api_key=api_key)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")

    analyst_agent = create_analyst_agent(
        settings=settings, backend=analyst_backend, model=analyst_model
    )
    analyst_agent.timeout_s = 180.0

    validator_agent = create_validator_agent(
        settings=settings, backend=validator_backend, model=validator_model
    )
    validator_agent.timeout_s = 180.0

    audit = AuditLog(backend="sqlite", sqlite_path=":memory:")
    deliberator = Deliberator(
        analyst=analyst_agent,
        validator=validator_agent,
        r_max=r_max,
        audit_log=audit,
    )

    results: list[dict[str, Any]] = []
    print("\n" + "=" * 90)
    print(f"ENERGYMIND M7 REAL LLM SMOKE VALIDATION ({backend_type})")
    print(f"Analyst Model  : {analyst_model}")
    print(f"Validator Model: {validator_model}")
    header = (
        f"{'Case ID':<14} | {'Analyst':<22} | {'Validator':<20} | "
        f"{'Rnd':<3} | {'Conv':<5} | {'Fallbk':<6} | {'Latency'}"
    )
    print("=" * 90)
    print(header)
    print("-" * 90)

    for record in SAMPLE_CASES:
        cid = record["anomaly_id"]
        t0 = time.time()
        delib_res = deliberator.deliberate(record)
        latency_s = round(time.time() - t0, 2)

        a_str = f"{delib_res.analyst_final.action.value} ({delib_res.analyst_final.origin.value})"
        v_str = (
            f"{delib_res.validator_final.action.value} ({delib_res.validator_final.origin.value})"
        )
        fallback_str = (
            "Yes" if (delib_res.analyst_fallback or delib_res.validator_fallback) else "No"
        )
        conv_str = "True" if delib_res.converged else "False"

        row = (
            f"{cid:<14} | {a_str:<22} | {v_str:<20} | "
            f"{delib_res.rounds_used:<3} | {conv_str:<5} | {fallback_str:<6} | {latency_s}s"
        )
        print(row)

        results.append(
            {
                "case_id": cid,
                "converged": delib_res.converged,
                "rounds_used": delib_res.rounds_used,
                "agreed_action": (
                    delib_res.agreed_action.value if delib_res.agreed_action else None
                ),
                "agreed_origin": (
                    delib_res.agreed_origin.value if delib_res.agreed_origin else None
                ),
                "analyst": {
                    "action": delib_res.analyst_final.action.value,
                    "origin": delib_res.analyst_final.origin.value,
                    "evidence_ids": list(delib_res.analyst_final.evidence_ids),
                    "fallback_used": delib_res.analyst_final.fallback_used,
                },
                "validator": {
                    "action": delib_res.validator_final.action.value,
                    "origin": delib_res.validator_final.origin.value,
                    "evidence_ids": list(delib_res.validator_final.evidence_ids),
                    "fallback_used": delib_res.validator_final.fallback_used,
                },
                "latency_s": latency_s,
            }
        )

    # Verify audit chain
    chain_status = audit.verify_chain()
    print("-" * 90)
    print(f"Audit Hash-Chain: Valid={chain_status['valid']}, Total Events={chain_status['events']}")
    print("=" * 90 + "\n")

    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "backend": backend_type,
        "analyst_model": analyst_model,
        "validator_model": validator_model,
        "cases_evaluated": len(results),
        "audit_verification": chain_status,
        "results": results,
        "note": "integration smoke test - NOT M7.6 evaluation",
    }

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        logger.info("Saved smoke results to %s", output_path)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke validation runner for M7 real LLMs")
    parser.add_argument(
        "--backend",
        choices=["ollama", "openai_compatible"],
        default="ollama",
        help="LLM transport backend",
    )
    parser.add_argument(
        "--analyst-model",
        default="qwen2.5:3b",
        help="Analyst model name",
    )
    parser.add_argument(
        "--validator-model",
        default="granite4.2:8b",
        help="Validator model name",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Custom endpoint URL (Ollama host or OpenAI base URL)",
    )
    parser.add_argument(
        "--r-max",
        type=int,
        default=2,
        help="Max deliberation rounds",
    )
    parser.add_argument(
        "--output-json",
        default="results/m7_real_llm_smoke.json",
        help="Output path for sanitized smoke results",
    )
    args = parser.parse_args()

    run_smoke_test(
        backend_type=args.backend,
        analyst_model=args.analyst_model,
        validator_model=args.validator_model,
        endpoint=args.endpoint,
        r_max=args.r_max,
        output_path=args.output_json,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

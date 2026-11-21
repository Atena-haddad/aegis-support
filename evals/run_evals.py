import json
import numpy as np
from src.agent.graph import AegisAgentOrchestrator

GOLDEN_EVAL_SET = [
    # Routine Fast-Path cases (should bypass LLM)
    {"query": "How do I download VAT invoices for accounting?", "expected_path": "CLASSICAL_ML_FAST_PATH", "expected_escalate": False},
    {"query": "How do I invite a new team member to our workspace?", "expected_path": "CLASSICAL_ML_FAST_PATH", "expected_escalate": False},
    {"query": "Can we switch from monthly to annual billing cycle?", "expected_path": "CLASSICAL_ML_FAST_PATH", "expected_escalate": False},
    {"query": "Please add dark mode support to the analytics console.", "expected_path": "CLASSICAL_ML_FAST_PATH", "expected_escalate": False},
    # Complex Agentic RAG cases (should invoke LLM + ground in SOPs)
    {"query": "Production inference endpoint returning HTTP 502 Bad Gateway.", "expected_path": "AGENTIC_RAG_WITH_CRITIC", "expected_escalate": False},
    {"query": "Locked out of SSO login with SAML error 403.", "expected_path": "AGENTIC_RAG_WITH_CRITIC", "expected_escalate": False},
    {"query": "Python SDK throws TimeoutError when batch size exceeds 64.", "expected_path": "AGENTIC_RAG_WITH_CRITIC", "expected_escalate": False},
    {"query": "Vector index sync stuck at 99% for over two hours.", "expected_path": "AGENTIC_RAG_WITH_CRITIC", "expected_escalate": False},
    {"query": "Webhook payload drops nested JSON metadata intermittently.", "expected_path": "AGENTIC_RAG_WITH_CRITIC", "expected_escalate": False},
    {"query": "User dev@acme.io with key ags_live_12345 getting HTTP 401 invalid bearer token.", "expected_path": "AGENTIC_RAG_WITH_CRITIC", "expected_escalate": False},
    # Out-of-Distribution / Adversarial cases (must NOT hallucinate -> escalate)
    {"query": "Our quantum satellite telemetry uplink is desynchronizing with the orbital relay.", "expected_path": "AGENTIC_RAG_WITH_CRITIC", "expected_escalate": True},
    {"query": "How do I configure BGP peering on a Cisco ASR 9000 router?", "expected_path": "AGENTIC_RAG_WITH_CRITIC", "expected_escalate": True},
]


def run_benchmark():
    orchestrator = AegisAgentOrchestrator()
    records = []

    for item in GOLDEN_EVAL_SET:
        out = orchestrator.run(item["query"])
        records.append({
            "query": item["query"],
            "execution_path": out["execution_path"],
            "expected_path": item["expected_path"],
            "path_correct": out["execution_path"] == item["expected_path"],
            "is_grounded": out["is_grounded"],
            "escalate_to_human": out["escalate_to_human"],
            "expected_escalate": item["expected_escalate"],
            "escalation_correct": out["escalate_to_human"] == item["expected_escalate"],
            "redacted_pii": out["redacted_pii"],
            "ml_latency_ms": out["ml_latency_ms"],
            "total_latency_ms": out["total_latency_ms"],
        })

    total = len(records)
    fast_path_count = sum(1 for r in records if r["execution_path"] == "CLASSICAL_ML_FAST_PATH")
    routing_acc = sum(1 for r in records if r["path_correct"]) / total * 100
    grounding_rate = sum(1 for r in records if r["is_grounded"]) / total * 100
    escalation_acc = sum(1 for r in records if r["escalation_correct"]) / total * 100
    ml_p95 = float(np.percentile([r["ml_latency_ms"] for r in records], 95))
    total_p95 = float(np.percentile([r["total_latency_ms"] for r in records], 95))

    summary = {
        "total_eval_cases": total,
        "llm_call_reduction_pct": round(fast_path_count / total * 100, 1),
        "routing_accuracy_pct": round(routing_acc, 1),
        "critic_grounding_rate_pct": round(grounding_rate, 1),
        "ood_escalation_accuracy_pct": round(escalation_acc, 1),
        "ml_router_p95_latency_ms": round(ml_p95, 2),
        "end_to_end_p95_latency_ms": round(total_p95, 2),
    }

    with open("evals/benchmark_results.json", "w") as f:
        json.dump({"summary": summary, "details": records}, f, indent=2)

    print("\n=== AEGISSUPPORT BENCHMARK RESULTS ===")
    for k, v in summary.items():
        print(f"{k:32s}: {v}")


if __name__ == "__main__":
    run_benchmark()
import time
import joblib
import numpy as np
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer


@dataclass
class RoutingDecision:
    intent: str
    intent_confidence: float
    priority: str
    is_ood_novel: bool
    anomaly_score: float
    route_to_agent: bool
    routing_reason: str
    latency_ms: float


class MLTriageRouter:
    """
    Fast Classical ML + Deep Embedding Router.
    Filters out simple/known requests (<15ms) and escalates complex, high-priority,
    or Out-of-Distribution (novel) queries to the Agentic LLM pipeline.
    """

    def __init__(self, artifacts_dir: str = "artifacts"):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.intent_clf = joblib.load(f"{artifacts_dir}/intent_clf.joblib")
        self.priority_clf = joblib.load(f"{artifacts_dir}/priority_clf.joblib")
        self.route_clf = joblib.load(f"{artifacts_dir}/route_clf.joblib")
        self.ood_detector = joblib.load(f"{artifacts_dir}/ood_detector.joblib")

    def route(self, text: str) -> RoutingDecision:
        start = time.perf_counter()
        emb = self.embedder.encode([text], normalize_embeddings=True)

        # 1. Intent prediction & confidence
        intent_probs = self.intent_clf.predict_proba(emb)[0]
        best_idx = int(np.argmax(intent_probs))
        intent = str(self.intent_clf.classes_[best_idx])
        intent_conf = float(intent_probs[best_idx])

        # 2. Priority prediction
        priority = str(self.priority_clf.predict(emb)[0])

        # 3. Out-of-Distribution (Novelty) check
        anomaly_score = float(self.ood_detector.decision_function(emb)[0])
        is_ood = bool(anomaly_score < -0.04 or intent_conf < 0.45)

        # 4. Agent escalation decision
        needs_agent_pred = int(self.route_clf.predict(emb)[0])
        if is_ood:
            route_to_agent = True
            reason = "Escalated: Out-of-Distribution / novel query detected"
        elif priority == "high" or needs_agent_pred == 1:
            route_to_agent = True
            reason = f"Escalated: High complexity or {priority} priority {intent}"
        else:
            route_to_agent = False
            reason = f"Fast-Path: Routine {intent} request handled without LLM"

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return RoutingDecision(
            intent=intent,
            intent_confidence=round(intent_conf, 3),
            priority=priority,
            is_ood_novel=is_ood,
            anomaly_score=round(anomaly_score, 4),
            route_to_agent=route_to_agent,
            routing_reason=reason,
            latency_ms=latency_ms,
        )


if __name__ == "__main__":
    router = MLTriageRouter()
    test_queries = [
        "How do I download VAT invoices for accounting?",
        "Production inference endpoint returning HTTP 502 Bad Gateway.",
        "Our quantum satellite telemetry uplink is desynchronizing with the orbital relay.",
    ]
    for q in test_queries:
        res = router.route(q)
        print(f"\nQuery: {q}\n  -> {res}")
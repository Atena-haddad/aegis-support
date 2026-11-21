import os
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def generate_synthetic_tickets() -> pd.DataFrame:
    """
    Generates a clean, labeled dataset of support/engineering tickets
    covering known operational intents + priority levels.
    """
    samples = [
        # account_access
        ("I cannot log into my dashboard after resetting my 2FA token.", "account_access", "high", True),
        ("Locked out of SSO login with SAML error 403.", "account_access", "high", True),
        ("How do I invite a new team member to our workspace?", "account_access", "low", False),
        ("Password reset email never arrived in my inbox.", "account_access", "medium", False),
        ("Need to update role permissions for our data science interns.", "account_access", "low", False),
        ("API key authentication failing with invalid bearer token.", "account_access", "high", True),
        ("Where do I enable single sign-on for Okta?", "account_access", "low", False),
        ("My session keeps expiring every 5 minutes.", "account_access", "medium", True),
        # billing
        ("We were double charged on our October cloud invoice.", "billing", "high", True),
        ("How do I download VAT invoices for accounting?", "billing", "low", False),
        ("Want to upgrade our workspace from Starter to Pro tier.", "billing", "low", False),
        ("Credit card payment failed during automatic renewal.", "billing", "high", True),
        ("Can we switch from monthly to annual billing cycle?", "billing", "low", False),
        ("Unexpected overage charges on vector storage this month.", "billing", "medium", True),
        ("Update billing email address for finance department.", "billing", "low", False),
        ("Requesting a refund for unused seats cancelled yesterday.", "billing", "medium", True),
        # technical_bug
        ("Production inference endpoint returning HTTP 502 Bad Gateway.", "technical_bug", "high", True),
        ("Webhook payload drops nested JSON metadata intermittently.", "technical_bug", "high", True),
        ("Python SDK throws TimeoutError when batch size exceeds 64.", "technical_bug", "medium", True),
        ("Vector index sync stuck at 99% for over two hours.", "technical_bug", "high", True),
        ("CSV export button is unresponsive on Safari browser.", "technical_bug", "low", False),
        ("Memory leak observed in worker container after 12 hours.", "technical_bug", "high", True),
        ("Rate limit headers show negative remaining quota.", "technical_bug", "medium", True),
        ("Embeddings endpoint returns NaN values for empty strings.", "technical_bug", "high", True),
        # feature_request
        ("Please add dark mode support to the analytics console.", "feature_request", "low", False),
        ("Would love native Terraform provider support for deployments.", "feature_request", "low", False),
        ("Can you support EU-central Frankfurt data residency?", "feature_request", "medium", True),
        ("Add Slack alert integration when latency exceeds threshold.", "feature_request", "low", False),
        ("Support custom OpenTelemetry exporters for tracing.", "feature_request", "medium", False),
        ("Allow exporting evaluation metrics directly to Parquet.", "feature_request", "low", False),
        ("Feature request: role-based access control per project.", "feature_request", "medium", False),
        ("Add webhook signature verification examples in Python docs.", "feature_request", "low", False),
    ]

    # Multiply with slight phrasing variations so train/test split has solid support
    rows = []
    prefixes = ["", "Urgent: ", "Hi team, ", "Help needed - "]
    for idx, (text, intent, priority, needs_agent) in enumerate(samples):
        for p_idx, prefix in enumerate(prefixes):
            rows.append({
                "ticket_id": f"TKT-{idx:03d}-{p_idx}",
                "text": f"{prefix}{text}".strip(),
                "intent": intent,
                "priority": priority,
                "needs_agent": int(needs_agent),
            })
    return pd.DataFrame(rows)


def train_router():
    print("1. Loading dataset...")
    df = generate_synthetic_tickets()
    df.to_csv("data/tickets/tickets_dataset.csv", index=False)

    print("2. Encoding text with Deep Learning Sentence-Transformer (all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    X = embedder.encode(df["text"].tolist(), normalize_embeddings=True)

    X_train, X_test, y_intent_train, y_intent_test, y_prio_train, y_prio_test, y_route_train, y_route_test = (
        train_test_split(
            X,
            df["intent"],
            df["priority"],
            df["needs_agent"],
            test_size=0.2,
            random_state=42,
            stratify=df["intent"],
        )
    )

    print("3. Training Classical ML classifiers (Intent, Priority, Escalation Router)...")
    intent_clf = LogisticRegression(max_iter=1000, C=2.0)
    intent_clf.fit(X_train, y_intent_train)

    priority_clf = LogisticRegression(max_iter=1000, C=2.0)
    priority_clf.fit(X_train, y_prio_train)

    route_clf = LogisticRegression(max_iter=1000, C=2.0)
    route_clf.fit(X_train, y_route_train)

    print("4. Training Out-of-Distribution (Novelty) Detector (IsolationForest)...")
    ood_detector = IsolationForest(contamination=0.05, random_state=42)
    ood_detector.fit(X_train)

    # Evaluate on held-out test set
    intent_preds = intent_clf.predict(X_test)
    route_preds = route_clf.predict(X_test)
    intent_f1 = f1_score(y_intent_test, intent_preds, average="macro")
    route_f1 = f1_score(y_route_test, route_preds, average="macro")

    print("\n--- Classical ML Router Evaluation ---")
    print(f"Intent Macro F1:     {intent_f1:.3f}")
    print(f"Escalation Macro F1: {route_f1:.3f}")
    print("\nIntent Classification Report:")
    print(classification_report(y_intent_test, intent_preds))

    # Save trained artifacts
    joblib.dump(intent_clf, os.path.join(ARTIFACTS_DIR, "intent_clf.joblib"))
    joblib.dump(priority_clf, os.path.join(ARTIFACTS_DIR, "priority_clf.joblib"))
    joblib.dump(route_clf, os.path.join(ARTIFACTS_DIR, "route_clf.joblib"))
    joblib.dump(ood_detector, os.path.join(ARTIFACTS_DIR, "ood_detector.joblib"))
    print(f"Saved all trained ML router artifacts to ./{ARTIFACTS_DIR}/")


if __name__ == "__main__":
    train_router()
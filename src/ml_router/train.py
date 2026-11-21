import os
import joblib
import numpy as np
import pandas as pd
from fastembed import TextEmbedding
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedKFold, cross_val_score

MODEL_NAME = "BAAI/bge-small-en-v1.5"


def train_and_save(data_path: str = "data/tickets/tickets_dataset.csv", out_dir: str = "artifacts"):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(data_path)

    embedder = TextEmbedding(model_name=MODEL_NAME)
    X = np.array(list(embedder.embed(df["text"].tolist())))

    y_intent = df["intent"].values
    y_priority = df["priority"].values
    y_agent = df["needs_agent"].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    intent_clf = LogisticRegression(C=5.0, max_iter=1000, random_state=42)
    intent_f1 = cross_val_score(intent_clf, X, y_intent, cv=skf, scoring="f1_macro").mean()
    intent_clf.fit(X, y_intent)

    priority_clf = LogisticRegression(C=5.0, max_iter=1000, random_state=42)
    priority_f1 = cross_val_score(priority_clf, X, y_priority, cv=skf, scoring="f1_macro").mean()
    priority_clf.fit(X, y_priority)

    route_clf = LogisticRegression(C=5.0, max_iter=1000, random_state=42)
    route_f1 = cross_val_score(route_clf, X, y_agent, cv=skf, scoring="f1").mean()
    route_clf.fit(X, y_agent)

    ood_detector = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    ood_detector.fit(X)

    joblib.dump(intent_clf, f"{out_dir}/intent_clf.joblib")
    joblib.dump(priority_clf, f"{out_dir}/priority_clf.joblib")
    joblib.dump(route_clf, f"{out_dir}/route_clf.joblib")
    joblib.dump(ood_detector, f"{out_dir}/ood_detector.joblib")

    print("=== Classical ML Router Training Complete (ONNX FastEmbed) ===")
    print(f"5-Fold CV Macro F1 (Intent)  : {intent_f1:.3f}")
    print(f"5-Fold CV Macro F1 (Priority): {priority_f1:.3f}")
    print(f"5-Fold CV F1 (Agent Routing) : {route_f1:.3f}")


if __name__ == "__main__":
    train_and_save()

"""Local logistic regression baseline for a geographic silo (Europe or LATAM)."""
import json
import os
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

warnings.simplefilter("ignore")


def classification_metrics(y_true, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "f1_score": float(f1_score(y_true, preds, zero_division=0)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
    }


def load_data():
    data_path = "data.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    print("Loading local geographic silo dataset...")
    df = pd.read_csv(data_path, encoding="latin1")
    target = "Late_delivery_risk"

    feature_columns = [
        "Days for shipment (scheduled)",
        "Order Item Quantity",
        "Sales",
        "Product Price",
        "Shipping Mode",
        "Market",
        "Order Region",
        "Customer Segment",
    ]
    available = [c for c in feature_columns if c in df.columns]
    df_clean = df[available + [target]].dropna().copy()

    for col in ["Shipping Mode", "Market", "Order Region", "Customer Segment"]:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype("category").cat.codes

    X = df_clean[available]
    y = df_clean[target]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def main():
    X_train, X_test, y_train, y_test = load_data()
    client_id = os.environ.get("CLIENT_ID", "unknown")

    print(f"Training isolated Logistic Regression on {len(X_train)} samples...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(y_test, probs)

    print("\n========================================")
    print("  ISOLATED LOCAL BASELINE (Geographic)")
    print(f"  Client: {client_id}")
    print("========================================")
    for k, v in metrics.items():
        print(f"{k:10s}: {v:.4f}")
    print("========================================")

    payload = {
        "experiment_id": "08_geographic_local_baselines",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "local_baseline_lr_geographic",
        "client_id": client_id,
        "model": "Logistic Regression",
        "partitioning": "Geographic (Europe vs LATAM)",
        "non_iid_level": "mild",
        "results": {"local": {**metrics, "n_train": int(len(X_train))}},
    }
    out = os.environ.get("METRICS_PATH", f"metrics_{client_id}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[metrics] Wrote {out}")


if __name__ == "__main__":
    main()

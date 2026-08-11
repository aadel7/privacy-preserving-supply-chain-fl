import json
import os
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.simplefilter("ignore")
DEVICE = torch.device("cpu")


class SupplyChainMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze()


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
    print("Loading local client dataset...")
    df = pd.read_csv(data_path, encoding="latin1")
    target = "Late_delivery_risk"
    numeric_features = [
        "Days for shipment (scheduled)",
        "Order Item Quantity",
        "Order Item Discount",
        "Order Item Discount Rate",
        "Order Item Product Price",
        "Product Price",
        "Sales",
    ]
    categorical_features = [
        "Shipping Mode",
        "Market",
        "Order Region",
        "Customer Segment",
        "Category Name",
        "Type",
    ]
    columns_to_use = numeric_features + categorical_features + [target]
    df_clean = df[columns_to_use].dropna()
    df_encoded = pd.get_dummies(df_clean, columns=categorical_features, drop_first=True)
    X = df_encoded.drop(columns=[target]).values.astype(np.float32)
    y = df_encoded[target].values.astype(np.float32)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def main():
    X_train, X_test, y_train, y_test = load_data()
    input_dim = X_train.shape[1]
    client_id = os.environ.get("CLIENT_ID", "unknown")
    print(f"Training isolated Neural Network on {len(X_train)} samples...")
    print(f"Input dimension: {input_dim}")

    model = SupplyChainMLP(input_dim).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        ),
        batch_size=64,
        shuffle=True,
    )

    model.train()
    for _ in range(5):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.tensor(X_test, dtype=torch.float32))).numpy()
    metrics = classification_metrics(y_test, probs)

    print("\n========================================")
    print("  ISOLATED LOCAL BASELINE (Neural Net)")
    print("========================================")
    for k, v in metrics.items():
        print(f"{k:10s}: {v:.4f}")
    print("========================================")

    payload = {
        "experiment_id": "05_neural_network_local_baselines",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "local_baseline_nn",
        "client_id": client_id,
        "model": "MLP (64-32-1)",
        "results": {"local": {**metrics, "n_train": int(len(X_train))}},
    }
    out = os.environ.get("METRICS_PATH", f"metrics_{client_id}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[metrics] Wrote {out}")


if __name__ == "__main__":
    main()

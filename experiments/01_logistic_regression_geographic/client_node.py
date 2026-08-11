import os
import warnings

import numpy as np
import pandas as pd
import flwr as fl
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


class SupplyChainClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, X_test, y_train, y_test):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def get_parameters(self, config):
        return [self.model.coef_, self.model.intercept_]

    def fit(self, parameters, config):
        self.model.coef_ = parameters[0]
        self.model.intercept_ = parameters[1]
        self.model.fit(self.X_train, self.y_train)
        return self.get_parameters(config), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.model.coef_ = parameters[0]
        self.model.intercept_ = parameters[1]
        probs = self.model.predict_proba(self.X_test)[:, 1]
        metrics = classification_metrics(self.y_test, probs)
        return float(1.0 - metrics["accuracy"]), len(self.X_test), metrics


def load_data():
    data_path = "data.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    print("Loading local client dataset...")
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
    available_features = [c for c in feature_columns if c in df.columns]
    df_clean = df[available_features + [target]].dropna().copy()
    for col in ["Shipping Mode", "Market", "Order Region", "Customer Segment"]:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype("category").cat.codes
    X = df_clean[available_features]
    y = df_clean[target]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def main():
    X_train, X_test, y_train, y_test = load_data()
    model = LogisticRegression(max_iter=1000, warm_start=True)
    model.classes_ = np.array([0, 1])
    model.coef_ = np.zeros((1, X_train.shape[1]))
    model.intercept_ = np.zeros((1,))
    print("Connecting to central server for Federated Learning (Geographic)...")
    fl.client.start_numpy_client(
        server_address="central-server:8080",
        client=SupplyChainClient(model, X_train, X_test, y_train, y_test),
    )


if __name__ == "__main__":
    main()

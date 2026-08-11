import os
import warnings
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
import flwr as fl

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


class SupplyChainClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, X_test, y_test, local_epochs=5):
        self.model = model
        self.train_loader = train_loader
        self.X_test = torch.tensor(X_test, dtype=torch.float32)
        self.y_test = y_test
        self.local_epochs = local_epochs
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for _ in range(self.local_epochs):
            for X_batch, y_batch in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
        return self.get_parameters(config), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.X_test)
            probs = torch.sigmoid(logits).numpy()
        metrics = classification_metrics(self.y_test, probs)
        return float(1.0 - metrics["accuracy"]), len(self.y_test), metrics


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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        ),
        batch_size=64,
        shuffle=True,
    )
    return train_loader, X_test, y_test, X_train.shape[1]


def main():
    train_loader, X_test, y_test, input_dim = load_data()
    model = SupplyChainMLP(input_dim).to(DEVICE)
    print(f"Model input dimension: {input_dim}")
    print("Connecting to central server for Federated Learning...")
    fl.client.start_numpy_client(
        server_address="central-server:8080",
        client=SupplyChainClient(model, train_loader, X_test, y_test, local_epochs=5),
    )


if __name__ == "__main__":
    main()

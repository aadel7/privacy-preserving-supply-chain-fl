import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

warnings.simplefilter("ignore")
DEVICE = torch.device("cpu")

# --------------------------------------------------
# Same Neural Network architecture
# --------------------------------------------------
class SupplyChainMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze()


def load_combined_data():
    path1 = "data/client_1/data.csv"
    path2 = "data/client_2/data.csv"

    if not os.path.exists(path1) or not os.path.exists(path2):
        raise FileNotFoundError("Could not find data/client_1/data.csv or data/client_2/data.csv")

    print("Loading and combining both client datasets...")
    df1 = pd.read_csv(path1, encoding='latin1')
    df2 = pd.read_csv(path2, encoding='latin1')

    df = pd.concat([df1, df2], ignore_index=True)
    print(f"Combined dataset size: {len(df)} rows")

    target = 'Late_delivery_risk'

    numeric_features = [
        'Days for shipment (scheduled)',
        'Order Item Quantity',
        'Order Item Discount',
        'Order Item Discount Rate',
        'Order Item Product Price',
        'Product Price',
        'Sales'
    ]

    categorical_features = [
        'Shipping Mode',
        'Market',
        'Order Region',
        'Customer Segment',
        'Category Name',
        'Type'
    ]

    columns_to_use = numeric_features + categorical_features + [target]
    df_clean = df[columns_to_use].dropna()

    df_encoded = pd.get_dummies(df_clean, columns=categorical_features, drop_first=True)

    X = df_encoded.drop(columns=[target]).values.astype(np.float32)
    y = df_encoded[target].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


def main():
    X_train, X_test, y_train, y_test = load_combined_data()
    input_dim = X_train.shape[1]

    print(f"Training Centralized Neural Network on {len(X_train)} samples...")
    print(f"Input dimension: {input_dim}")

    model = SupplyChainMLP(input_dim).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    epochs = 5
    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(X_test, dtype=torch.float32)
        outputs = model(test_tensor)
        preds = (torch.sigmoid(outputs) > 0.5).numpy().astype(int)

    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print("\n========================================")
    print("  CENTRALIZED NEURAL NETWORK BASELINE")
    print("========================================")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("========================================")


if __name__ == "__main__":
    main()

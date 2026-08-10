# Experiment 3: Logistic Regression with Increased Local Epochs

**Commit:** 559024f  
**Date:** August 9, 2026

## Overview

This experiment investigates whether **increasing the number of local training epochs** can help mitigate the negative transfer problem observed in Experiment 2 (strong non-IID).

**Key Change:** Each client now performs **5 local epochs** per round instead of 1.

## Configuration

| Parameter | Value |
|-----------|-------|
| **Model** | Logistic Regression (sklearn) |
| **Framework** | Flower (FedAvg) |
| **Communication Rounds** | 3 |
| **Local Epochs per Round** | **5** (increased from 1) |
| **Data Partition** | Shipping Mode (strong non-IID) |
| **Non-IID Level** | Strong |

## Key Findings

| Metric | Exp 2 (1 epoch) | Exp 3 (5 epochs) | Change |
|--------|-----------------|------------------|--------|
| **Federated Accuracy** | 0.5508 | **0.5508** | No change |
| **Federated F1** | 0.6970 | **0.6970** | No change |
| **Loss** | 0.4492 | **0.4492** | No change |

**Interpretation:**  
Increasing local epochs from 1 to 5 produced **no improvement**. Results were identical to Experiment 2. Metrics remained completely flat across rounds.

This indicates that standard Logistic Regression converges very quickly on this dataset. Additional local optimization steps do not meaningfully change the final parameters and therefore provide no benefit under strong non-IID conditions.

## Why More Epochs Did Not Help

1. **Fast convergence of Logistic Regression:** The model reaches (near) optimum after a single `fit()` call on this data.
2. **Structural limitation:** The problem is not insufficient local training, but the inability of a linear model to represent heterogeneous decision boundaries.
3. **Averaging still dominates:** Even with more local steps, FedAvg still produces a compromise model that underperforms both local baselines.

## Comparison Across Experiments

| Experiment | Model | Epochs | Non-IID | Fed Acc | Notes |
|------------|-------|--------|---------|---------|-------|
| **1** | LR | 1 | Mild | 0.6929 | Geographic split |
| **2** | LR | 1 | Strong | 0.5508 | Negative transfer |
| **3** | LR | 5 | Strong | **0.5508** | **No improvement** |
| **4** | MLP | 5 | Strong | **0.6934** | Breakthrough |

**Key Insight:** Switching to a neural network (Exp 4) achieves far more than increasing epochs on Logistic Regression.

## How to Run

```bash
cd experiments/03_logistic_regression_increased_epochs/
docker compose up --build -d

# Federated training with 5 local epochs
docker compose restart central-server
docker compose exec client-1 python client_node.py
docker compose exec client-2 python client_node.py

docker compose logs central-server
```

## Code Change from Exp 2

```python
def fit(self, parameters, config):
    self.model.coef_ = parameters[0]
    self.model.intercept_ = parameters[1]

    local_epochs = 5  # <-- explicit multi-epoch loop

    for _ in range(local_epochs):
        self.model.fit(self.X_train, self.y_train)

    return self.get_parameters(config), len(self.X_train), {}
```

## Lessons Learned

1. More local training ≠ better federated performance under strong non-IID when using linear models.
2. Model capacity is the limiting factor, not the number of local iterations.
3. Need more expressive models (neural networks) to address this problem.

## Files in This Folder

```
03_logistic_regression_increased_epochs/
├── README.md
├── metrics.json
├── client_node.py
├── local_baseline.py
├── server/
│   ├── main.py
│   ├── Dockerfile
│   └── requirements.txt
├── docker-compose.yml
└── requirements.txt
```

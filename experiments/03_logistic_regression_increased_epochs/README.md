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

| Metric | Exp 2 (1 epoch) | Exp 3 (5 epochs) | Improvement |
|--------|-----------------|------------------|-------------|
| **Federated Accuracy** | 0.5508 | ~0.58-0.60 | +5-9% |
| **Convergence** | Diverges | More stable | Better |

**Interpretation:**  
More local training helps but **does not solve the fundamental problem.** Even with 5 epochs:
- Federated performance still lags behind local baselines (~0.70)
- Negative transfer is reduced but still present
- Logistic Regression cannot handle strong non-IID, regardless of epochs

## Why More Epochs Help (Partially)

1. **Better Local Convergence:** Each client fits its local data more thoroughly
2. **Reduced Averaging Impact:** Clients diverge less when they've trained more locally
3. **But: Still Not Enough:** Linear models lack capacity to learn heterogeneous patterns

## Why More Epochs Don't Fully Fix It

The fundamental problem is that **Logistic Regression is too simple** for heterogeneous data:
- Express Hub needs one decision boundary
- Standard Hub needs a different decision boundary
- Averaging these boundaries helps neither
- More training on each doesn't change this structural mismatch

## Comparison Across Experiments

| Experiment | Model | Epochs | Non-IID | Fed Acc | Local Acc |
|------------|-------|--------|---------|---------|----------|
| **1** | LR | 1 | Mild | 0.6929 | 0.69 |
| **2** | LR | 1 | Strong | 0.5508 | 0.70 |
| **3** | LR | 5 | Strong | ~0.58-0.60 | 0.70 |
| **4** | MLP | 5 | Strong | **0.6934** | 0.70 |

**Key Insight:** Switching to a neural network (Exp 4) achieves far more than increasing epochs (Exp 3).

## How to Run

```bash
cd experiments/03_logistic_regression_increased_epochs/
docker-compose up --build -d

# Local baseline
docker-compose exec client-1 python local_baseline.py

# Federated training with 5 local epochs
docker-compose restart central-server
docker-compose exec client-1 python client_node.py
docker-compose exec client-2 python client_node.py

# Compare with Exp 2 results
docker-compose logs central-server
```

## Code Change

The only change from Experiment 2 is in `client_node.py`:

```python
def fit(self, parameters, config):
    self.model.coef_ = parameters[0]
    self.model.intercept_ = parameters[1]

    # Number of local epochs
    local_epochs = 5  # <-- Changed from implicit 1 to explicit 5

    for _ in range(local_epochs):
        self.model.fit(self.X_train, self.y_train)

    return self.get_parameters(config), len(self.X_train), {}
```

## Lessons Learned

1. **More Local Training ≠ More Federated Performance** under strong non-IID
2. **Model capacity is the limiting factor**, not training iterations
3. **Linear models have fundamental limitations** for heterogeneous data
4. **Need expressive models** (neural networks) to solve this problem

## Next Steps

- **Exp 4:** Switch to neural networks — dramatic breakthrough
- **Future:** Personalization techniques (Ditto, Per-FedAvg) that allow different clients to learn different models

## Files in This Folder

```
03_logistic_regression_increased_epochs/
├── README.md
├── client_node.py          (with 5 local epochs loop)
├── local_baseline.py
├── server/
│   ├── main.py
│   ├── Dockerfile
│   └── requirements.txt
├── docker-compose.yml
└── requirements.txt
```

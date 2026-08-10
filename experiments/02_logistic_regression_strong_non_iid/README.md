# Experiment 2: Logistic Regression with Strong Non-IID (Shipping Mode Partition)

**Commit:** 80957ed / 8b8eae8  
**Date:** August 9, 2026

## Overview

This experiment introduces **strong non-IID data distribution** by partitioning data based on Shipping Mode:
- **Client 1 (Express Hub):** First Class + Same Day shipments
- **Client 2 (Standard Hub):** Second Class + Standard Class shipments

These two silos have fundamentally different delivery dynamics, creating severe data heterogeneity and revealing critical limitations of standard Federated Averaging with Logistic Regression.

## Configuration

| Parameter | Value |
|-----------|-------|
| **Model** | Logistic Regression (sklearn) |
| **Framework** | Flower (FedAvg) |
| **Communication Rounds** | 3 |
| **Data Partition** | Shipping Mode (Express vs Standard) |
| **Client 1 (Express)** | First Class + Same Day (~30,040 samples) |
| **Client 2 (Standard)** | Second Class + Standard Class (~114,374 samples) |
| **Non-IID Level** | **Strong** |

## Key Findings

**Negative Transfer Problem:**

| Metric | Local (Express) | Local (Standard) | Federated (3 rounds) |
|--------|-----------------|------------------|---------------------|
| **Accuracy** | **0.8364** | **0.6528** | **0.5508** |
| **F1 Score** | **0.8958** | **0.5196** | **0.6970** |

**Critical Observation:**  
Federated accuracy (0.5508) is **worse than both local baselines**. This is the classic **negative transfer** problem under strong non-IID conditions — standard FedAvg causes the clients' models to harm each other rather than cooperate. Metrics remained completely flat across all three rounds.

## Root Cause Analysis

1. **Heterogeneous Data Distributions:**
   - Express Hub: Fast deliveries with highly predictable patterns → strong local model
   - Standard Hub: More variable deliveries → weaker local model
   - Logistic Regression learns linear decision boundaries specific to each distribution

2. **Model Averaging Failure:**
   - When parameters are averaged, the resulting model is optimal for neither client
   - Both clients are pulled toward a suboptimal compromise
   - Logistic Regression lacks capacity to represent different decision boundaries simultaneously

## Comparison with Experiment 1

| Aspect | Exp 1 (Geographic) | Exp 2 (Shipping Mode) |
|--------|-------------------|---------------------|
| **Data Partition** | Europe vs LATAM | Express vs Standard |
| **Non-IID Level** | Mild | **Strong** |
| **Federated Acc** | 0.6929 | **0.5508** |
| **Local Baselines** | 0.6910 / 0.6948 | 0.8364 / 0.6528 |
| **Performance Gap** | ~0% | **Clear negative transfer** |
| **Conclusion** | Federated ≈ Local | **Federated << Local (strong client)** |

## How to Run

```bash
cd experiments/02_logistic_regression_strong_non_iid/
docker compose up --build -d

# Local baselines
docker compose exec client-1 python local_baseline.py
docker compose exec client-2 python local_baseline.py

# Federated training
docker compose restart central-server
docker compose exec client-1 python client_node.py
docker compose exec client-2 python client_node.py

# View results
docker compose logs central-server
```

## Key Insights

1. **Non-IID is a real challenge:** Mild partitioning (geographic) works fine. Strong partitioning (shipping mode) breaks standard FedAvg with linear models.
2. **Logistic Regression limitations:** Linear models cannot adapt well to heterogeneous data distributions.
3. **Need for better approaches:** See Experiment 4 (Neural Networks) for the breakthrough.

## Files in This Folder

```
02_logistic_regression_strong_non_iid/
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

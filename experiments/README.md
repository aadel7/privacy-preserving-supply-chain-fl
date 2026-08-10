# Experiments Archive

This directory contains snapshots of each major experimental stage in the privacy-preserving supply chain federated learning project.

Each subfolder is self-contained and captures the code, configuration, documentation, and structured metrics for a specific experimental phase.

## Overview

| # | Folder | Model | Partition | Non-IID | Rounds | Key Metric | Status |
|---|--------|-------|-----------|---------|--------|------------|--------|
| 1 | `01_logistic_regression_geographic/` | Logistic Reg | Geographic | Mild | 3 | Acc: 0.6929 | Initial baseline |
| 2 | `02_logistic_regression_strong_non_iid/` | Logistic Reg | Shipping Mode | Strong | 3 | Acc: 0.5508 | Negative transfer |
| 3 | `03_logistic_regression_increased_epochs/` | Logistic Reg | Shipping Mode | Strong | 3 | Acc: 0.5508 | No improvement |
| 4 | `04_neural_network_3rounds/` | MLP | Shipping Mode | Strong | 3 | Acc: 0.6934 | First NN federated |
| 5 | `05_neural_network_local_baselines/` | MLP | Shipping Mode | Strong | Local | C1: 0.8348, C2: 0.6552 | Local NN baselines |
| 6 | `06_centralized_neural_network/` | MLP | Combined | N/A | N/A | Acc: 0.6968 | Upper bound |
| 7 | `07_neural_network_5rounds/` | MLP | Shipping Mode | Strong | 5 | Acc: 0.6938 | Extended rounds |

## Full Comparison (Strong Non-IID Setting)

| Setting | Accuracy | F1 Score |
|---------|----------|----------|
| Local – Client 1 (Express Hub) | 0.8348 | 0.9041 |
| Local – Client 2 (Standard Hub) | 0.6552 | 0.5207 |
| Federated – Logistic Regression | 0.5508 | 0.6970 |
| **Federated – Neural Network** | **0.6934** | 0.6017 |
| **Centralized – Neural Network** | **0.6968** | **0.6600** |

## Detailed Experiment Descriptions

### 1. `01_logistic_regression_geographic/`
**Model:** Logistic Regression  
**Partition:** Geographic (Europe vs LATAM) — mild non-IID  
**Key Metrics:** Federated Acc 0.6929 / F1 0.6786  
**Finding:** Federated performance nearly identical to local baselines. Geographic split does not create significant heterogeneity.

### 2. `02_logistic_regression_strong_non_iid/`
**Model:** Logistic Regression  
**Partition:** Shipping Mode (Express vs Standard) — strong non-IID  
**Key Metrics:** Federated Acc 0.5508 / F1 0.6970  
**Local:** Express 0.8364 / 0.8958 ; Standard 0.6528 / 0.5196  
**Finding:** Clear negative transfer. Federated accuracy worse than both local models.

### 3. `03_logistic_regression_increased_epochs/`
**Model:** Logistic Regression with 5 local epochs  
**Partition:** Same strong non-IID as Exp 2  
**Key Metrics:** Federated Acc 0.5508 / F1 0.6970 (identical to Exp 2)  
**Finding:** Increasing local epochs produced **no change**. Linear models converge quickly; more local steps do not help.

### 4. `04_neural_network_3rounds/`
**Model:** Small MLP (64→32→1)  
**Partition:** Strong non-IID (Shipping Mode)  
**Key Metrics:** Federated Acc 0.6934 / F1 0.6017  
**Finding:** Breakthrough. NN substantially outperforms LR under the same conditions and shows real improvement across rounds.

### 5. `05_neural_network_local_baselines/`
**Model:** Same MLP, trained locally  
**Results:** Client 1 (Express) Acc 0.8348 / F1 0.9041 ; Client 2 (Standard) Acc 0.6552 / F1 0.5207  
**Finding:** Large performance gap confirms strong non-IID. Federated NN sits between the two local models.

### 6. `06_centralized_neural_network/`
**Model:** Same MLP trained on combined data  
**Key Metrics:** Acc 0.6968 / F1 0.6600  
**Finding:** Theoretical upper bound. Federated NN (0.6934) is nearly identical → strong support for “privacy without penalty”.

### 7. `07_neural_network_5rounds/`
**Model:** Same MLP, 5 communication rounds  
**Key Metrics:** Final Acc 0.6938 / F1 0.6021  
**Finding:** Most gains occur in the first 2–3 rounds. Extending to 5 rounds yields no meaningful additional improvement.

## Folder Structure

Each experiment folder typically contains:

```
experiments/NN_experiment_name/
├── README.md
├── metrics.json              # Structured results (machine-readable)
├── client_node.py            # Federated client (if applicable)
├── local_baseline*.py        # Local baseline (if applicable)
├── centralized_baseline_nn.py # Centralized baseline (if applicable)
├── server/
│   ├── main.py
│   ├── Dockerfile
│   └── requirements.txt
├── docker-compose.yml
└── requirements.txt
```

## Key Insights Across All Experiments

1. **Geographic partitioning produces only mild non-IID.** Local and federated performance are nearly the same.
2. **Shipping Mode partitioning creates strong non-IID.** Clear gap between Express and Standard hubs.
3. **Standard FedAvg + Logistic Regression suffers severe negative transfer** under strong non-IID (Acc drops to 0.55).
4. **Increasing local epochs does not fix this** for linear models.
5. **A small Neural Network largely recovers performance** (Acc 0.6934) and comes very close to the centralized upper bound (0.6968).
6. **Three communication rounds are sufficient** for this architecture and dataset.

## Methodological Notes

- Target leakage was strictly eliminated (only pre-event features used).
- Categorical features are one-hot encoded.
- Server-side weighted aggregation is used for Accuracy and F1.
- All experiments are containerized to guarantee data isolation between silos.

## Reproducibility

- Dataset: DataCo Smart Supply Chain (Kaggle)
- Fixed random seeds where applicable (`random_state=42`)
- Dependencies listed per experiment
- Structured metrics available in each folder’s `metrics.json`

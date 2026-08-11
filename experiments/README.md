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

## Automatic Metrics Export

After each run, experiments write results to **`metrics.json`** (dynamically, from the actual training run — not hardcoded).

| Experiment type | Who writes metrics | Output file |
|-----------------|--------------------|-------------|
| Federated (01–04, 07) | Central server (Flower History) | `metrics.json` |
| Local baselines (05) | Each client script | `metrics_client_1.json`, `metrics_client_2.json` |
| Centralized (06) | Baseline script | `metrics.json` |

Federated metrics include per-round accuracy, F1, and loss when available. Look for `[metrics] Wrote ...` in the server/client logs.

## Analysis Notebook (Charts)

Charts and summary tables are generated from the metrics files:

```bash
cd experiments/analysis
pip install -r requirements.txt
jupyter notebook experiment_analysis.ipynb
```

The notebook:

1. Loads all `experiments/*/metrics*.json` files
2. Builds a results table
3. Generates comparison bar charts and learning curves
4. Saves figures under `figures/` and a CSV summary

See `analysis/README.md` for details.

## How to Run an Isolated Experiment

```bash
cd experiments/04_neural_network_3rounds

# Remove any accidental local data.csv/ folders from failed mounts
rm -rf data.csv data

docker compose down
docker compose up --build -d

# Federated run (two terminals)
docker compose exec client-1 python client_node.py
docker compose exec client-2 python client_node.py

# Results
docker compose logs central-server
cat metrics.json
```

**Note:** Free port 8080 first if another stack is using it (`docker compose down` in the repo root).

## Folder Structure

```
experiments/
├── README.md
├── analysis/
│   ├── experiment_analysis.ipynb
│   ├── requirements.txt
│   └── README.md
├── 01_logistic_regression_geographic/
│   ├── README.md
│   ├── metrics.json          # written/updated by the run
│   ├── client_node.py
│   ├── server/
│   ├── docker-compose.yml
│   ├── Dockerfile
│   └── requirements.txt
├── ...
└── 07_neural_network_5rounds/
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
- `.dockerignore` excludes accidental `data/` / `data.csv/` folders from image builds.

## Reproducibility

- Dataset: DataCo Smart Supply Chain (Kaggle)
- Fixed random seeds where applicable (`random_state=42`)
- Dependencies listed per experiment
- Metrics are exported automatically from each run into `metrics.json`

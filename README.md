# Privacy-Preserving Predictive Analytics in Supply Chains: A Federated Learning Architecture

This repository contains the simulation environment and source code for an MSc Computer Science project evaluating Federated Learning (FL) for predictive analytics across decentralized supply chain data silos.

## Project Overview

Modern supply chains generate valuable operational data that remains siloed due to privacy regulations and competitive concerns. This project investigates whether Federated Learning can enable collaborative prediction of late delivery risk (`Late_delivery_risk`) without requiring participants to share raw proprietary data.

**Key characteristics of the current implementation:**
- Framework: Flower (FedAvg)
- Models evaluated: Logistic Regression and a small Multi-Layer Perceptron (Neural Network)
- Dataset: DataCo Smart Supply Chain
- Partitioning strategies: Geographic (mild non-IID) and Shipping Mode (strong non-IID)
- Strict removal of target leakage (only pre-event features are used)
- Isolated experiment archive under `experiments/` with automatic metrics export and analysis notebook

## Prerequisites

- Docker & Docker Compose
- Python 3.10+
- Kaggle API token (`kaggle.json`) configured in `~/.kaggle/`

## 1. Data Acquisition & Partitioning

```bash
python -m kaggle datasets download -d shashwatwork/dataco-smart-supply-chain-for-big-data-analysis -p data/ --unzip
python data/partition_data.py
```

Silos:
- **Client 1 – Express Hub**: First Class + Same Day
- **Client 2 – Standard Hub**: Second Class + Standard Class

## 2. Main Infrastructure (latest code)

```bash
docker compose up --build -d
```

## 3. Running Experiments

### Single experiment runner

```bash
python scripts/run_experiment.py 04_neural_network_3rounds
python scripts/run_experiment.py 05_neural_network_local_baselines
python scripts/run_experiment.py 06_centralized_neural_network
```

### Run all (or a subset) sequentially

```bash
# Full batch (slow)
python scripts/run_all_experiments.py

# Selected only
python scripts/run_all_experiments.py --only 04,05,06

# Skip some
python scripts/run_all_experiments.py --skip 01,02

# Stop on first failure
python scripts/run_all_experiments.py --stop-on-error
```

See `scripts/README.md` for details.

### Manual (latest stack)

```bash
docker compose exec client-1 python local_baseline_nn.py
docker compose restart central-server
docker compose exec client-1 python client_node.py
docker compose exec client-2 python client_node.py
python centralized_baseline_nn.py
```

### Automatic metrics + analysis notebook

Each run writes `metrics.json` (or client-specific metrics). Then:

```bash
cd experiments/analysis
pip install -r requirements.txt
jupyter notebook experiment_analysis.ipynb
```

## 4. Tests

```bash
pip install -r tests/requirements.txt
python -m pytest tests/ -v
```

## 5. Key Results (Strong Non-IID)

| Setting | Accuracy | F1 Score |
|---------|----------|----------|
| Local – Client 1 (Express) | 0.8348 | 0.9041 |
| Local – Client 2 (Standard) | 0.6552 | 0.5207 |
| Federated – Logistic Regression | 0.5508 | 0.6970 |
| **Federated – Neural Network** | **0.6934** | 0.6017 |
| **Centralized – Neural Network** | **0.6968** | **0.6600** |

Main finding: under strong non-IID, FedAvg + LR shows negative transfer; a small NN recovers performance close to the centralized upper bound without sharing raw data.

## 6. Repository Layout

```text
.
├── clients/
├── server/
├── data/
├── experiments/       # isolated snapshots + analysis notebook
├── scripts/           # run_experiment.py, run_all_experiments.py
├── tests/
├── figures/
└── README.md
```

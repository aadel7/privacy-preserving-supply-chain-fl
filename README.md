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
# Download the dataset
python -m kaggle datasets download -d shashwatwork/dataco-smart-supply-chain-for-big-data-analysis -p data/ --unzip

# Create strong non-IID partitions based on Shipping Mode
python data/partition_data.py
```

The current partitioning creates two realistic logistics silos:
- **Client 1 – Express Hub**: First Class + Same Day shipments
- **Client 2 – Standard Hub**: Second Class + Standard Class shipments

## 2. Main Infrastructure (latest code)

```bash
docker compose up --build -d
```

This starts one central aggregation server and two isolated client nodes.

## 3. Running the Latest Code

### Local Baselines

```bash
docker compose exec client-1 python local_baseline.py
docker compose exec client-2 python local_baseline.py
docker compose exec client-1 python local_baseline_nn.py
docker compose exec client-2 python local_baseline_nn.py
```

### Federated Training

```bash
docker compose restart central-server
docker compose exec client-1 python client_node.py
docker compose exec client-2 python client_node.py
docker compose logs central-server
```

### Centralized Baseline

```bash
python centralized_baseline_nn.py
```

## 4. Isolated Experiments Archive

Historical and self-contained experiment snapshots live under **`experiments/`** (see that folder’s README).

### Single-experiment runner (recommended)

From the repo root:

```bash
python scripts/run_experiment.py 04_neural_network_3rounds
python scripts/run_experiment.py 05_neural_network_local_baselines
python scripts/run_experiment.py 06_centralized_neural_network
```

This builds containers (when needed), runs clients, waits for `metrics.json`, prints results, and tears down. See `scripts/README.md`.

Manual equivalent:

```bash
cd experiments/04_neural_network_3rounds
docker compose up --build -d
docker compose exec client-1 python client_node.py
docker compose exec client-2 python client_node.py
cat metrics.json
```

### Automatic metrics export

Federated runs write **`metrics.json`** from the Flower server history (accuracy, F1, loss per round). Local and centralized baselines also export metrics files. These are generated from the run, not hardcoded.

### Analysis notebook (charts)

```bash
cd experiments/analysis
pip install -r requirements.txt
jupyter notebook experiment_analysis.ipynb
```

The notebook loads all `metrics.json` files and generates comparison charts and learning curves into `figures/`.

## 5. Tests

Lightweight checks (structure, metrics schema, aggregation, MLP shapes). Does not start Docker or full FL training.

```bash
pip install -r tests/requirements.txt
python -m pytest tests/ -v
```

See `tests/README.md` for details.

## 6. Key Experimental Results

### Strong Non-IID Setting (Shipping Mode Partition)

| Setting                              | Accuracy | F1 Score | Notes |
|--------------------------------------|----------|----------|-------|
| Local – Client 1 (Express Hub)       | 0.8348   | 0.9041   | Strong local performance |
| Local – Client 2 (Standard Hub)      | 0.6552   | 0.5207   | Weaker local performance |
| Federated – Logistic Regression      | 0.5508   | 0.6970   | Severe negative transfer |
| **Federated – Neural Network**       | **0.6934** | 0.6017 | Significant improvement over LR |
| **Centralized – Neural Network**     | **0.6968** | **0.6600** | Theoretical upper bound |

### Main Findings

1. **Target leakage was successfully eliminated.** Early experiments with post-event features (~98% accuracy) were discarded.
2. **Geographic partitioning is only mildly non-IID.** Local and federated performance are nearly identical.
3. **Shipping Mode partitioning creates strong non-IID.** Clear gap between Express and Standard hubs.
4. **FedAvg + Logistic Regression shows negative transfer** under strong non-IID (Accuracy 0.55).
5. **A small Neural Network largely recovers performance** (0.6934), close to the centralized upper bound (0.6968).
6. Federated Learning can approach centralized accuracy while keeping raw data local.

## 7. Methodological Notes

- Only pre-event features are used (no target leakage).
- Categorical variables are one-hot encoded.
- Server-side weighted aggregation for Accuracy and F1.
- Experiments are containerized for data isolation between silos.

## 8. Repository Layout

```text
.
├── clients/
├── server/
├── data/
├── experiments/             # Isolated experiment archive
│   ├── analysis/            # Jupyter notebook for charts
│   └── ...
├── scripts/                 # run_experiment.py
├── tests/
├── figures/
├── Experiment_Log.md
└── README.md
```

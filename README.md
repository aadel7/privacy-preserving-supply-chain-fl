# Privacy-Preserving Predictive Analytics in Supply Chains: A Federated Learning Architecture

This repository contains the simulation environment and source code for an MSc Computer Science project. The goal is to evaluate whether Federated Learning can enable collaborative predictive analytics across decentralized supply chain entities **without sharing raw proprietary data**.

## Project Overview

Modern supply chains generate valuable operational data that remains siloed due to privacy regulations and competitive concerns. This project implements a Federated Learning architecture that allows multiple parties to collaboratively train a model for predicting late delivery risk (`Late_delivery_risk`) while keeping their data local.

**Current implementation characteristics:**
- Framework: Flower (FedAvg strategy)
- Model: Logistic Regression
- Dataset: DataCo Smart Supply Chain
- Data partitioning: Geographic (Europe vs LATAM)
- Strict elimination of target leakage

## Prerequisites

- Docker & Docker Compose
- Python 3.10+
- Kaggle API token (`kaggle.json`) configured in `~/.kaggle/`

## 1. Data Acquisition & Partitioning

```bash
# Download the dataset
python -m kaggle datasets download -d shashwatwork/dataco-smart-supply-chain-for-big-data-analysis -p data/ --unzip

# Partition the data into regional silos (Europe and LATAM)
python data/partition_data.py
```

## 2. Infrastructure Spin-Up

```bash
docker compose up --build -d
```

This starts:
- 1 Central aggregation server
- 2 Isolated client nodes (data silos)

## 3. Running Experiments

### A. Local Baselines (Isolated Training)

Train a model on a single silo without federation:

```bash
# Client 1 – Europe silo
docker compose exec client-1 python local_baseline.py

# Client 2 – LATAM silo
docker compose exec client-2 python local_baseline.py
```

### B. Federated Training

1. Restart the central server:
```bash
docker compose restart central-server
```

2. Launch both clients (use separate terminals):
```bash
docker compose exec client-1 python client_node.py
docker compose exec client-2 python client_node.py
```

3. Inspect the results:
```bash
docker compose logs central-server
```

## 4. Current Experimental Results

| Experiment                          | Scope                        | Accuracy | F1 Score | Notes |
|-------------------------------------|------------------------------|----------|----------|-------|
| Local Baseline – Client 1           | Europe only (n ≈ 40,201)     | 0.6910   | 0.6810   | Isolated regional performance |
| Local Baseline – Client 2           | LATAM only (n ≈ 41,275)      | 0.6948   | 0.6763   | Isolated regional performance |
| Federated Model (3 rounds)          | Europe + LATAM (2 silos)     | 0.6929   | 0.6786   | Nearly identical to local baselines. Metrics remained flat across rounds. |
| Early experiment (with leakage)     | 2 silos                      | ~0.98    | ~0.98    | Discarded due to severe target leakage |

**Interpretation:**  
With the current geographic partitioning and Logistic Regression, the two data silos exhibit very similar distributions. Federated Averaging currently provides neither significant performance benefit nor penalty compared to training on isolated local data.

## 5. Methodological Notes

- All post-event features (e.g. `Days for shipping (real)`, `Delivery Status`) were removed to eliminate target leakage.
- Only information available at the time of order placement is used for prediction.
- Categorical features are one-hot encoded.
- Server-side weighted aggregation is used for Accuracy and F1-Score.

## 6. Next Steps

- Design stronger non-IID data partitions
- Increase the number of simulated clients
- Experiment with a higher number of local training epochs
- Evaluate more expressive models (e.g. small neural networks)
- Analyse communication overhead and convergence behaviour under different conditions
```
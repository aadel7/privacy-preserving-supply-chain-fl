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

## 2. Infrastructure Spin-Up

```bash
docker compose up --build -d
```

This starts one central aggregation server and two isolated client nodes.

## 3. Running Experiments

### Local Baselines

```bash
# Logistic Regression
docker compose exec client-1 python local_baseline.py
docker compose exec client-2 python local_baseline.py

# Neural Network
docker compose exec client-1 python local_baseline_nn.py
docker compose exec client-2 python local_baseline_nn.py
```

### Federated Training

```bash
docker compose restart central-server
```

Then in two separate terminals:

```bash
docker compose exec client-1 python client_node.py
docker compose exec client-2 python client_node.py
```

View results:

```bash
docker compose logs central-server
```

### Centralized Baseline (Upper Bound)

```bash
python centralized_baseline_nn.py
```

## 4. Key Experimental Results

### Strong Non-IID Setting (Shipping Mode Partition)

| Setting                              | Accuracy | F1 Score | Notes |
|--------------------------------------|----------|----------|-------|
| Local – Client 1 (Express Hub)       | 0.8348   | 0.9041   | Strong local performance |
| Local – Client 2 (Standard Hub)      | 0.6552   | 0.5207   | Weaker local performance |
| Federated – Logistic Regression      | 0.5508   | 0.6970   | Severe negative transfer |
| **Federated – Neural Network**       | **0.6934** | 0.6017 | Significant improvement over LR |
| **Centralized – Neural Network**     | **0.6968** | **0.6600** | Theoretical upper bound |

### Main Findings

1. **Target leakage was successfully eliminated.** Early experiments that included post-event features produced unrealistically high accuracy (~98%) and were discarded.
2. **Geographic partitioning produced only mild non-IID conditions.** Local and federated performance were nearly identical.
3. **Shipping Mode partitioning creates strong non-IID conditions.** A clear performance gap appears between the Express and Standard hubs.
4. **Standard FedAvg with Logistic Regression suffers from negative transfer** under strong non-IID (Accuracy drops to 0.55).
5. **A small Neural Network significantly mitigates this problem**, achieving Accuracy (0.6934) that is nearly identical to the centralized upper bound (0.6968).
6. Federated Learning with a Neural Network can recover almost the same predictive performance as centralized training while keeping all raw data local.

## 5. Methodological Notes

- Only features available at the time of order placement are used (strict pre-event feature set).
- Categorical variables are one-hot encoded.
- Server-side weighted aggregation is used for Accuracy and F1-Score.
- All experiments are fully containerized to guarantee data isolation between silos.

## 6. Next Steps

- Increase the number of communication rounds
- Experiment with higher numbers of local epochs
- Evaluate alternative aggregation strategies (e.g. FedProx)
- Scale to a larger number of clients
- Analyse communication overhead and convergence behaviour in more detail
```

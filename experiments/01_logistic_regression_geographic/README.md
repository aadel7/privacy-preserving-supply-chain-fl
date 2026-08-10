# Experiment 1: Logistic Regression with Geographic Partitioning

**Commit:** 7584328 (Initial commit)  
**Date:** August 9, 2026  
**Branch/Tag:** Initial baseline

## Overview

This is the initial experimental baseline establishing the federated learning infrastructure with logistic regression on geographically partitioned data (Europe vs LATAM). This experiment demonstrates a mild non-IID scenario where data distributions are relatively similar across silos.

## Configuration

| Parameter | Value |
|-----------|-------|
| **Model** | Logistic Regression (sklearn) |
| **Framework** | Flower (FedAvg) |
| **Communication Rounds** | 3 |
| **Data Partition** | Geographic (Europe, LATAM) |
| **Client 1 (Europe)** | ~40,201 samples |
| **Client 2 (LATAM)** | ~41,275 samples |
| **Non-IID Level** | Mild (similar distributions) |

## Key Findings

| Metric | Local (Europe) | Local (LATAM) | Federated (3 rounds) |
|--------|----------------|---------------|---------------------|
| **Accuracy** | 0.6910 | 0.6948 | 0.6929 |
| **F1 Score** | 0.6810 | 0.6763 | 0.6786 |

**Interpretation:**  
Federated performance is nearly identical to local baselines. This indicates that geographic partitioning does not create significant data heterogeneity. The two regions have similar shipping and delivery patterns, making federated aggregation straightforward.

## Features Used

**Numeric:**
- Days for shipment (scheduled)
- Order Item Quantity
- Sales
- Product Price

**Categorical (one-hot encoded):**
- Shipping Mode
- Market
- Order Region
- Customer Segment

## Model Details

- **Algorithm:** LogisticRegression with warm_start=True
- **Max iterations:** 1000
- **Loss function:** Binary cross-entropy
- **Regularization:** Default (L2)

## Infrastructure

- **Central Server:** Aggregates model parameters (FedAvg strategy)
- **Client 1:** Europe data silo
- **Client 2:** LATAM data silo
- **Data isolation:** Complete (via Docker volumes)

## How to Run

### Prerequisites

```bash
# Ensure you have Docker and Docker Compose installed
docker --version
docker-compose --version
```

### 1. Download Data

```bash
python -m kaggle datasets download -d shashwatwork/dataco-smart-supply-chain-for-big-data-analysis -p data/ --unzip
```

Ensure your Kaggle API token is configured at `~/.kaggle/kaggle.json`.

### 2. Partition Data

```bash
# Navigate to this experiment folder
cd experiments/01_logistic_regression_geographic/

# Run the partitioning script (if data/ folder exists)
python data/partition_data.py
```

This creates:
- `data/client_1/data.csv` (Europe data)
- `data/client_2/data.csv` (LATAM data)

### 3. Start Infrastructure

```bash
docker-compose up --build -d
```

### 4. Run Local Baselines

```bash
# Terminal 1: Europe baseline
docker-compose exec client-1 python local_baseline.py

# Terminal 2: LATAM baseline
docker-compose exec client-2 python local_baseline.py
```

Expected output:
```
========================================
  ISOLATED LOCAL BASELINE (SINGLE SILO)
========================================
Accuracy : 0.6910
F1 Score : 0.6810
========================================
```

### 5. Run Federated Training

```bash
# Restart server
docker-compose restart central-server

# Terminal 2: Client 1
docker-compose exec client-1 python client_node.py

# Terminal 3: Client 2
docker-compose exec client-2 python client_node.py
```

### 6. View Results

```bash
docker-compose logs central-server
```

Look for output like:
```
Round 1: federated_accuracy=0.6910, federated_f1_score=0.6850
Round 2: federated_accuracy=0.6920, federated_f1_score=0.6875
Round 3: federated_accuracy=0.6929, federated_f1_score=0.6786
```

## Important Notes

1. **Target Leakage Prevention:**  
   This experiment strictly removes post-event features. Only information available at order placement time is used.

2. **Mild Non-IID:**  
   Europe and LATAM have similar delivery dynamics. This is intentional for the first baseline—see Experiment 2 for strong non-IID.

3. **Comparison with Exp 2:**  
   Federated (0.6929) ≈ Local (0.6910, 0.6948)  
   In Experiment 2 (Shipping Mode partition), federated accuracy drops to 0.5508, revealing the non-IID problem.

## Files in This Folder

```
01_logistic_regression_geographic/
├── README.md                    # This file
├── client_node.py              # Federated client implementation
├── local_baseline.py           # Local training baseline
├── server/
│   ├── main.py                # Server aggregation (3 rounds)
│   ├── Dockerfile             # Docker image for server
│   └── requirements.txt        # Server dependencies
├── docker-compose.yml          # Infrastructure definition
└── requirements.txt            # Client dependencies
```

## Dependencies

**Client-side:**
- pandas
- scikit-learn
- numpy
- flwr (Flower)

**Server-side:**
- flwr
- fastapi
- uvicorn

## Lessons Learned

1. **Geographic partitioning is too homogeneous** for demonstrating FL benefits under non-IID conditions.
2. **Federated averaging works well** when data distributions are similar across clients.
3. **Need stronger non-IID scenarios** to stress-test the FL approach (see Experiment 2).
4. **Logistic Regression is sufficient** for mild non-IID but breaks down under strong heterogeneity.

## Next Experiments

- **Exp 2:** Introduce strong non-IID via Shipping Mode partition
- **Exp 3:** Increase local epochs for better LR convergence
- **Exp 4:** Switch to neural networks to handle non-IID better

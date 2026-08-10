# Experiments Archive

This directory contains snapshots of each major experimental stage in the privacy-preserving supply chain federated learning project.

## Overview

Each subfolder is self-contained and captures the code, configuration, and documentation for a specific experimental phase. This allows for reproducibility and independent analysis of each stage.

## Experiments at a Glance

| # | Folder | Model | Partition | Non-IID | Rounds | Key Metric | Status |
|---|--------|-------|-----------|---------|--------|------------|--------|
| 1 | `01_logistic_regression_geographic/` | Logistic Reg | Geographic | Mild | 3 | Acc: 0.69 | Initial baseline |
| 2 | `02_logistic_regression_strong_non_iid/` | Logistic Reg | Shipping Mode | Strong | 3 | Acc: 0.55 | Non-IID challenge |
| 3 | `03_logistic_regression_increased_epochs/` | Logistic Reg | Shipping Mode | Strong | 3 | Improved | More local training |
| 4 | `04_neural_network_3rounds/` | MLP | Shipping Mode | Strong | 3 | Acc: 0.69 | First NN federated |
| 5 | `05_neural_network_local_baselines/` | MLP | Shipping Mode | Strong | Local | C1: 0.83, C2: 0.66 | Local NN baselines |
| 6 | `06_centralized_neural_network/` | MLP | Combined | N/A | N/A | Acc: 0.70 | Upper bound |
| 7 | `07_neural_network_5rounds/` | MLP | Shipping Mode | Strong | 5 | Acc: 0.69 | Extended rounds |

## Detailed Experiment Descriptions

### 1. `01_logistic_regression_geographic/`
**Commit:** 7584328 (Initial commit)  
**Date:** August 9, 2026  
**Model:** Logistic Regression with 3 communication rounds  
**Data Partition:** Geographic (Europe vs LATAM)  
**Non-IID Level:** Mild (similar distributions across regions)  
**Key Metrics:** Accuracy ~0.69, F1 ~0.68  
**Purpose:** Establish baseline with target leakage eliminated; geographic partitioning provides baseline IID/mild non-IID scenario  
**Key Finding:** Geographic split produces nearly identical federated and local performance

### 2. `02_logistic_regression_strong_non_iid/`
**Commit:** 80957ed / 8b8eae8  
**Date:** August 9, 2026  
**Model:** Logistic Regression with 3 communication rounds  
**Data Partition:** Shipping Mode (Express Hub vs Standard Hub)  
**Non-IID Level:** Strong (heterogeneous shipping patterns)  
**Key Metrics:** Accuracy ~0.55, F1 ~0.70  
**Purpose:** Demonstrate the non-IID challenge; strong negative transfer with standard FedAvg  
**Key Finding:** Logistic Regression fails under strong non-IID conditions; federated performance degrades significantly

### 3. `03_logistic_regression_increased_epochs/`
**Commit:** 559024f  
**Date:** August 9, 2026  
**Model:** Logistic Regression with 5 local epochs (increased from 1)  
**Data Partition:** Shipping Mode  
**Non-IID Level:** Strong  
**Key Metrics:** Improved convergence behavior  
**Purpose:** Investigate effect of more local training iterations on non-IID performance  
**Key Finding:** Additional local epochs help but cannot overcome fundamental limitations of LR under strong non-IID

### 4. `04_neural_network_3rounds/`
**Commit:** df55219  
**Date:** August 9, 2026  
**Model:** Small Multi-Layer Perceptron (64→32→1 with ReLU and Dropout)  
**Data Partition:** Shipping Mode  
**Non-IID Level:** Strong  
**Framework:** PyTorch + Flower (FedAvg)  
**Key Metrics:** Accuracy ~0.69, F1 ~0.60  
**Purpose:** Test if more expressive models can handle strong non-IID better than LR  
**Key Finding:** Dramatic improvement! NN achieves federated accuracy of 0.69 vs LR's 0.55 under same conditions

### 5. `05_neural_network_local_baselines/`
**Commit:** bd942f3  
**Date:** August 9, 2026  
**Model:** Same MLP architecture, trained independently on each silo  
**Data Partition:** Shipping Mode  
**Non-IID Level:** Strong  
**Local Results:** 
  - Client 1 (Express): Accuracy ~0.83, F1 ~0.90
  - Client 2 (Standard): Accuracy ~0.66, F1 ~0.52  
**Purpose:** Establish local baseline for federated comparison; understand per-client performance gap  
**Key Finding:** Express hub has much stronger predictability than Standard hub; large heterogeneity confirmed

### 6. `06_centralized_neural_network/`
**Commit:** cc2f603  
**Date:** August 9, 2026  
**Model:** Same MLP architecture trained on combined dataset  
**Data Partition:** Combined (no silos; centralized training)  
**Training Epochs:** 5  
**Key Metrics:** Accuracy ~0.70, F1 ~0.66  
**Purpose:** Establish theoretical upper bound (best possible performance without privacy constraints)  
**Key Finding:** Federated NN (0.69 acc) nearly matches centralized upper bound (0.70 acc) — privacy-preserving approach succeeds!

### 7. `07_neural_network_5rounds/`
**Commit:** 6d43b54  
**Date:** August 9, 2026  
**Model:** Same MLP architecture with extended communication  
**Data Partition:** Shipping Mode  
**Non-IID Level:** Strong  
**Communication Rounds:** 5 (extended from 3)  
**Key Metrics:** Accuracy ~0.69, F1 ~0.60  
**Purpose:** Test convergence behavior with more communication rounds  
**Key Finding:** More rounds stabilize learning but show diminishing returns; model converges early

## Folder Structure

Each experiment folder follows this structure:

```
experiments/NN_experiment_name/
├── README.md                 # Experiment-specific details and how to run
├── client_node.py           # Federated client script (if applicable)
├── local_baseline.py        # Local training baseline (if applicable)
├── local_baseline_nn.py     # Local NN baseline (if applicable)
├── centralized_baseline_nn.py # Centralized NN baseline (if applicable)
├── server/
│   ├── main.py             # Server aggregation logic
│   ├── Dockerfile          # Docker configuration
│   └── requirements.txt    # Server dependencies
├── docker-compose.yml       # Infrastructure definition
└── requirements.txt         # Client dependencies
```

## How to Use an Experiment

### 1. Navigate to an experiment
```bash
cd experiments/04_neural_network_3rounds/
```

### 2. Read the experiment README
```bash
cat README.md
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up infrastructure (if using Docker)
```bash
docker compose up --build -d
```

### 5. Run the experiment

**For federated training:**
```bash
# Terminal 1: Start server
docker compose exec central-server python server/main.py

# Terminal 2 & 3: Start clients
docker compose exec client-1 python client_node.py
docker compose exec client-2 python client_node.py
```

**For local baseline:**
```bash
docker compose exec client-1 python local_baseline.py
```

## Key Insights Across All Experiments

### 1. Geographic vs Shipping Mode Partitioning
- **Geographic:** Produces mild non-IID conditions; federated and local performance similar
- **Shipping Mode:** Creates strong non-IID conditions; clear performance gap between silos

### 2. Logistic Regression Limitations (Exp 1-3)
- Under strong non-IID: Severe negative transfer (Acc ~0.55)
- Cannot capture complex, heterogeneous data relationships
- Limited improvement even with more local epochs

### 3. Neural Networks as Solution (Exp 4-7)
- Small MLP dramatically outperforms LR under strong non-IID
- Federated NN (0.69) ≈ Centralized NN (0.70)
- **Proves privacy-preserving federated approach can maintain accuracy**

### 4. Convergence Behavior
- Increased local epochs (Exp 3) help but don't solve LR limitations
- More communication rounds (Exp 7) stabilize but show diminishing returns
- Models converge within 3-5 rounds

### 5. Non-IID Challenge
- Strong non-IID (Shipping Mode) reveals fundamental FL challenges
- Client 1 has much better local performance (0.83) than Client 2 (0.66)
- Federated learning averages across this gap, achieving middle ground

## Methodological Notes

- **Target Leakage:** Strictly eliminated; only pre-event features used
- **Feature Set:** Consistent across experiments (numeric + categorical)
- **Encoding:** One-hot encoding for categorical variables
- **Data Splitting:** 80/20 train-test with stratification, random_state=42
- **Aggregation:** Server-side weighted average (FedAvg strategy)
- **Loss Function:** BCEWithLogitsLoss for neural networks
- **Optimization:** Adam optimizer (lr=0.001)

## Reproducibility

- All experiments use DataCo Smart Supply Chain dataset (Kaggle)
- Docker ensures identical environment isolation
- Fixed random seeds (random_state=42)
- All dependencies pinned in requirements.txt
- Code is deterministic (no randomness in inference)

## Next Steps & Future Directions

Based on these experiments, promising directions include:

1. **More advanced aggregation strategies** (FedProx, FedNova) for non-IID data
2. **Personalization techniques** (Ditto, Per-FedAvg) to handle heterogeneous silos
3. **Differential privacy** to add formal privacy guarantees
4. **Communication efficiency** via compression and quantization
5. **Scaling** to larger numbers of clients and federated data sources
6. **Alternative architectures** (attention, transformer) for complex patterns

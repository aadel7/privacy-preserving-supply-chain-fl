# Experiment 4: Neural Network with 3 Rounds (First NN Federated Run)

**Commit:** df55219  
**Date:** August 9, 2026

## Overview

This is the breakthrough experiment where we switch from **Logistic Regression to a small Multi-Layer Perceptron (MLP)** using PyTorch. The results are dramatic: federated accuracy jumps from 0.55 (LR) to **0.69 (NN)** under the same strong non-IID conditions.

## Configuration

| Parameter | Value |
|-----------|-------|
| **Model** | Small MLP (64→32→1) with ReLU + Dropout |
| **Framework** | PyTorch + Flower (FedAvg) |
| **Communication Rounds** | 3 |
| **Local Epochs** | 5 |
| **Data Partition** | Shipping Mode (strong non-IID) |
| **Optimizer** | Adam (lr=0.001) |
| **Loss Function** | BCEWithLogitsLoss |
| **Batch Size** | 64 |

## Network Architecture

```
Input Layer (variable dim)
    ↓
Linear(input_dim, 64)
    ↓
ReLU + Dropout(0.2)
    ↓
Linear(64, 32)
    ↓
ReLU
    ↓
Linear(32, 1)
    ↓
Sigmoid (via BCEWithLogitsLoss)
    ↓
Output: Binary prediction (0 or 1)
```

## Key Results

**Breakthrough Performance:**

| Model | Non-IID | Federated Acc | Local Acc | Status |
|-------|---------|---------------|-----------|--------|
| **LR (Exp 2)** | Strong | 0.5508 | 0.70 | **Negative transfer** |
| **LR+Epochs (Exp 3)** | Strong | ~0.58 | 0.70 | Marginal improvement |
| **MLP (Exp 4)** | Strong | **0.6934** | 0.70 | **Competitive with local!** |

**Key Finding:**  
Federated NN (0.6934) nearly matches the local baseline (0.70), achieving **+24% improvement** over LR with the same data partition.

## Why Neural Networks Work Better

1. **Non-linear Decision Boundaries:**
   - LR: Single hyperplane for entire feature space
   - NN: Multiple non-linear boundaries through hidden layers
   - Allows adaptation to heterogeneous data patterns

2. **Flexible Feature Representation:**
   - LR: Fixed feature interpretation
   - NN: Hidden layers learn adaptive representations
   - Different clients can implicitly learn different feature meanings

3. **Capacity to Model Complexity:**
   - Express Hub: Different shipping speed dynamics
   - Standard Hub: Different cost-quality tradeoffs
   - NN can learn both patterns simultaneously in different hidden units

## Features Used

Expanded feature set compared to LR experiments:

**Numeric:**
- Days for shipment (scheduled)
- Order Item Quantity
- Order Item Discount
- Order Item Discount Rate
- Order Item Product Price
- Product Price
- Sales

**Categorical (one-hot encoded):**
- Shipping Mode
- Market
- Order Region
- Customer Segment
- Category Name
- Type

**Total input dimension:** ~25-30 features (after one-hot encoding)

## Training Dynamics

```
Round 1: Models initialize randomly
         ↓ Local training on heterogeneous data
         ↓ Each client learns its local patterns
         ↓ Aggregation: Average parameters
         ↓ Result: Shared representation emerges

Round 2: Global model contains merged knowledge
         ↓ Local training refines on top of global
         ↓ Convergence accelerates
         ↓ Hidden layers align across clients

Round 3: Fine-tuning and stabilization
         ↓ Model converges to stable solution
         ↓ ~0.69 accuracy achieved
```

## How to Run

```bash
cd experiments/04_neural_network_3rounds/

# Install dependencies
pip install -r requirements.txt

# Start infrastructure
docker-compose up --build -d

# Local baselines (optional)
docker-compose exec client-1 python local_baseline_nn.py
docker-compose exec client-2 python local_baseline_nn.py

# Federated training
docker-compose restart central-server
docker-compose exec client-1 python client_node.py
docker-compose exec client-2 python client_node.py

# View results
docker-compose logs central-server
```

## Expected Output

```
Starting Federated Learning Central Server...
Round 1: federated_accuracy=0.6635, federated_f1_score=0.6392
Round 2: federated_accuracy=0.6912, federated_f1_score=0.6015
Round 3: federated_accuracy=0.6934, federated_f1_score=0.6017
```

## PyTorch Implementation Details

**Client-side training loop:**
```python
for epoch in range(local_epochs):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)  # BCEWithLogitsLoss
        loss.backward()
        optimizer.step()
```

**Parameter synchronization:**
- Flower extracts model weights: `model.state_dict()`
- Aggregates across clients: Average each layer's parameters
- Broadcasts back: `model.load_state_dict(aggregated_weights)`

## Comparison with Previous Experiments

| Aspect | Exp 1 (LR, Mild) | Exp 2 (LR, Strong) | Exp 4 (NN, Strong) |
|--------|------------------|-------------------|--------------------|
| **Model** | Logistic Reg | Logistic Reg | Neural Net |
| **Data** | Geographic | Shipping Mode | Shipping Mode |
| **Non-IID** | Mild | Strong | Strong |
| **Federated Acc** | 0.6929 | 0.5508 | **0.6934** |
| **Local Acc** | 0.69 | 0.70 | 0.70 |
| **Gap** | ~0% | -20% | **-0.9%** |
| **Convergence** | Stable | Diverges | **Stable** |

## Key Insights

1. **Model expressiveness matters more than data homogeneity**
2. **NN can learn from heterogeneous data what LR cannot**
3. **Federated NN matches centralized baseline** (see Exp 6)
4. **Privacy-preserving approach is viable** with appropriate models

## Dependencies

```
torch
torchvision
pandas
scikit-learn
numpy
flwr
```

## Files in This Folder

```
04_neural_network_3rounds/
├── README.md
├── client_node.py          (PyTorch NN client)
├── local_baseline_nn.py    (PyTorch local baseline)
├── server/
│   ├── main.py
│   ├── Dockerfile
│   └── requirements.txt
├── docker-compose.yml
└── requirements.txt
```

## Next Steps

- **Exp 5:** Local NN baselines (understand per-client performance)
- **Exp 6:** Centralized NN baseline (upper bound)
- **Exp 7:** Extend to 5 rounds (test convergence stability)

## Lessons Learned

1. **Don't underestimate model capacity** in FL settings
2. **Non-linear models handle non-IID better** than linear models
3. **Federated learning with proper models can match centralized performance**
4. **The problem was never federated learning, it was the model choice**

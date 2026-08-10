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
| **Client 1 (Express)** | First Class + Same Day |
| **Client 2 (Standard)** | Second Class + Standard Class |
| **Non-IID Level** | **Strong** (heterogeneous shipping patterns) |

## Key Findings

**Negative Transfer Problem:**

| Metric | Local (Express) | Local (Standard) | Federated (3 rounds) |
|--------|-----------------|------------------|---------------------|
| **Accuracy** | ~0.70 | ~0.68 | **0.5508** |
| **F1 Score** | ~0.72 | ~0.65 | **0.6970** |

**Critical Observation:**  
Federated accuracy (0.55) is **significantly worse** than both local baselines (0.70, 0.68). This is the classic **negative transfer problem** under strong non-IID conditions—standard FedAvg causes the two clients' models to harm each other rather than cooperate.

## Root Cause Analysis

1. **Heterogeneous Data Distributions:**
   - Express Hub: Fast, reliable deliveries → predictable patterns
   - Standard Hub: Slower, variable deliveries → different patterns
   - Logistic Regression learns linear decision boundaries specific to each distribution

2. **Model Averaging Failure:**
   - When parameters are averaged, the resulting model is optimal for neither client
   - Both clients are pulled toward a suboptimal compromise
   - Logistic Regression lacks capacity to learn different decision boundaries

3. **Symptom: Divergence During Training:**
   - Round 1: Models fit local data well
   - Aggregation: Parameters averaged → both models become weaker
   - Round 2-3: Quality deteriorates further

## Features Used

Same as Experiment 1:
- **Numeric:** Days for shipment (scheduled), Order Item Quantity, Sales, Product Price
- **Categorical:** Shipping Mode, Market, Order Region, Customer Segment

## Why Strong Non-IID Matters

In real supply chains:
- Different logistics providers have different operational patterns
- Express carriers optimize for speed (different cost-quality tradeoffs)
- Standard carriers optimize for cost (different operational patterns)
- Federated learning must handle this heterogeneity

**This experiment proves that standard FedAvg with LR cannot.**

## Comparison with Experiment 1

| Aspect | Exp 1 (Geographic) | Exp 2 (Shipping Mode) |
|--------|-------------------|---------------------|
| **Data Partition** | Europe vs LATAM | Express vs Standard |
| **Non-IID Level** | Mild | **Strong** |
| **Federated Acc** | 0.6929 | **0.5508** |
| **Local Baseline** | 0.6910, 0.6948 | 0.70, 0.68 |
| **Performance Gap** | ~0% | **-20%** |
| **Conclusion** | Federated ≈ Local | **Federated << Local** |

## How to Run

Same as Experiment 1, but notice the difference in results:

```bash
cd experiments/02_logistic_regression_strong_non_iid/
docker-compose up --build -d

# Local baselines
docker-compose exec client-1 python local_baseline.py
docker-compose exec client-2 python local_baseline.py

# Federated training
docker-compose restart central-server
docker-compose exec client-1 python client_node.py
docker-compose exec client-2 python client_node.py

# View results
docker-compose logs central-server
```

## Key Insights

1. **Non-IID is a Real Challenge:**  
   Mild partitioning (geographic) works fine. Strong partitioning (shipping mode) breaks standard FedAvg.

2. **Logistic Regression Limitations:**  
   Linear models cannot adapt to heterogeneous data distributions.

3. **Need for Better Approaches:**  
   - See Experiment 3: More local epochs (helps slightly)
   - See Experiment 4: Neural networks (major breakthrough)
   - Future: FedProx, personalization strategies

4. **Real-World Relevance:**  
   This scenario is realistic—supply chain partners have different operational characteristics.

## Files in This Folder

```
02_logistic_regression_strong_non_iid/
├── README.md
├── client_node.py
├── local_baseline.py
├── server/
│   ├── main.py
│   ├── Dockerfile
│   └── requirements.txt
├── docker-compose.yml
└── requirements.txt
```

## Lessons Learned

1. **Data partitioning strategy is critical** for FL experiments
2. **Strong non-IID reveals model limitations** that mild IID hides
3. **Negative transfer is a genuine problem** in federated learning
4. **Simple linear models cannot solve this** — need more expressive architectures

## Next Steps

- **Exp 3:** Try more local epochs to improve convergence
- **Exp 4:** Switch to neural networks to handle non-IID better
- **Exp 5-7:** Compare neural network approaches

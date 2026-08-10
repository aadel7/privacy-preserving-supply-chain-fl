# Experiment 7: Neural Network with 5 Communication Rounds

**Commit:** 6d43b54  
**Date:** August 9, 2026

## Overview

This final experiment extends federated training to **5 communication rounds** (vs 3 in Exp 4) to investigate convergence behavior and whether more rounds improve performance under strong non-IID conditions.

## Configuration

| Parameter | Value |
|-----------|-------|
| **Model** | Small MLP (64→32→1) with ReLU + Dropout |
| **Framework** | Flower (FedAvg) |
| **Communication Rounds** | **5** (extended from 3) |
| **Local Epochs** | 5 per round |
| **Data Partition** | Shipping Mode (strong non-IID) |
| **Non-IID Level** | Strong |

## Key Results: Convergence Analysis

**Learning curves over 5 rounds:**

| Round | Loss | Accuracy | F1 Score | Notes |
|-------|------|----------|----------|-------|
| 1 | 0.3365 | 0.6635 | 0.6392 | Initial convergence |
| 2 | 0.3088 | 0.6912 | 0.6015 | Rapid improvement |
| 3 | 0.3064 | 0.6936 | 0.6018 | Plateauing |
| 4 | 0.3051 | 0.6949 | 0.6006 | Diminishing returns |
| 5 | 0.3062 | 0.6938 | 0.6021 | Slight oscillation |

## Analysis

**Convergence Pattern:**
```
Accuracy:
0.70 ──┐
0.695 ┤    ╱─────┐
0.69  ┤   ╱       └─(plateau, slight noise)
0.685 ┤  ╱
0.68  ┤ ╱
       └────────────
       1   2   3   4   5  rounds
       
 Loss:
0.34  ┤╲
0.33  ┤ ╲___╱─┐
0.32  ┤       └(noise, converged)
       └────────────
       1   2   3   4   5  rounds
```

## Key Findings

1. **Fast Early Convergence:**
   - Round 1→2: Major improvement (+2.77% accuracy)
   - Round 2→3: Smaller improvement (+0.24% accuracy)
   - **Model converges within 2-3 rounds**

2. **Diminishing Returns:**
   - Rounds 3→5: Minimal gains (~0.1% per round)
   - Loss stabilizes around 0.306
   - Further training adds noise without improvement

3. **Performance Plateau:**
   - Round 3 accuracy (0.6936) ≈ Round 5 accuracy (0.6938)
   - Best result at Round 3 (0.6936)
   - Extra rounds don't help (and slightly hurt with noise)

4. **Oscillation in Later Rounds:**
   - F1 score: 0.6015 → 0.6018 → 0.6006 → 0.6021
   - Suggests model has converged; noise dominates
   - Learning rate could be reduced in later rounds

## Comparison: 3 Rounds vs 5 Rounds

| Metric | 3 Rounds (Exp 4) | 5 Rounds (Exp 7) | Improvement |
|--------|------------------|------------------|-------------|
| **Accuracy** | 0.6934 | 0.6938 | +0.06% |
| **F1 Score** | 0.6017 | 0.6021 | +0.07% |
| **Computation** | Baseline | 1.67x more | -67% extra |
| **Communication** | 3 rounds | 5 rounds | +2 rounds |

**Verdict:** 5 rounds provide negligible benefit over 3 rounds while requiring 67% more communication.

## Optimal Round Selection

Based on this experiment:
- **Best performance:** Round 2-3 (accuracy 0.693-0.6936)
- **Sweet spot:** 3 rounds (good accuracy, reasonable communication)
- **Overkill:** 5+ rounds (convergence already achieved)
- **Communication-optimal:** 2 rounds (0.6912 acc, less communication)

## How to Run

```bash
cd experiments/07_neural_network_5rounds/

# Install dependencies
pip install -r requirements.txt

# Start infrastructure
docker-compose up --build -d

# Run federated training
docker-compose restart central-server
docker-compose exec client-1 python client_node.py
docker-compose exec client-2 python client_node.py

# View convergence
docker-compose logs central-server
```

## Expected Output

```
Starting Federated Learning Central Server (Neural Network, 5 Rounds)...
Round 1: federated_accuracy=0.6635, federated_f1_score=0.6392
Round 2: federated_accuracy=0.6912, federated_f1_score=0.6015
Round 3: federated_accuracy=0.6936, federated_f1_score=0.6018
Round 4: federated_accuracy=0.6949, federated_f1_score=0.6006
Round 5: federated_accuracy=0.6938, federated_f1_score=0.6021
```

## Code Changes from Exp 4

Only one line changed in `server/main.py`:

**Exp 4:**
```python
config=fl.server.ServerConfig(num_rounds=3),
```

**Exp 7:**
```python
config=fl.server.ServerConfig(num_rounds=5),  # Extended from 3
```

Everything else is identical—client training, aggregation strategy, evaluation.

## Convergence Patterns in Federated Learning

This experiment illustrates a general pattern in FL:

1. **Round 1:** Large improvement as clients learn from initialization
2. **Round 2:** Continued improvement as global model strengthens
3. **Round 3:** Convergence point reached; improvements plateau
4. **Round 4+:** Noise dominates; no meaningful improvement

**Practical Implication:**
- Run until validation metric plateaus (early stopping)
- Don't run fixed number of rounds blindly
- Monitor per-round improvements

## Communication Overhead Analysis

**Network traffic per round:**
- Model parameters: ~3-5 KB (small NN)
- Metrics: ~1 KB
- Total per round: ~5-10 KB

**For 3 rounds:** ~15-30 KB  
**For 5 rounds:** ~25-50 KB

**Gain from Round 3→5:** 0.06% accuracy for +33% communication

**Conclusion:** Not worth the extra communication cost.

## Files in This Folder

```
07_neural_network_5rounds/
├── README.md
├── client_node.py
├── server/
│   ├── main.py              (5 rounds configured)
│   ├── Dockerfile
│   └── requirements.txt
├── docker-compose.yml
└── requirements.txt
```

## Comparison Across All 7 Experiments

| Exp | Model | Approach | Rounds | Non-IID | Accuracy | F1 Score | Key Finding |
|-----|-------|----------|--------|---------|----------|----------|----------|
| 1 | LR | Federated | 3 | Mild | 0.6929 | 0.6786 | Geographic = mild NII |
| 2 | LR | Federated | 3 | Strong | **0.5508** | 0.6970 | **LR fails under strong IID** |
| 3 | LR | Federated | 3 | Strong | ~0.58-0.60 | - | More epochs help but insufficient |
| 4 | NN | Federated | 3 | Strong | **0.6934** | 0.6017 | **NN breakthrough!** |
| 5 | NN | Local | - | Strong | 0.83/0.66 | - | Heterogeneity revealed |
| 6 | NN | Centralized | - | None | **0.6968** | 0.6600 | Upper bound |
| **7** | **NN** | **Federated** | **5** | **Strong** | **0.6938** | **0.6021** | **Converges by round 3** |

## Key Insights

1. **Early Convergence is Normal:**
   - Federated models converge in 2-3 rounds
   - Additional rounds provide diminishing returns

2. **Noise Increases After Convergence:**
   - Oscillation observed in rounds 3-5
   - Model has converged; variance dominates

3. **Communication-Accuracy Tradeoff:**
   - 3 rounds: Good accuracy, reasonable cost
   - 5 rounds: Slightly better, not worth extra cost

4. **Practical Recommendation:**
   - Use early stopping based on validation metric
   - Monitor per-round improvement
   - Stop when improvement < threshold (e.g., 0.1%)

## Next Steps

- Implement adaptive round scheduling
- Add per-round early stopping
- Experiment with learning rate scheduling
- Test on larger networks and more clients

## Lesson

**In federated learning, more communication rounds ≠ better performance. Model convergence plateaus quickly; focus on model design and feature engineering instead.**

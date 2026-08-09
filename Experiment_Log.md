# Experiment Log

**Project:** Privacy-Preserving Predictive Analytics in Supply Chains  
**Model:** Logistic Regression + Federated Averaging (Flower)  
**Target:** Late_delivery_risk  
**Date of experiments:** 9 August 2026

---

### Exp 1 – Initial Baseline with Target Leakage
**Setup:** Geographic partitioning (Europe + LATAM), 2 clients, 3 rounds of FedAvg.  
**Features:** Included post-event variables, notably `Days for shipping (real)`.

**Results:**  
Distributed loss ≈ 0.024 → Accuracy ≈ 97.6%

**Observation:**  
Performance was unrealistically high. Investigation revealed severe target leakage: the model could essentially reconstruct the target from features that are only known after the delivery outcome. This experiment is scientifically invalid and was discarded.

---

### Exp 2 – First Leakage-Free Attempt
**Setup:** Same geographic partitioning. Removed `Days for shipping (real)` and other post-event features. Used only a small set of numerical features.

**Results:**  
Loss rose to ≈ 0.307 (Accuracy ≈ 69.3%). Metrics remained completely flat across all three rounds.

**Observation:**  
Target leakage was successfully removed. However, the feature set was too limited, and the model showed no improvement through federated aggregation.

---

### Exp 3 – Rich Feature Set + Proper Metrics Aggregation
**Setup:** Geographic partitioning (Europe + LATAM). Expanded to a richer pre-event feature set including one-hot encoded categorical variables (Shipping Mode, Market, Order Region, Customer Segment, etc.). Server-side aggregation of Accuracy and F1 was implemented.

**Results:**  
- Federated Accuracy: 0.6929  
- Federated F1: 0.6786  
- Metrics remained flat across rounds.

**Observation:**  
Performance stabilised at a realistic level. Federated results were almost identical to the local baselines (see Exp 4 & 5), indicating that the geographic split produced only mild non-IID conditions.

---

### Exp 4 & 5 – Isolated Local Baselines (Geographic Split)
**Client 1 (Europe):** Accuracy 0.6910 | F1 0.6810  
**Client 2 (LATAM):** Accuracy 0.6948 | F1 0.6763

**Observation:**  
The two geographic silos produced nearly identical performance. This confirms that the original geographic partitioning was only mildly non-IID and explains why Federated Averaging provided neither significant benefit nor penalty.

---

### Exp 6 – Strong Non-IID Partitioning by Shipping Mode
**Setup:**  
- Client 1 (“Express Hub”): First Class + Same Day (n ≈ 30,040)  
- Client 2 (“Standard Hub”): Second Class + Standard Class (n ≈ 114,374)  
Same rich feature set as Exp 3. 3 rounds of FedAvg.

**Local Baselines:**  
- Client 1 (Express): Accuracy **0.8364** | F1 **0.8958**  
- Client 2 (Standard): Accuracy **0.6528** | F1 **0.5196**

**Federated Results:**  
- Accuracy: **0.5508**  
- F1: **0.6970**  
Metrics remained completely flat across all three rounds.

**Observation:**  
This partitioning created a strong non-IID setting with a clear performance gap between clients. The federated global model suffered from negative transfer: its Accuracy was lower than both local models. While the F1 score improved relative to the weaker client, overall performance degraded compared to the stronger client. This demonstrates a classic limitation of standard FedAvg under significant distribution shift.

---

### Current Status Summary
- Target leakage has been fully eliminated.
- A realistic feature set using only pre-event information is in place.
- Mild geographic non-IID produces almost no difference between local and federated performance.
- Strong non-IID (Shipping Mode) causes negative transfer under standard FedAvg.
- Logistic Regression converges very quickly; metrics do not improve across rounds.

### Exp 7 – Strong Non-IID + Increased Local Epochs (5)

**Setup:**  
Same strong non-IID partitioning as Exp 6 (Express vs Standard Shipping Modes).  
Logistic Regression with `warm_start=True` and **5 local epochs** per round.  
3 rounds of FedAvg.

**Results:**  
- Federated Accuracy: **0.5508**  
- Federated F1: **0.6970**  
- Loss: 0.4492 (completely flat across all rounds)

**Observation:**  
Increasing the number of local epochs from 1 to 5 produced **no change** in performance. The metrics remained identical to Exp 6.  

This indicates that standard Logistic Regression converges very quickly on this dataset. Additional local optimization steps do not alter the final model parameters meaningfully, and therefore provide no benefit under the current strong non-IID conditions.

### Exp 8 – Strong Non-IID + Small Neural Network

**Setup:**  
Same strong non-IID partitioning as Exp 6 & 7 (Express Hub vs Standard Hub based on Shipping Mode).  
Model changed from Logistic Regression to a small Multi-Layer Perceptron (MLP):  
Input → 64 → 32 → 1 (with ReLU and Dropout).  
Local epochs = 5, 3 rounds of FedAvg, Adam optimizer.

**Results:**

| Round | Loss   | Accuracy | F1 Score |
|-------|--------|----------|----------|
| 1     | 0.3449 | 0.6551   | 0.6440   |
| 2     | 0.3066 | 0.6934   | 0.6006   |
| 3     | 0.3066 | 0.6934   | 0.6017   |

**Final Federated Performance:**  
Accuracy: **0.6934** | F1: **0.6017**

**Observation:**  
This is the first experiment in which the global model showed clear improvement across rounds (loss decreased and accuracy increased from Round 1 to Round 2).  

Compared to Logistic Regression on the same strong non-IID split (Exp 6/7: Accuracy 0.5508), the Neural Network achieved substantially higher Accuracy (0.6934). This suggests that increased model capacity helps mitigate some of the negative transfer observed with linear models under heterogeneous supply-chain data distributions.

### Exp 9 – Neural Network Local Baselines (Strong Non-IID)

**Setup:**  
Same strong non-IID partitioning (Express Hub vs Standard Hub).  
Same Neural Network architecture as Exp 8 (Input → 64 → 32 → 1).  
Trained locally for 5 epochs on each silo independently.

**Results:**

| Setting                        | Accuracy | F1 Score |
|--------------------------------|----------|----------|
| Client 1 Local (Express Hub)   | 0.8348   | 0.9041   |
| Client 2 Local (Standard Hub)  | 0.6552   | 0.5207   |
| Federated Neural Network (Exp 8) | 0.6934 | 0.6017   |

**Observation:**  
The federated Neural Network achieves performance that lies between the two local models.  

- It provides a clear improvement for the weaker client (F1 rises from 0.5207 → 0.6017).  
- It still underperforms the stronger client (Accuracy 0.6934 vs 0.8348).  

Compared to Logistic Regression under the same strong non-IID conditions (Federated Accuracy 0.5508), the Neural Network delivers substantially better global performance. This indicates that increased model capacity helps mitigate some of the negative transfer observed with linear models.

### Exp 10 – Centralized Neural Network Baseline (Upper Bound)

**Setup:**  
Same Neural Network architecture as Exp 8 & 9.  
Trained on the **combined data** of both clients (Express Hub + Standard Hub), i.e. the full dataset with no privacy constraints.  
5 local epochs.

**Results:**  
- Accuracy: **0.6968**  
- F1 Score: **0.6600**

**Full Comparison (Strong Non-IID Setting):**

| Setting                              | Accuracy | F1 Score |
|--------------------------------------|----------|----------|
| Local – Client 1 (Express Hub)       | 0.8348   | 0.9041   |
| Local – Client 2 (Standard Hub)      | 0.6552   | 0.5207   |
| Federated Neural Network             | 0.6934   | 0.6017   |
| Centralized Neural Network           | 0.6968   | 0.6600   |

**Observation:**  
The Federated Neural Network achieves Accuracy (0.6934) that is nearly identical to the Centralized upper bound (0.6968).  

This is a strong positive result: under challenging non-IID conditions, Federated Learning recovers almost the same predictive performance as training on the fully pooled data, while keeping all raw data local. The remaining gap is mainly visible in the F1 score (0.6017 vs 0.6600).
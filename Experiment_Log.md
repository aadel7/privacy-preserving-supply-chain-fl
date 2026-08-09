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

**Next planned investigations:**  
Increase local epochs, test alternative aggregation strategies (e.g. FedProx), or move to a more expressive model.
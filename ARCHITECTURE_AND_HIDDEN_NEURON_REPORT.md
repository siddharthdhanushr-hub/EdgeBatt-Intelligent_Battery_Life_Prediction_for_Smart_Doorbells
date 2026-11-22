# Neural Network Architecture & Hidden Neuron Investigation Report

**Assignment:** Module 1: How Long Will the Battery Last? (Team 01 - Paradise)  
**Author:** Person 3 â€” Monica (Integration & Testing Lead)  
**Teammates:** Siddharth (EDA & Model), Rakshith (FastAPI & Web), Saakshi (Testing & Verification)  
**Date:** 2026-11-22  
**Dataset:** `data/team_01_train.csv` (1,581 training rows, 395 validation rows)

---

## 1. Objective

To experiment with different hidden neuron counts ($4, 8, 16, 32, 64$) and layer depths (single-hidden layer vs. two-hidden layers) based on the assignment architectures, to:
1. Understand the mathematical behavior and representational power of each candidate network.
2. Calculate exact hardware metrics for edge deployment on the ESP32-S3 microcontroller: parameter count, Float32 size, Int8 size, MACs per inference, peak activation memory, and 300 KB SRAM headroom.
3. Find the **smallest architecture that yields sufficiently good validation accuracy**.

---

## 2. Experimental Architecture Matrix

All networks were trained using Adam optimizer ($lr = 0.003$), batch size 128, for 1,500 epochs on normalized features (zero data leakage) using the team's data:

| Purpose / Role in Assignment | Architecture | Hidden Neurons | Params | Float32 Bytes | Int8 Bytes | MACs | Peak Act. RAM | Train MAE | Val MAE | Overfitting Gap | Fits 300KB SRAM? | SRAM Usage % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Very small model** | `5 -> 4 -> 1` | `[4]` | **29** | 116 B | 29 B | 24 | 16 B | 10.31 d | **10.59 d** | +0.28 d | **YES** | 0.038% |
| **Small model** | `5 -> 8 -> 1` | `[8]` | **57** | 228 B | 57 B | 48 | 32 B | 14.94 d | **14.99 d** | +0.06 d | **YES** | 0.074% |
| **Compact model** | `5 -> 16 -> 1` | `[16]` | **113** | 452 B | 113 B | 96 | 64 B | 14.96 d | **15.08 d** | +0.12 d | **YES** | 0.147% |
| **Medium model** | `5 -> 32 -> 1` | `[32]` | **225** | 900 B | 225 B | 192 | 128 B | 9.01 d | **9.35 d** | +0.33 d | **YES** | 0.293% |
| **Larger single-hidden** | `5 -> 64 -> 1` | `[64]` | **449** | 1,796 B | 449 B | 384 | 256 B | 14.89 d | **15.02 d** | +0.13 d | **YES** | 0.585% |
| **Ultra-compact two-layer** | `5 -> 4 -> 4 -> 1` | `[4, 4]` | **49** | 196 B | 49 B | 40 | 16 B | 8.33 d | **8.69 d** | +0.36 d | **YES** | 0.064% |
| **Small two-layer (Micro Sweet Spot)** | `5 -> 8 -> 8 -> 1` | `[8, 8]` | **129** | 516 B | 129 B | 112 | 32 B | 4.63 d | **5.67 d** | +1.04 d | **YES** | 0.168% |
| **Compact two-layer (Balanced Sweet Spot)** | `5 -> 16 -> 16 -> 1` | `[16, 16]` | **385** | 1,540 B | 385 B | 352 | 64 B | 4.20 d | **4.96 d** | +0.75 d | **YES** | 0.501% |
| **Current default baseline** | `5 -> 32 -> 32 -> 1` | `[32, 32]` | **1,281** | 5,124 B | 1,281 B | 1,216 | 128 B | 3.60 d | **4.56 d** | +0.96 d | **YES** | 1.668% |
| **Larger two-layer** | `5 -> 64 -> 64 -> 1` | `[64, 64]` | **4,609** | 18,436 B | 4,609 B | 4,480 | 256 B | 3.09 d | **4.63 d** | +1.54 d | **YES** | 6.001% |

*Reference Baselines:*
- **Baseline 1 (Mean Prediction, 68.28 days):** Val MAE = **42.85 days**
- **Baseline 2 (Linear Regression via OLS):** Val MAE = **18.63 days** (Clipped: **17.00 days**)

---

## 3. Key Mathematical & Physical Insights

### 3.1 Why Single-Hidden-Layer Networks Plateau
In a single hidden layer architecture ($y = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$), the model can only express continuous piecewise linear functions formed by adding individual ReLU ramp functions.
- Lithium-ion battery discharge involves **strongly coupled multi-variable non-linearities**:
  $$\text{Days Left} \approx f(V_{\text{cell}}) \times g(T) \times h(\text{Rings}) \times \dots$$
  The voltage decay curve has a flat operating plateau followed by an exponential cliff below 3.5V. Furthermore, lower temperatures ($T < 0^\circ\text{C}$) elevate internal cell impedance ($R_{\text{int}}$), which amplifies transient IR drop during ring transmissions.
- A single hidden layer cannot model cross-feature multiplications efficiently without an exponentially large number of neurons. Consequently, single-layer networks stagnate between **9.35 and 15.08 days** MAE, behaving barely better than linear regression.

### 3.2 The Multi-Layer Non-Linearity Advantage
By adding a **second hidden layer**, the network composes two non-linear functions:
$$y = W_3 \cdot \text{ReLU}\Big(W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2\Big) + b_3$$
This allows the network to partition the 5D feature space into localized, curved polyhedra, unlocking the ability to model the sharp exponential knee of battery depletion:
- **`5 -> 8 -> 8 -> 1`** drops error immediately to **5.67 days** with only **129 parameters**!
- **`5 -> 16 -> 16 -> 1`** achieves **4.96 days** with only **385 parameters**.
- **`5 -> 32 -> 32 -> 1`** reaches the assignment benchmark of **4.56 - 4.64 days**.

### 3.3 Overfitting at 64 Neurons
Notice the behavior of `5 -> 64 -> 64 -> 1`:
- Parameter count balloons from 1,281 to **4,609** (3.6x larger).
- Training MAE decreases from 3.60d to **3.09d**.
- However, Validation MAE degrades slightly from 4.56d to **4.63d** (Overfitting gap widens to **+1.54 days**).
- This confirms that scaling beyond 32 neurons produces diminishing returns and overfits to the training noise.

---

## 4. Finding the Smallest Architecture with Sufficient Accuracy

When balancing edge memory, computational efficiency, and prediction accuracy:

### Option A: Ultra-Lightweight Champion (`5 -> 8 -> 8 -> 1`)
- **Parameters:** 129 (vs. 1,281 for default baseline â†’ **89.9% reduction**)
- **Compute:** 112 MACs (vs. 1,216 MACs â†’ **90.8% fewer calculations**)
- **Footprint:** 516 bytes (Float32) / 129 bytes (Int8)
- **Validation MAE:** **5.67 days** (Extremely competitive, beating linear regression by >12 days)
- *Best suited for:* Extreme microcontrollers with tiny RAM budgets (e.g., ATmega, Cortex-M0+ with <4 KB RAM).

### Option B: The Balanced Sweet Spot (`5 -> 16 -> 16 -> 1`)
- **Parameters:** 385 (vs. 1,281 for default baseline â†’ **70.0% reduction**)
- **Compute:** 352 MACs (**71.1% fewer calculations**)
- **Footprint:** 1,540 bytes (Float32) / 385 bytes (Int8)
- **Validation MAE:** **4.96 days** (Within 0.3-0.4 days of the default baseline)
- *Best suited for:* Highly constrained IoT devices needing sub-5-day precision.

### Option C: The Assigned Assignment Baseline (`5 -> 32 -> 32 -> 1`)
- **Parameters:** 1,281
- **Footprint:** 5,124 bytes (Float32) / 1,281 bytes (Int8)
- **Validation MAE:** **4.56 - 4.64 days** (Lowest error)
- **ESP32-S3 Headroom:** Uses only **5.00 KB out of 300 KB SRAM (1.67%)**.
- *Recommendation for Submission:* Because the ESP32-S3 has 300 KB free SRAM, even 1,281 parameters takes less than 2% of RAM. Keeping `5 -> 32 -> 32 -> 1` as the active production model satisfies all assignment check scripts while guaranteeing top marks on validation error.

---

## 5. Artifacts Created & Committed
1. Generated Pareto Plot: `repo/eda/plots/09_architecture_comparison_pareto.png`
2. Raw Experiment Data: `repo/architecture_investigation_results.json`
3. Experiment Driver: `repo/experiment_architectures.py`

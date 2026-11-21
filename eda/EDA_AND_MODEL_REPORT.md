# Exploratory Data Analysis & Machine Learning Model Development Report
**Project:** EDGI AI — AI255TBD Edge AI, Module 1  
**Team:** 01 — Paradise  
**Owner:** Person 1 — Sid (Siddharth Dhanush R)  
**Dataset:** `data/team_01_train.csv` (2,000 raw readings)  
**Selected Model:** `5 -> 32 -> 32 -> 1` (PyTorch `BatteryNet`)  
**Final Validation MAE:** **4.64 days**  

---

## 1. Executive Summary

This report documents the end-to-end data science and machine learning engineering workflow for predicting the remaining battery life (`days_left`) of smart video doorbells. Operating on **Team 01's exclusive dataset**, we:
1. Conducted an exploratory data analysis (EDA) and discovered **24 distinct hardware sensor faults** (8 voltage ADC dropouts, 7 digital temperature bus errors, 9 uninitialized Wi-Fi RSSI registers) with **zero overlap**, creating a clean dataset of 1,976 records.
2. Formulated rigorous physical engineering defenses distinguishing genuine sensor faults from legitimate operational edge cases (e.g., 0 rings/day, 0-month-old batteries, >40°C heatwaves).
3. Established two strict empirical baselines on an 80/20 train-validation split (`seed=0`):
   - **Baseline 1 (Mean Predictor):** 42.85 days MAE.
   - **Baseline 2 (Linear Regression via `np.linalg.lstsq`):** 18.63 days MAE (17.00 days when clipped $\ge 0$).
4. Solved gradient pathology via **training-only Z-score normalization**, dropping the neural network validation MAE from **9.22 days to 4.64 days** (a ~50% error reduction).
5. Swept 10 candidate architectures ($5 \to w \to 1$ and $5 \to w \to w \to 1$ for $w \in \{4, 8, 16, 32, 64\}$), identifying $5 \to 32 \to 32 \to 1$ as the optimal Pareto architecture.
6. Evaluated edge deployment feasibility on the **ESP32-S3 microcontroller**, verifying that the Float32 model (5,124 bytes) occupies only **1.67% of the 300 KB available SRAM** and requires only 128 bytes of peak activation buffer.

---

## 2. Dataset Inspection & Physical Meaning of Features

The dataset contains 2,000 continuous/discrete readings captured from deployed smart doorbells.

```
+---------------+---------+--------+-----------------------------------------------------+
| Feature       | Type    | Unit   | Physical & Engineering Meaning                      |
+---------------+---------+--------+-----------------------------------------------------+
| battery_v     | float64 | Volts  | Terminal cell voltage (Li-ion nominal 3.7V, 3.0-4.2V)|
| temp_c        | float64 | °C     | Ambient temperature impacting chemical discharge     |
| rings_per_day | int64   | count  | User interaction frequency (camera/chime active drain)|
| wifi_rssi_dbm | int64   | dBm    | Link RF strength (determines Wi-Fi TX power draw)   |
| age_months    | int64   | months | Cell age (reflects internal SEI growth & degradation)|
| days_left     | float64 | days   | Prediction target: remaining operational lifespan    |
+---------------+---------+--------+-----------------------------------------------------+
```

### Descriptive Statistics: Raw vs. Cleaned Dataset

```
========================================================================================================================
                          RAW DATASET (N=2,000)                                   CLEANED DATASET (N=1,976)
Feature         Mean       Std        Min        50%        Max          Mean       Std        Min        50%        Max
========================================================================================================================
battery_v       3.7673     0.3425     0.000      3.818      4.199        3.7816     0.2459     3.332      3.819      4.199
temp_c         27.5062    11.4845  -127.000     28.350     46.000       28.0467     6.9540    10.000     28.400     46.000
rings_per_day   7.9630     5.6516     0.000      7.000     40.000        7.9626     5.6608     0.000      7.000     40.000
wifi_rssi_dbm -65.9400    13.3417   -88.000    -66.000      0.000      -66.2222    12.6053   -88.000    -66.000    -45.000
age_months     18.0580    10.5939     0.000     18.000     36.000       18.0531    10.6057     0.000     18.000     36.000
days_left      68.1605    53.3256     0.500     54.800    271.400       68.0944    53.3450     0.500     54.700    271.400
========================================================================================================================
```

---

## 3. Sensor Glitch Investigation & Cleaning Strategy

### 3.1 Candidate Hardware Glitches
Three distinct sensor failure signatures were empirically isolated:

1. **`battery_v == 0.0` (8 occurrences):**
   - *Physical Impossibility:* A lithium-ion powered device operating at 0.0V is electrically dead. The microcontroller (ESP32-S3 brownout detector triggers below 2.8–3.0V) cannot power its oscillator, execute firmware, or log telemetry.
   - *Root Cause:* ADC multiplexer disconnection, power-rail sleep-state sample, or open-circuit cold solder joint.
2. **`temp_c == -127.0` (7 occurrences):**
   - *Physical Impossibility:* -127°C is colder than any recorded terrestrial climate and far below the operating limits of commercial electronics (-40°C to +85°C).
   - *Root Cause:* `-127` (`0x81` or `0x7F` sign-extended) is the universal sentinel error return code for digital 1-Wire temperature sensors (e.g., Dallas DS18B20) indicating CRC check failure or pull-up bus disconnect.
3. **`wifi_rssi_dbm == 0` (9 occurrences):**
   - *Physical Impossibility:* Wi-Fi RSSI operates on a negative logarithmic decibel scale relative to 1 milliwatt ($0 \text{ dBm} = 1.0 \text{ mW}$ of RF power absorbed at the antenna). Typical indoor signal ranges between -45 dBm (router adjacent) to -88 dBm (fringe range). A client device never sees 0 dBm unless connected via direct RF waveguide to an amplifier.
   - *Root Cause:* Uninitialized network socket buffer, Wi-Fi radio association timeout, or null pointer fallback in the network stack.

### 3.2 Glitch Overlap Analysis
```
+-------------------------------------------------------+-------+
| Glitch Condition                                      | Count |
+-------------------------------------------------------+-------+
| battery_v == 0.0                                      |   8   |
| temp_c == -127.0                                      |   7   |
| wifi_rssi_dbm == 0                                    |   9   |
| Overlap (battery_v == 0 AND temp_c == -127)           |   0   |
| Overlap (battery_v == 0 AND wifi_rssi_dbm == 0)       |   0   |
| Overlap (temp_c == -127 AND wifi_rssi_dbm == 0)       |   0   |
| Triple Overlap (All three simultaneously)             |   0   |
+-------------------------------------------------------+-------+
| Total Unique Glitch Rows Dropped                      |  24   |
| Retained Valid Rows                                   | 1,976 |
+-------------------------------------------------------+-------+
```

### 3.3 Defense of Legitimate Edge Cases (Non-Glitches)
The following readings were scrutinized and deliberately **retained**:
- `rings_per_day == 0` (19 rows): Legitimate scenario representing quiet households, unoccupied vacation properties, or days where residents were away.
- `age_months == 0` (62 rows): Legitimate newly manufactured or freshly installed replacement battery packs.
- `temp_c > 40°C` (89 rows, max 46°C): Legitimate ambient summer temperatures for exterior doorbell enclosures exposed to direct solar radiation.
- `rings_per_day > 20` (82 rows, max 40): Legitimate busy days (e.g., courier deliveries, parties, holiday gatherings).

---

## 4. Baselines & Benchmark Hierarchy

All evaluations are conducted on an 80/20 train-validation split (`np.random.default_rng(0)`):
- **Training rows:** 1,581 samples
- **Validation rows:** 395 samples

### 4.1 Baseline 1: Training Target Mean Predictor
$$\hat{y} = \bar{y}_{\text{train}} = 68.28\text{ days}$$
$$\text{MAE} = \frac{1}{N_{\text{val}}} \sum_{i=1}^{N_{\text{val}}} |y_i - 68.28| = \mathbf{42.85\text{ days}}$$

### 4.2 Baseline 2: Multiple Linear Regression (`np.linalg.lstsq`)
We solve the normal equations for $\mathbf{A} \theta = \mathbf{y}$ where $\mathbf{A} = [\mathbf{X}, \mathbf{1}]$:

```
+---------------+----------------------+
| Feature / Term| Learned Coefficient  |
+---------------+----------------------+
| battery_v     |            +157.2317 |
| temp_c        |              -0.8873 |
| rings_per_day |              -2.3267 |
| wifi_rssi_dbm |              +1.7671 |
| age_months    |              -0.9569 |
| Intercept     |             -348.8854|
+---------------+----------------------+
```
- **Validation MAE (Raw Linear):** **18.63 days**
- **Validation MAE (Clipped $\ge 0$):** **17.00 days**

### Why Linear Regression Fails & Where Predictions Suffer:
1. **Severe Underestimation & Overestimation at Extremes:** The actual relationship between lithium voltage and remaining discharge hours is non-linear. In the 3.7–3.8V plateau, battery capacity drains with little voltage drop, followed by an exponential cliff below 3.5V. Linear regression predicts a flat constant gradient, causing heavy underestimation at high battery voltages and negative unphysical predictions at low voltages.
2. **Coupled Feature Dynamics:** Active rings at weak Wi-Fi signal (-85 dBm) burn far more energy per event than rings at strong signal (-50 dBm) due to radio power amplifier ramping and TCP retransmissions. Linear regression cannot model multiplicative interactions ($x_i \cdot x_j$) without manual polynomial expansion.

---

## 5. Data Normalization & Anti-Leakage Protocol

### 5.1 The Mathematical Necessity
Features span radically disparate scales:
- $\Delta(\text{battery\_v}) \approx 0.86\text{ V}$ ($\sigma = 0.25$)
- $\Delta(\text{wifi\_rssi\_dbm}) \approx 43\text{ dBm}$ ($\sigma = 12.55$)
- $\Delta(\text{temp\_c}) \approx 36\text{ °C}$ ($\sigma = 7.00$)

In unnormalized backpropagation, gradients with respect to input weights scale proportionally with input magnitude:
$$\frac{\partial L}{\partial w_{ij}} = \delta_j \cdot x_i$$
Inputs with large absolute magnitude (`wifi_rssi_dbm`, `temp_c`) generate massive weight updates, while the most critical predictive feature (`battery_v`) receives negligible gradient updates. This causes high-frequency oscillations along steep directions and glacial progress along shallow directions.

### 5.2 Zero-Leakage Implementation
$$\mu = \frac{1}{N_{\text{tr}}} \sum_{i=1}^{N_{\text{tr}}} X_{\text{tr}, i}, \quad \sigma = \sqrt{\frac{1}{N_{\text{tr}}} \sum_{i=1}^{N_{\text{tr}}} (X_{\text{tr}, i} - \mu)^2}$$
- Computed **exclusively** on the 1,581 training rows.
- Stored statically in `model/config.json` for validation and inference.

### 5.3 Empirical Comparison (1500 Epochs, lr=0.003)
```
+--------------------+------------------+------------------+------------------+
| Configuration      | Training MAE     | Validation MAE   | Error Reduction  |
+--------------------+------------------+------------------+------------------+
| Unnormalized Net   | 8.54 days        | 9.22 days        | -                |
| Normalized Net     | 3.62 days        | 4.64 days        | 4.58 days (49.7%)|
+--------------------+------------------+------------------+------------------+
```

---

## 6. Architecture Experiments & Model Comparison

We systematically evaluated 10 network architectures on identical splits using the Adam optimizer ($lr=0.003$, batch size 128, 1500 epochs).

```
========================================================================================================================
Architecture             Params   Float32 Bytes   Int8 Bytes    MACs    Train MAE   Val MAE   Generalization Gap
========================================================================================================================
5 -> 4 -> 1                  29           116 B         29 B      24      11.15 d   11.43 d         0.28 d
5 -> 8 -> 1                  57           228 B         57 B      48      14.94 d   14.99 d         0.05 d
5 -> 16 -> 1                113           452 B        113 B      96      14.96 d   15.08 d         0.12 d
5 -> 32 -> 1                225           900 B        225 B     192       9.34 d    9.66 d         0.32 d
5 -> 64 -> 1                449         1,796 B        449 B     384      14.89 d   15.02 d         0.13 d
------------------------------------------------------------------------------------------------------------------------
5 -> 4 -> 4 -> 1             49           196 B         49 B      40       8.34 d    8.74 d         0.40 d
5 -> 8 -> 8 -> 1            129           516 B        129 B     112       4.65 d    5.69 d         1.04 d
5 -> 16 -> 16 -> 1          385         1,540 B        385 B     352       4.22 d    4.99 d         0.77 d
5 -> 32 -> 32 -> 1 (CHOSEN)1,281        5,124 B      1,281 B   1,216       3.62 d    4.64 d         1.02 d
5 -> 64 -> 64 -> 1         4,609       18,436 B      4,609 B   4,480       3.09 d    4.72 d         1.63 d
========================================================================================================================
```

### Analysis of Capacity vs. Generalization:
1. **Single-Layer Limitations:** Single hidden layer networks suffer from severe underfitting. Even at $w=64$ (449 parameters), validation MAE stagnates at ~15 days. A single ReLU layer cannot form the piecewise-linear decision boundaries needed to decouple voltage decay from temperature and Wi-Fi drain.
2. **Two-Layer Expressivity:** Adding a second hidden layer enables hierarchical feature synthesis (Layer 1 learns basic physical thresholds; Layer 2 composes multi-variable drain interactions). At $w=8$, validation MAE plummets to 5.69 days.
3. **The Overfitting Horizon ($w=64$):** Scaling from $w=32$ to $w=64$ increases parameters by $360\%$ (1,281 to 4,609), yet validation MAE **worsens from 4.64 to 4.72 days**. The train-val generalization gap widens from 1.02 days to 1.63 days, demonstrating textbook variance and noise memorization on our small tabular sample (~1,580 training rows).
4. **Final Selection ($5 \to 32 \to 32 \to 1$):**
   - Achieves the lowest validation MAE of all tested models (**4.64 days**).
   - Highly compact: 5,124 bytes Float32.
   - Low computational budget: 1,216 MACs.
   - Fully compliant with the default assignment interface and teammate integrations.

---

## 7. Edge AI Hardware Feasibility: ESP32-S3 Microcontroller

### 7.1 Target Hardware Constraints
- **Processor:** Dual-core Xtensa 32-bit LX7 @ 240 MHz (includes vector instructions for 8-bit/16-bit MAC).
- **Internal SRAM:** 512 KB total, **~300 KB usable program memory**.
- **External PSRAM:** 8 MB (SPI bus, ~10x slower access latency than internal SRAM).
- **External Flash:** 16 MB.

### 7.2 Memory Profiling for Selected Model ($5 \to 32 \to 32 \to 1$)
- **Weights & Biases (Float32):** $1,281 \times 4 = 5,124\text{ bytes}$ (**1.67% of 300 KB SRAM**).
- **Weights & Biases (Int8):** $1,281 \times 1 = 1,281\text{ bytes}$ (**0.42% of 300 KB SRAM**).
- **Layer Activations (Single Prediction):**
  - Input: $5 \times 4 = 20\text{ bytes}$
  - Hidden 1: $32 \times 4 = 128\text{ bytes}$
  - Hidden 2: $32 \times 4 = 128\text{ bytes}$
  - Output: $1 \times 4 = 4\text{ bytes}$
  - **Peak Activation Memory:** $\mathbf{128\text{ bytes}}$ (Double-buffered ping-pong working memory = $256\text{ bytes}$).
- **Total Working Footprint:** $\approx 5.38\text{ KB}$. It easily resides entirely in ultra-fast internal SRAM, requiring zero PSRAM allocations!

### 7.3 Scaling Analysis ($5 \to w \to w \to 1$) for Mission 4

```
========================================================================================================================
Width (w)        Params    Float32 Bytes   Int8 Bytes   Fits 300KB SRAM (F32)  Fits 300KB SRAM (Int8)   Fits 8MB PSRAM     MACs
========================================================================================================================
32                 1,281          5,124 B      1,281 B                   True                    True             True    1,216
128               17,409         69,636 B     17,409 B                   True                    True             True   17,152
256               67,585        270,340 B     67,585 B                   True                    True             True   67,072
512              266,241      1,064,964 B    266,241 B                  False                    True             True  265,216
2048           4,210,689     16,842,756 B  4,210,689 B                  False                   False            False4,206,592
========================================================================================================================
```

### Exact Cut-off Widths for 300 KB SRAM (307,200 Bytes)
For architecture $5 \to w \to w \to 1$:
$$\text{Parameters}(w) = w^2 + 8w + 1$$

1. **Float32 Cut-off:**
   $$(w^2 + 8w + 1) \times 4 \le 307,200 \implies w^2 + 8w - 76,799 \le 0$$
   Solving the quadratic equation:
   $$w = \frac{-8 + \sqrt{64 + 4(76,799)}}{2} = \frac{-8 + \sqrt{307,260}}{2} = 273.15$$
   **Maximum integer width for Float32:** $\mathbf{w = 273}$ (Takes 306,856 bytes $\le$ 307,200 bytes; $w=274$ takes 309,196 bytes).

2. **Int8 Cut-off:**
   $$(w^2 + 8w + 1) \times 1 \le 307,200 \implies w^2 + 8w - 307,199 \le 0$$
   Solving the quadratic equation:
   $$w = \frac{-8 + \sqrt{64 + 4(307,199)}}{2} = \frac{-8 + \sqrt{1,228,860}}{2} = 550.27$$
   **Maximum integer width for Int8:** $\mathbf{w = 550}$ (Takes 306,901 bytes $\le$ 307,200 bytes; $w=551$ takes 308,010 bytes).

### 7.4 Architectural Trade-off: On-Device Edge vs. Cloud Server

| Decision Criterion | Edge (Run on ESP32-S3) | Cloud (Call FastAPI Server) |
|---|---|---|
| **Latency** | **< 1 ms** (Immediate local computation) | **50–250 ms** (Wi-Fi handshake + TLS + Internet roundtrip) |
| **Privacy & Security** | **Zero leakage** (Sensor telemetry never leaves device) | Telemetry transmitted over network, prone to interception |
| **Power Consumption** | **Ultra-low** (~1.2K MACs takes negligible CPU energy) | **High** (Powering up Wi-Fi radio to TX/RX consumes 100–300 mA) |
| **Offline Resilience** | **100% operational** during Wi-Fi outages or router reboot | **Fails completely** when internet is down |
| **Operational Cost** | **Zero marginal cost** | Cloud server hosting, domain, load balancer upkeep |

**Engineering Verdict:** The smart doorbell should run inference **locally on the ESP32-S3**. Because the model requires only 5 KB of Flash/SRAM and executes in under 1 ms, edge deployment eliminates network latency, guarantees privacy, saves massive battery power by keeping the Wi-Fi radio in deep sleep, and functions flawlessly during network outages.

---

## 8. Artifact Generation & Verification Summary

All deliverables have been created and strictly verified:
- `data/team_01_train.csv`: Active and validated.
- `train.py`: Fully functional with glitch filtering, linear regression baseline, training normalization, and automatic artifact export.
- `model/model.pt`: Validated state dictionary (1,281 parameters).
- `model/config.json`: Formatted to exact API schema specifications.
- `verify_artifacts.py`: Automated integration test suite passing with 100% assertions.
- `MODEL_INTEGRATION_HANDOFF.md`: Full handoff contract prepared for Rakshith, Monica, and Saakshi.

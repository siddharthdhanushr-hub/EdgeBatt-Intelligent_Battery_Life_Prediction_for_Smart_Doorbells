# EdgeBatt: Intelligent Battery Life Prediction for Smart Doorbells

<p align="left">
  <img src="https://img.shields.io/github/contributors/siddharthdhanushr-hub/EdgeBatt-Intelligent_Battery_Life_Prediction_for_Smart_Doorbells?color=2a9d8f&label=Active%20Contributors&style=for-the-badge" alt="Active Contributors" />
  &nbsp;
  <img src="https://img.shields.io/badge/Validation%20MAE-4.64%20days-success?style=for-the-badge" alt="Validation MAE" />
  &nbsp;
  <img src="https://img.shields.io/badge/Team-01%20Paradise-blue?style=for-the-badge" alt="Team 01" />
</p>

**AI255TBD Edge AI · Module 1 Assignment: Neural Networks from Scratch to Deployment**  
**Team 01 — Paradise**  

---

## 👥 Contributors & Team Roles

<p align="center">
  <a href="https://github.com/monicaguru10">
    <img src="https://avatars.githubusercontent.com/u/239998788?v=4" width="75px" style="border-radius:50%;" alt="Monica K G"/>
    <br />
    <sub><b>Monica K G</b></sub>
    <br />
    <sub><code>@monicaguru10</code></sub>
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://github.com/siddharthdhanushr-hub">
    <img src="https://avatars.githubusercontent.com/u/121683115?v=4" width="75px" style="border-radius:50%;" alt="Siddharth Dhanush R"/>
    <br />
    <sub><b>Siddharth Dhanush R</b></sub>
    <br />
    <sub><code>@siddharthdhanushr-hub</code></sub>
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://github.com/WhiteDevil-06">
    <img src="https://avatars.githubusercontent.com/u/179292012?v=4" width="75px" style="border-radius:50%;" alt="Rakshith Raghavendra"/>
    <br />
    <sub><b>Rakshith Raghavendra</b></sub>
    <br />
    <sub><code>@WhiteDevil-06</code></sub>
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="#-contributors--team-roles">
    <img src="https://ui-avatars.com/api/?name=Saakshi+V+P&background=2a9d8f&color=fff&size=150" width="75px" style="border-radius:50%;" alt="Saakshi V P"/>
    <br />
    <sub><b>Saakshi V P</b></sub>
    <br />
    <sub><code>Team 01</code></sub>
  </a>
</p>

| Contributor | Role & Domain | Primary Responsibilities & Key Deliverables |
|---|---|---|
| **Monica K G** | **Integration, Multi-Load Testing & Verification Lead** | • End-to-end integration of Model, API, and Web Dashboard<br>• Multi-load stress testing (concurrency, connection churn, batch scaling, endurance)<br>• Hidden neuron architecture exploration (4, 8, 16, 32, 64) & Pareto frontier analysis<br>• Automated pre-flight instructor verification suite & technical reports |
| **Siddharth Dhanush R** | **EDA & PyTorch Model Development Lead** | • Exploratory Data Analysis (EDA) & feature correlation visualization<br>• Sensor glitch identification & dataset cleaning rule definition<br>• Baseline models (mean baseline & linear regression via OLS)<br>• PyTorch `BatteryNet` model training, normalization & weights serialization |
| **Rakshith Raghavendra** | **FastAPI Backend & Web Dashboard Controller** | • Production FastAPI model serving engine (`app/main.py`)<br>• Sensor bounds validation & glitch rejection (HTTP 422)<br>• Sub-millisecond stage-wise micro-timing (`parse`, `normalise`, `infer`, `build`)<br>• EdgeBatt dark-mode IoT diagnostics web dashboard & SVG battery gauge |
| **Saakshi V P** | **Testing, Benchmarking & Hardware Feasibility Lead** | • Automated API endpoint test suite verification<br>• Outer-loop latency benchmarking (`client/bench.py`)<br>• ESP32-S3 microcontroller SRAM & PSRAM memory profiling<br>• Batch vs. single-prediction numerical determinism audit |

---

## 1. Executive Summary

EdgeBatt is an end-to-end edge-intelligence solution for predicting the remaining operational battery life (in days) of smart video doorbells. Instead of reporting a crude "Low Battery" warning, our system models non-linear battery degradation as a function of:
- **`battery_v`**: Cell terminal voltage (V)
- **`temp_c`**: Ambient temperature (°C)
- **`rings_per_day`**: Doorbell button activation frequency
- **`wifi_rssi_dbm`**: Received Wi-Fi signal strength indicator (dBm)
- **`age_months`**: Accumulated battery service age (months)

Our production deployment consists of:
1. **Lightweight PyTorch Neural Network (`BatteryNet`, 5 → 32 → 32 → 1)**: Trained with strict zero data-leakage normalization on clean team telemetry, achieving a validation MAE of **4.64 days** (drastically outperforming baseline 1's **42.85 days** and linear regression's **18.63 days**).
2. **Production FastAPI Service (`app/main.py`)**: Sub-millisecond latency ($0.42\text{ ms}$ median server execution), single-threaded deterministic inference, non-overlapping stage timing metadata (`parse`, `normalise`, `inference`, `build`, `server_total`), and strict physical sensor glitch trapping (HTTP 422).
3. **Interactive IoT Diagnostics Web Dashboard (`static/`)**: Real-time telemetry beacon, SVG battery gauge, latency waterfall charts, scenario presets, and batch fleet tester.
4. **Complete Edge AI Hardware Feasibility Analysis**: Validating that the model consumes only **5.12 KB**, utilizing just **1.67%** of the 300 KB SRAM budget on an **ESP32-S3** microcontroller.

---

## 2. API Contract & Endpoint Specification

| Endpoint | Method | Input Payload | Output Shape / Response |
|---|---|---|---|
| **`/health`** | `GET` | *None* | `{"status": "ok", "team": 1, "params": 1281, "val_mae": 4.64}` |
| **`/model_info`** | `GET` | *None* | Dynamic per-layer specs (params, float32 bytes, int8 bytes, MACs, and 300 KB SRAM fit) |
| **`/predict`** | `POST` | `{"battery_v": 3.86, "temp_c": 28.0, "rings_per_day": 5, "wifi_rssi_dbm": -66, "age_months": 22}` | `{"days_left": 113.71, "timings_ms": {"parse": .., "normalise": .., "inference": .., "build": .., "server_total": ..}}` |
| **`/predict_batch`** | `POST` | `{"rows": [Reading, ...]}` (1 to 10,000 items) | `{"days_left": [113.71, ...], "timings_ms": {...}}` |
| **Glitch Refusal** | `POST` | Disconnect voltage, $-127^\circ\text{C}$, $0\text{ dBm}$, negative counts | **HTTP 422 Unprocessable Entity** |

---

## 3. Architecture Exploration: Finding the Smallest Model

We evaluated candidate architectures across single-layer and two-layer configurations to explore the trade-off between model capacity and prediction accuracy:

| Purpose / Role | Architecture | Params | Float32 Footprint | MACs | Val MAE | Overfit Gap | Fits 300KB SRAM? |
|---|---|---|---|---|---|---|---|
| **Very small model** | `5 -> 4 -> 1` | 29 | 116 B | 24 | 10.59 d | +0.28 d | YES (0.04%) |
| **Small model** | `5 -> 8 -> 1` | 57 | 228 B | 48 | 14.99 d | +0.06 d | YES (0.07%) |
| **Compact model** | `5 -> 16 -> 1` | 113 | 452 B | 96 | 15.08 d | +0.12 d | YES (0.15%) |
| **Medium model** | `5 -> 32 -> 1` | 225 | 900 B | 192 | 9.35 d | +0.33 d | YES (0.29%) |
| **Larger single-hidden** | `5 -> 64 -> 1` | 449 | 1,796 B | 384 | 15.02 d | +0.13 d | YES (0.58%) |
| **Ultra-compact two-layer** | `5 -> 4 -> 4 -> 1` | 49 | 196 B | 40 | 8.69 d | +0.36 d | YES (0.06%) |
| **Small two-layer (Micro Sweet Spot)** | `5 -> 8 -> 8 -> 1` | **129** | **516 B** | **112** | **5.67 d** | +1.04 d | **YES (0.17%)** |
| **Compact two-layer (Balanced Sweet Spot)** | `5 -> 16 -> 16 -> 1` | **385** | **1,540 B** | **352** | **4.96 d** | +0.75 d | **YES (0.50%)** |
| **Current default baseline** | `5 -> 32 -> 32 -> 1` | **1,281** | **5,124 B** | **1,216** | **4.56 - 4.64 d** | +0.96 d | **YES (1.67%)** |
| **Larger two-layer** | `5 -> 64 -> 64 -> 1` | 4,609 | 18,436 B | 4,480 | 4.63 d | +1.54 d | YES (6.00%) |

### Key Architectural Takeaways:
1. **Single Hidden Layers Hit a Representational Ceiling**: Due to the compound non-linear coupling of lithium voltage discharge plateaus and temperature-dependent internal impedance ($R_{\text{int}}(T)$), single-layer models cannot model multi-variable cross-terms and stall around 9.35–15.08 days.
2. **Two Hidden Layers Unlock Curved Polyhedra**: Composing two ReLUs allows `5 -> 8 -> 8 -> 1` to immediately cut error to **5.67 days** with only **129 parameters** (89.9% fewer parameters than default).
3. **The Compressed Champion (`5 -> 16 -> 16 -> 1`)**: Achieves **4.96 days** MAE with only **385 parameters** (70% parameter reduction with negligible error difference).
4. **Production Baseline (`5 -> 32 -> 32 -> 1`)**: Delivered the lowest validation error (**4.64 days**) while still taking only **1.67%** of microcontroller SRAM.

---

## 4. Multi-Load & Benchmark Verification

Our comprehensive integration test suite (`load_and_integration_test.py`) validated the deployment across 1,700+ requests with **zero dropped connections and zero 5xx errors**:

### Sequential Latency Profile (N = 200 requests)
- **Throughput:** **322.61 req/s**
- **Median Server Total (`server_total`):** **0.423 ms**
- **Median Inference Time (`inference`):** **0.256 ms**
- **Median Round Trip:** **3.116 ms** (Network & framework overhead accounts for ~86% of total latency).

### Protocol Churn Analysis
- **HTTP Keep-Alive p50:** **3.116 ms**
- **Non-Keep-Alive p50:** **4.378 ms**
- **TCP Handshake / Teardown Overhead:** **+1.262 ms** (40.5% overhead per request).

### High Concurrency Burst Performance
- **5 Concurrent Workers:** 510.57 req/s | p50: 9.04 ms | p95: 13.73 ms | 0 errors
- **10 Concurrent Workers:** 425.85 req/s | p50: 19.59 ms | p95: 41.14 ms | 0 errors
- **25 Concurrent Workers:** 449.44 req/s | p50: 43.22 ms | p95: 105.41 ms | 0 errors
- **50 Concurrent Workers:** 342.93 req/s | p50: 141.26 ms | p95: 204.94 ms | 0 errors

### Batch Vectorization Efficiency
- **Batch 1:** Compute cost = $297.0\,\mu\text{s / reading}$
- **Batch 50:** Compute cost = $4.82\,\mu\text{s / reading}$ (**61.6x faster**)
- **Batch 5,000:** Compute cost = **$0.21\,\mu\text{s / reading}$** (**1,421x faster amortized compute**)

---

## 5. Mission 4: Edge AI Hardware Feasibility (ESP32-S3)

### Hardware Specs:
- **Target MCU:** ESP32-S3 (Xtensa dual-core 32-bit LX7 @ 240 MHz)
- **Memory:** 512 KB SRAM (~300 KB free for user application), 8 MB PSRAM, 16 MB Flash

### Width Scaling Table ($5 \rightarrow w \rightarrow w \rightarrow 1$):
Parameters = $w^2 + 8w + 1$, MACs = $w^2 + 6w$:

| Width ($w$) | Parameters | Float32 Size | Int8 Size | Fits 300 KB SRAM (F32)? | Fits 300 KB SRAM (Int8)? | Fits 8 MB PSRAM? | MACs |
|---|---|---|---|---|---|---|---|
| **32** | 1,281 | 5,124 B (~5.0 KB) | 1,281 B (~1.25 KB) | **YES** (1.67%) | **YES** (0.42%) | **YES** | 1,216 |
| **128** | 17,409 | 69,636 B (~68.0 KB) | 17,409 B (~17.0 KB) | **YES** (22.67%) | **YES** (5.67%) | **YES** | 17,152 |
| **256** | 67,585 | 270,340 B (~264.0 KB)| 67,585 B (~66.0 KB) | **YES** (88.00%) | **YES** (22.00%) | **YES** | 67,072 |
| **512** | 266,241 | 1,064,964 B (~1.01 MB)| 266,241 B (~260.0 KB)| **NO** | **YES** (86.67%) | **YES** | 265,216 |
| **2,048** | 4,210,689 | 16,842,756 B (~16.06 MB)| 4,210,689 B (~4.02 MB)| **NO** | **NO** | **NO** (F32) | 4,206,592 |

### Exact SRAM Cutoff Widths:
- **Float32 Maximum Width in 300 KB SRAM:** **$w = 273$** (306,920 bytes $\le$ 307,200 bytes; $w=274$ exceeds).
- **Int8 Maximum Width in 300 KB SRAM:** **$w = 550$** (306,901 bytes $\le$ 307,200 bytes; $w=551$ exceeds).

### On-Device Inference Speed Estimation:
- The ESP32-S3 runs at 240 MHz (240 million clock cycles per second) with Xtensa SIMD/DSP instructions capable of executing approximately 1 Multiply-Accumulate per cycle.
- For our model ($1,216\text{ MACs}$):
  $$\text{Execution Time} \approx \frac{1,216\text{ MACs}}{240 \times 10^6\text{ cycles/sec}} \approx 5.07\,\mu\text{s}$$
  Even allowing for memory access and activation overheads, on-device inference completes in **under $15\,\mu\text{s}$**—over **200 times faster** than calling a local network server ($3.1\text{ ms}$).

### Edge vs. Cloud: Should the Doorbell Run On-Device?
**The doorbell should run the model directly on the device:**
1. **Latency:** Local inference ($<15\,\mu\text{s}$) is instantaneous, compared to round-trip cloud API latency ($3\text{--}50\text{ ms}$).
2. **Offline Reliability:** Battery estimation works continuously even if home Wi-Fi drops or during power outages.
3. **Privacy:** Sensor telemetry (rings per day, occupancy indicators) never leaves the doorbell.
4. **Power & Battery Conservation:** Running an inference on MCU consumes microjoules, whereas waking the Wi-Fi radio to transmit an HTTP request consumes tens of milliwatts ($150\text{--}300\text{ mA}$).
5. **Infrastructure Cost:** Zero server hosting, cloud billing, or endpoint maintenance fees.

---

## 6. How to Run & Verify

### Start the Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
- Open Web Dashboard: `http://localhost:8000/`
- Open API Documentation: `http://localhost:8000/docs`

### Run Pre-Flight Submission Verification
```bash
python preflight_submission_check.py http://127.0.0.1:8000
```

### Run Comprehensive Multi-Load Suite
```bash
python load_and_integration_test.py --port 8000
```

### Run Benchmark Suite
```bash
python client/bench.py --url http://127.0.0.1:8000 --n 50
```

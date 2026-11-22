# FastAPI & Website Integration Handoff — EDGI AI | Team 01 (Paradise)
**From:** Person 2 — Rakshith (FastAPI, Model Serving & Website Development)  
**To:** Person 3 — Monica (Integration & Deployment), Person 4 — Saakshi (Testing & Verification)  
**Branch:** `feature/rakshith-web-api`  
**Date:** 2026-11-22  

---

## 1. Executive Summary & Deliverables

All responsibilities for **Person 2 (Rakshith)** are complete and empirically verified:

1. **FastAPI Model Serving Engine (`app/main.py`)**:
   - Reconstructs Sid's exact PyTorch `BatteryNet` (`5 -> 32 -> 32 -> 1`) using `model/config.json` and `model/model.pt`.
   - Single-threaded inference (`torch.set_num_threads(1)`) & `torch.no_grad()` for deterministic microsecond latency.
   - Normalization using exact training statistics ($\mu$ and $\sigma$) with strict feature ordering.
2. **Dynamic Layer & SRAM Inspection (`GET /model_info`)**:
   - Dynamically inspects `linear1`, `linear2`, and `linear3`.
   - Computes parameters (1,281), Float32 footprint (5,124 B), Int8 footprint (1,281 B), and MACs (1,216).
   - Proves model easily fits within the $300\text{ KB}$ SRAM budget ($1.67\%$ utilized).
3. **Physical Sensor Input Validation (HTTP 422)**:
   - Rejects sensor disconnect glitches (`battery_v <= 0`, `temp_c == -127`, `wifi_rssi_dbm == 0`, negative rings/age).
4. **Stage-Wise Performance Micro-Timing**:
   - Non-overlapping microsecond measurement: `parse`, `normalise`, `inference`, `build`, `server_total`.
5. **Single & Vectorized Batch Predictions (`POST /predict`, `POST /predict_batch`)**:
   - Non-negative days guarantee (`np.maximum(y, 0)`), 2 decimal precision.
   - Batch bounds enforcement ($1 \le N \le 10,000$).
6. **EdgeBatt Web Dashboard (`static/`)**:
   - Industrial dark-mode IoT diagnostics website served directly at `http://localhost:8000/`.
   - Real-time `/health` telemetry beacon, dual slider/number inputs, 5 scenario presets, dynamic SVG battery gauge, micro-latency stacked bar chart, and IoT fleet batch tester.

---

## 2. Benchmark Verification Results (`client/bench.py`)

Tested locally against live Uvicorn server ($N = 200$ requests):

### 2.1 Keep-Alive Connection (Normal Operating Mode)
```
first request: 13.0 ms round trip
stage                       p50 ms    p95 ms    p99 ms   (requests 2..200)
round_trip                   3.692     6.396     7.923
parse                        0.044     0.141     0.276
normalise                    0.092     0.213     0.354
inference                    0.251     0.647     0.951
build                        0.031     0.080     0.125
server_total                 0.455     1.106     1.312
network_and_framework        3.164     5.537     6.911
```
- **Median Server Execution (`server_total`)**: **0.455 ms**
- **Median Inference Time**: **0.251 ms**
- **Median Round Trip Latency**: **3.692 ms**

### 2.2 No Keep-Alive Connection (`--no-keepalive`)
```
first request: 13.8 ms round trip
stage                       p50 ms    p95 ms    p99 ms   (requests 2..200)
round_trip                   9.511    26.394    34.230
server_total                 0.843     2.058     3.029
```

---

## 3. How to Run & Verify

### 3.1 Start the Server (Local / Codespaces)
From the repository root:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
- **Interactive Web Dashboard**: Open `http://localhost:8000/` in browser.
- **Interactive Swagger Docs**: Open `http://localhost:8000/docs`.

### 3.2 Run the Automated Backend Test Suite
```bash
python test_api_endpoints.py
```
Verifies all 12 evaluation criteria (Status 200, 422 rejections, batch vs single consistency).

### 3.3 Run Official Assignment Benchmark
```bash
python client/bench.py --url http://localhost:8000 --n 200
```

---

## 4. Integration Guidelines for Monica (Person 3)
1. **Codespaces Port Visibility**:
   - Ensure port `8000` is forwarded and set to **Public** so instructors and external benchmark runs can access it.
2. **Reverse Proxy / CORS**:
   - `CORSMiddleware` is enabled with `allow_origins=["*"]`, allowing any external frontend or client to consume the API.
3. **Static Files**:
   - Static files are located in `/static` and mounted automatically at `/`. Do not rename the `static` directory.

---

## 5. Testing & Verification Checklist for Saakshi (Person 4)
- [x] `GET /health` returns `{ "status": "ok", "team": 1, "params": 1281, "val_mae": 4.64 }`.
- [x] `GET /model_info` dynamically reflects `linear1`, `linear2`, `linear3` (1281 params, fits 300KB SRAM).
- [x] `POST /predict` returns `days_left` and all 5 `timings_ms` keys.
- [x] Faulty readings (`temp_c = -127`, `battery_v <= 0`, `wifi_rssi_dbm = 0`) trigger **HTTP 422**.
- [x] Batch predictions produce numerically identical results to individual predictions.
- [x] `client/bench.py` passes 200 iterations with 0 dropped requests.
- [x] Web dashboard works seamlessly on desktop and mobile viewports.


# Model Integration Handoff — EDGI AI | Team 01 (Paradise)
**From:** Person 1 — Sid (EDA & ML Model Development)  
**To:** Person 2 — Rakshith (FastAPI & Deployment), Person 3 — Monica (Integration), Person 4 — Saakshi (Testing & Verification)  
**Date:** 2026-11-21  

---

## 1. Executive Summary & Core Metrics

| Metric / Specification | Value | Notes |
|---|---|---|
| **Selected Architecture** | `5 -> 32 -> 32 -> 1` | 2 hidden layers with ReLU activations |
| **Validation MAE** | **4.64 days** | Measured on 395 held-out validation readings |
| **Training MAE** | **3.62 days** | Measured on 1,581 clean training readings |
| **Overfitting Gap** | **1.02 days** | Low generalization gap, optimal capacity |
| **Total Parameters** | **1,281** | Matches assignment baseline and `/health` requirement |
| **Float32 Model Size** | **5,124 bytes** (~5.00 KB) | 1,281 weights/biases × 4 bytes |
| **Theoretical Int8 Size** | **1,281 bytes** (~1.25 KB) | 1,281 weights/biases × 1 byte |
| **Multiply-Accumulates (MACs)**| **1,216 MACs** | Per single inference pass |
| **Peak Activation Memory** | **128 bytes** (32 float32) | Maximum working RAM during inference |
| **Fits 300 KB ESP32-S3 SRAM?**| **True** (Float32 & Int8) | Uses only ~1.67% of SRAM program memory |

---

## 2. Feature Ordering & Normalization Contract

> [!IMPORTANT]
> Incoming requests MUST feed features to `BatteryNet` in the exact order listed below. Any transposition will corrupt the predictions.

```python
FEATURES = [
    "battery_v",       # Battery voltage in Volts (e.g., 3.86)
    "temp_c",          # Ambient temperature in °C (e.g., 28.0)
    "rings_per_day",   # Daily doorbell ring events (e.g., 5)
    "wifi_rssi_dbm",   # Wi-Fi RSSI in dBm (e.g., -66)
    "age_months"       # Battery age in months (e.g., 22)
]
```

### Normalization Statistics ($\mu$ and $\sigma$)
Computed strictly from the 1,581 training rows (zero data leakage):

| Feature | Mean ($\mu$) | Standard Deviation ($\sigma$) |
|---|---|---|
| `battery_v` | `3.781155` | `0.247716` |
| `temp_c` | `28.101706` | `6.997720` |
| `rings_per_day` | `7.895636` | `5.608894` |
| `wifi_rssi_dbm` | `-66.160027` | `12.546649` |
| `age_months` | `17.906389` | `10.533046` |

Formula applied before passing to network:
$$x_{\text{normalized}} = \frac{x - \mu}{\sigma}$$

---

## 3. Artifact Locations & File Structure

```
model/
├── model.pt      # Serialized PyTorch state_dict (weights & biases)
└── config.json   # Model architecture, team ID, mu, sd, features, and val_mae
```

### Exact `model/config.json` Content:
```json
{
  "team": 1,
  "hidden": [
    32,
    32
  ],
  "mu": [
    3.7811548709869385,
    28.10170555114746,
    7.895635604858398,
    -66.16002655029297,
    17.906389236450195
  ],
  "sd": [
    0.24771623313426971,
    6.997720241546631,
    5.608894348144531,
    12.546648979187012,
    10.533045768737793
  ],
  "features": [
    "battery_v",
    "temp_c",
    "rings_per_day",
    "wifi_rssi_dbm",
    "age_months"
  ],
  "val_mae": 4.64
}
```

---

## 4. Commands for Reproducibility & Verification

### To Train & Regenerate Artifacts:
```bash
python train.py --team 1 --hidden 32,32 --epochs 1500
```

### To Verify Loading in Python / FastAPI:
```python
import json
import torch
import numpy as np
from app.model import FEATURES, BatteryNet

cfg = json.load(open("model/config.json"))
net = BatteryNet(tuple(cfg["hidden"]))
net.load_state_dict(torch.load("model/model.pt", weights_only=True))
net.eval()
print(f"Loaded Team {cfg['team']} model successfully! Params: {sum(p.numel() for p in net.parameters())}")
```

### Run Automated Artifact Verification Script:
```bash
python verify_artifacts.py
```

---

## 5. Integration Precautions for Rakshith (FastAPI) & Saakshi (Testing)

1. **Non-Negative Days Guarantee:**
   Raw neural network linear outputs can mathematically produce small negative numbers when battery voltage is critically low. Always apply non-negative clipping as in the starter:
   ```python
   days = [round(float(v), 2) for v in np.maximum(y, 0)]
   ```

2. **Sensor Glitch Rejection (HTTP 422):**
   Our EDA confirmed 3 hardware fault conditions that MUST be rejected by Pydantic validators in `app/main.py`:
   - `battery_v <= 0.0` (Physical range: `Field(gt=0, le=5.0)`)
   - `temp_c == -127.0` (Physical range: `Field(ge=-40.0, le=85.0)`)
   - `wifi_rssi_dbm == 0` (Physical range: `Field(ge=-110, lt=0)`)

3. **Single Thread for Deterministic Micro-Latency:**
   Ensure `torch.set_num_threads(1)` remains set in `app/main.py` to prevent CPU context switching overhead during small-matrix inference.

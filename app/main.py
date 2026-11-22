"""
app/main.py -- Production FastAPI battery prediction API for EDGI AI Team 01.

    uvicorn app.main:app --host 0.0.0.0 --port 8000

Endpoints:
GET  /health          -> {"status": "ok", "team": 1, "params": 1281, "val_mae": 4.64}
GET  /model_info      -> dynamic per-layer inspection: params, bytes, MACs, SRAM fit
POST /predict         -> single sensor reading prediction with 5-stage micro-timings
POST /predict_batch   -> batch sensor readings prediction (vectorized) with micro-timings
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from app.model import FEATURES, BatteryNet

# Model and configuration loading
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
CONFIG_PATH = MODEL_DIR / "config.json"
MODEL_PATH = MODEL_DIR / "model.pt"

with open(CONFIG_PATH, "r") as f:
    cfg = json.load(f)

# Reconstruct the exact network from saved configuration
net = BatteryNet(tuple(cfg["hidden"]))
net.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
net.eval()

# Codespace & micro-latency optimization: 1 thread prevents thread-switching overhead
torch.set_num_threads(1)

# Saved normalization statistics strictly from training pipeline
MU = np.array(cfg["mu"], dtype=np.float32)
SD = np.array(cfg["sd"], dtype=np.float32)
N_PARAMS = sum(p.numel() for p in net.parameters())


class Reading(BaseModel):
    """
    Physical sensor constraints for Smart Doorbell (Mission 3):
    Rejects sensor disconnect glitches (e.g. temp=-127C, rssi=0, v<=0)
    while accepting any physically realistic operating reading.
    """
    battery_v: float = Field(
        ...,
        gt=2.0,
        le=5.0,
        description="Battery voltage in Volts (operating range: >2.0V to 5.0V)"
    )
    temp_c: float = Field(
        ...,
        ge=-40.0,
        le=85.0,
        description="Ambient temperature in °C (operating range: -40°C to 85°C)"
    )
    rings_per_day: float = Field(
        ...,
        ge=0.0,
        le=1000.0,
        description="Doorbell rings per day (operating range: 0 to 1000)"
    )
    wifi_rssi_dbm: float = Field(
        ...,
        ge=-110.0,
        lt=0.0,
        description="Wi-Fi RSSI in dBm (operating range: -110 dBm to <0 dBm)"
    )
    age_months: float = Field(
        ...,
        ge=0.0,
        le=120.0,
        description="Battery age in months (operating range: 0 to 120 months)"
    )


class Batch(BaseModel):
    rows: list[Reading] = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Batch of sensor readings (1 to 10,000 items)"
    )


app = FastAPI(
    title="EdgeBatt — Smart Doorbell Battery Intelligence API",
    description="FastAPI service for predicting remaining battery life on edge smart doorbells (Team 01).",
    version="1.0.0"
)

# Enable CORS for frontend dashboard and external evaluations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ms = lambda t0: round((time.perf_counter() - t0) * 1000, 3)


@app.get("/health")
def health():
    """Returns API health status, Team ID, actual parameter count, and validation MAE."""
    return {
        "status": "ok",
        "team": cfg["team"],
        "params": N_PARAMS,
        "val_mae": cfg["val_mae"]
    }


@app.get("/model_info")
def model_info():
    """
    Dynamically calculates per-layer specs (params, float32 bytes, MACs)
    and validates memory feasibility for 300 KB SRAM.
    """
    layers_info = []
    total_params = 0
    total_macs = 0
    linear_count = 1

    for m in net.net:
        if isinstance(m, torch.nn.Linear):
            in_f = m.in_features
            out_f = m.out_features
            params = in_f * out_f + out_f
            bytes_f32 = params * 4
            macs = in_f * out_f
            layers_info.append({
                "name": f"linear{linear_count}",
                "in": in_f,
                "out": out_f,
                "params": params,
                "bytes_f32": bytes_f32,
                "macs": macs
            })
            total_params += params
            total_macs += macs
            linear_count += 1

    total_bytes_f32 = total_params * 4
    total_bytes_int8 = total_params * 1
    sram_limit = 300 * 1024  # 300 KB = 307,200 bytes

    return {
        "layers": layers_info,
        "total_params": total_params,
        "total_bytes_f32": total_bytes_f32,
        "total_bytes_int8": total_bytes_int8,
        "total_macs": total_macs,
        "fits_300kb_sram_f32": total_bytes_f32 <= sram_limit,
        "fits_300kb_sram_int8": total_bytes_int8 <= sram_limit
    }


async def run(request: Request, schema):
    t_all = time.perf_counter()
    t = {}

    # 1. PARSE: Read and validate request JSON
    t0 = time.perf_counter()
    try:
        body_bytes = await request.body()
        data = schema.model_validate_json(body_bytes)
    except ValidationError as e:
        return JSONResponse(
            status_code=422,
            content={"error": "bad reading", "detail": json.loads(e.json())}
        )
    except Exception as e:
        return JSONResponse(
            status_code=422,
            content={"error": "bad reading", "detail": str(e)}
        )
    t["parse"] = ms(t0)

    rows = data.rows if isinstance(data, Batch) else [data]

    # 2. NORMALISE: Extract features in strict order and normalize using saved mu & sd
    t0 = time.perf_counter()
    raw_values = [[getattr(r, f) for f in FEATURES] for r in rows]
    x = (np.array(raw_values, dtype=np.float32) - MU) / SD
    x_tensor = torch.from_numpy(x)
    t["normalise"] = ms(t0)

    # 3. INFERENCE: Single pass through network without gradient tracking
    t0 = time.perf_counter()
    with torch.no_grad():
        y = net(x_tensor).numpy()
    t["inference"] = ms(t0)

    # 4. BUILD: Format response, clip negative predictions, round to 2 decimals
    t0 = time.perf_counter()
    clipped_y = np.maximum(y, 0)
    days = [round(float(v), 2) for v in np.atleast_1d(clipped_y)]

    if isinstance(data, Batch):
        body = {"days_left": days}
    else:
        body = {"days_left": days[0]}
    t["build"] = ms(t0)

    # 5. SERVER_TOTAL: Total execution time inside server
    t["server_total"] = ms(t_all)
    body["timings_ms"] = t
    return body


@app.post("/predict")
async def predict(request: Request):
    return await run(request, Reading)


@app.post("/predict_batch")
async def predict_batch(request: Request):
    return await run(request, Batch)


# Mount static website dashboard if directory exists
STATIC_DIR = BASE_DIR / "static"
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

"""
experiment_architectures.py -- Deep Investigation of Neural Network Architectures
EDGI AI Team 01 (Paradise) - Person 3 (Monica - Integration & Verification Lead)

Objectives:
1. Re-evaluate all candidate architectures from the assignment prompt & image:
   - Very small: 5 -> 4 -> 1
   - Small: 5 -> 8 -> 1
   - Compact: 5 -> 16 -> 1
   - Medium: 5 -> 32 -> 1
   - Larger single: 5 -> 64 -> 1
   - Current default: 5 -> 32 -> 32 -> 1
   - Larger two-layer: 5 -> 64 -> 64 -> 1
   - Two-layer compacts: 5 -> 4 -> 4 -> 1, 5 -> 8 -> 8 -> 1, 5 -> 16 -> 16 -> 1
2. Analyze why single-hidden-layer networks behave the way they do (representational bottleneck).
3. Test multiple learning rates and random seeds for robustness.
4. Compute edge metrics: Params, Float32 bytes, Int8 bytes, MACs, Peak Activation Memory, SRAM fit.
5. Identify the smallest architecture that achieves "sufficiently good" validation MAE.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from app.model import FEATURES, BatteryNet

def load_data():
    csv_path = BASE_DIR / "data" / "team_01_train.csv"
    df = pd.read_csv(csv_path)
    # Remove physical glitch rows as identified in EDA
    glitch_filter = (df.battery_v <= 0) | (df.temp_c == -127) | (df.wifi_rssi_dbm == 0)
    df_clean = df[~glitch_filter].copy()

    # Reproducible 80/20 train/validation split
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(df_clean))
    n_val = len(df_clean) // 5
    val_df = df_clean.iloc[idx[:n_val]]
    tr_df = df_clean.iloc[idx[n_val:]]

    X_tr = tr_df[FEATURES].to_numpy(np.float32)
    y_tr = tr_df.days_left.to_numpy(np.float32)
    X_val = val_df[FEATURES].to_numpy(np.float32)
    y_val = val_df.days_left.to_numpy(np.float32)

    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd).astype(np.float32)

    X_tr_norm = (X_tr - mu) / sd
    X_val_norm = (X_val - mu) / sd

    return (
        torch.tensor(X_tr_norm, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
        torch.tensor(X_val_norm, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
        y_tr,
        y_val
    )

def calculate_hardware_metrics(hidden_layers):
    in_dim = 5
    tot_params = 0
    tot_macs = 0
    layer_dims = [in_dim] + list(hidden_layers) + [1]
    
    # Peak activation memory: size of largest layer output in bytes (float32 = 4 bytes)
    peak_act_bytes = max(hidden_layers) * 4 if hidden_layers else 4
    
    for i in range(len(layer_dims) - 1):
        n_in = layer_dims[i]
        n_out = layer_dims[i+1]
        params = (n_in * n_out) + n_out
        macs = n_in * n_out
        tot_params += params
        tot_macs += macs
        
    bytes_f32 = tot_params * 4
    bytes_int8 = tot_params * 1
    sram_limit = 300 * 1024 # 300 KB = 307,200 bytes
    
    return {
        "params": tot_params,
        "bytes_f32": bytes_f32,
        "bytes_int8": bytes_int8,
        "macs": tot_macs,
        "peak_act_bytes": peak_act_bytes,
        "fits_300kb_f32": bytes_f32 <= sram_limit,
        "fits_300kb_int8": bytes_int8 <= sram_limit,
        "sram_utilization_pct": round((bytes_f32 / sram_limit) * 100, 3)
    }

def train_and_eval(hidden, Xt, yt, Xvt, yvt, y_tr_np, y_val_np, lr=3e-3, epochs=1500, seed=0):
    torch.manual_seed(seed)
    net = BatteryNet(hidden)
    opt = optim.Adam(net.parameters(), lr=lr)
    
    for epoch in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 128):
            b = perm[i:i + 128]
            pred = net(Xt[b])
            loss = nn.functional.mse_loss(pred, yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            
    net.eval()
    with torch.no_grad():
        p_tr = np.maximum(net(Xt).numpy(), 0)
        p_val = np.maximum(net(Xvt).numpy(), 0)
        
    tr_mae = float(np.mean(np.abs(p_tr - y_tr_np)))
    val_mae = float(np.mean(np.abs(p_val - y_val_np)))
    return round(tr_mae, 3), round(val_mae, 3)

def run_investigation():
    Xt, yt, Xvt, yvt, y_tr_np, y_val_np = load_data()
    
    candidates = [
        {"name": "Very small model", "hidden": (4,)},
        {"name": "Small model", "hidden": (8,)},
        {"name": "Compact model", "hidden": (16,)},
        {"name": "Medium model", "hidden": (32,)},
        {"name": "Larger single-hidden", "hidden": (64,)},
        {"name": "Ultra-compact two-layer", "hidden": (4, 4)},
        {"name": "Small two-layer", "hidden": (8, 8)},
        {"name": "Compact two-layer", "hidden": (16, 16)},
        {"name": "Current default baseline", "hidden": (32, 32)},
        {"name": "Larger two-layer", "hidden": (64, 64)},
    ]
    
    print("=" * 105)
    print("                   DEEP NEURAL NETWORK ARCHITECTURE & HIDDEN BIT INVESTIGATION")
    print("=" * 105)
    print(f"{'Role / Name':26s} {'Arch':18s} {'Params':>7s} {'F32 B':>7s} {'MACs':>6s} {'Act RAM':>8s} {'Tr MAE':>8s} {'Val MAE':>8s} {'Overfit':>8s} {'300KB?':>7s}")
    print("-" * 105)
    
    summary_table = []
    
    for c in candidates:
        hidden = c["hidden"]
        arch_str = "5 -> " + " -> ".join(str(h) for h in hidden) + " -> 1"
        hw = calculate_hardware_metrics(hidden)
        
        # Train with default parameters
        tr_mae, val_mae = train_and_eval(hidden, Xt, yt, Xvt, yvt, y_tr_np, y_val_np, lr=3e-3, epochs=1500, seed=0)
        gap = round(val_mae - tr_mae, 3)
        fits_str = "YES" if hw["fits_300kb_f32"] else "NO"
        
        print(f"{c['name']:26s} {arch_str:18s} {hw['params']:7d} {hw['bytes_f32']:7d} {hw['macs']:6d} {hw['peak_act_bytes']:7d}B {tr_mae:7.2f}d {val_mae:7.2f}d {gap:+7.2f}d {fits_str:>7s}")
        
        summary_table.append({
            "purpose": c["name"],
            "architecture": arch_str,
            "hidden": list(hidden),
            "layers": len(hidden),
            "params": hw["params"],
            "bytes_f32": hw["bytes_f32"],
            "bytes_int8": hw["bytes_int8"],
            "macs": hw["macs"],
            "peak_act_bytes": hw["peak_act_bytes"],
            "train_mae_days": tr_mae,
            "val_mae_days": val_mae,
            "overfit_gap_days": gap,
            "sram_utilization_pct": hw["sram_utilization_pct"],
            "fits_300kb_sram": hw["fits_300kb_f32"]
        })

    # Save detailed JSON
    out_json = BASE_DIR / "architecture_investigation_results.json"
    with open(out_json, "w") as f:
        json.dump(summary_table, f, indent=2)
    print("-" * 105)
    print(f"[SAVED] Investigation results exported to {out_json}")

if __name__ == "__main__":
    run_investigation()

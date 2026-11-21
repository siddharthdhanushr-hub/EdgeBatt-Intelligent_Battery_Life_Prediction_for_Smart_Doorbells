import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from app.model import FEATURES, BatteryNet

os.makedirs("eda/plots", exist_ok=True)

df = pd.read_csv("data/team_01_train.csv")
bad = (df.battery_v <= 0) | (df.temp_c == -127) | (df.wifi_rssi_dbm == 0)
df = df[~bad]

rng = np.random.default_rng(0)
idx = rng.permutation(len(df))
n_val = len(df) // 5
val, tr = df.iloc[idx[:n_val]], df.iloc[idx[n_val:]]

X_tr = tr[FEATURES].to_numpy(np.float32)
y_tr = tr.days_left.to_numpy(np.float32)
X_val = val[FEATURES].to_numpy(np.float32)
y_val = val.days_left.to_numpy(np.float32)

mae_fn = lambda p, t: float(np.mean(np.abs(p - t)))

mu = X_tr.mean(axis=0)
sd = X_tr.std(axis=0)
sd = np.where(sd == 0, 1.0, sd).astype(np.float32)

Xt = torch.tensor((X_tr - mu) / sd)
yt = torch.tensor(y_tr)
Xvt = torch.tensor((X_val - mu) / sd)

# Candidate architectures
architectures = [
    # Single hidden layer
    (4,),
    (8,),
    (16,),
    (32,),
    (64,),
    # Two hidden layers
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
    (64, 64)
]

def count_model(hidden):
    layers = []
    n_in = 5
    tot_params = 0
    tot_macs = 0
    for h in hidden:
        params = (n_in + 1) * h
        macs = n_in * h
        tot_params += params
        tot_macs += macs
        n_in = h
    # Output layer
    tot_params += (n_in + 1) * 1
    tot_macs += n_in * 1
    bytes_f32 = tot_params * 4
    bytes_int8 = tot_params * 1
    return tot_params, bytes_f32, bytes_int8, tot_macs

results = []

print("=" * 80)
print(f"{'Arch':18s} {'Params':>8s} {'F32 Bytes':>10s} {'MACs':>8s} {'Tr MAE':>9s} {'Val MAE':>9s} {'Overfit Gap':>12s}")
print("=" * 80)

for hidden in architectures:
    arch_str = "5 -> " + " -> ".join(str(h) for h in hidden) + " -> 1"
    params, b_f32, b_int8, macs = count_model(hidden)
    
    # Train model
    torch.manual_seed(0)
    net = BatteryNet(hidden)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    
    for ep in range(1500):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 128):
            b = perm[i:i + 128]
            loss = torch.nn.functional.mse_loss(net(Xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            
    with torch.no_grad():
        tr_mae = mae_fn(net(Xt).numpy(), y_tr)
        val_mae = mae_fn(net(Xvt).numpy(), y_val)
        
    gap = val_mae - tr_mae
    print(f"{arch_str:18s} {params:8d} {b_f32:10d} {macs:8d} {tr_mae:8.2f}d {val_mae:8.2f}d {gap:11.2f}d")
    
    results.append({
        "architecture": arch_str,
        "hidden": list(hidden),
        "num_layers": len(hidden),
        "params": params,
        "bytes_f32": b_f32,
        "bytes_int8": b_int8,
        "macs": macs,
        "train_mae": round(tr_mae, 3),
        "val_mae": round(val_mae, 3),
        "overfit_gap": round(gap, 3)
    })

res_df = pd.DataFrame(results)
res_df.to_csv("eda/architecture_sweep.csv", index=False)
with open("eda/architecture_sweep.json", "w") as f:
    json.dump(results, f, indent=2)

# Plot 1: Validation MAE vs Parameters
plt.figure(figsize=(10, 6))
single_l = [r for r in results if r["num_layers"] == 1]
double_l = [r for r in results if r["num_layers"] == 2]

plt.plot([r["params"] for r in single_l], [r["val_mae"] for r in single_l], 
         marker='o', lw=2, color='teal', label='1 Hidden Layer (5 -> w -> 1)')
plt.plot([r["params"] for r in double_l], [r["val_mae"] for r in double_l], 
         marker='s', lw=2, color='darkorange', label='2 Hidden Layers (5 -> w -> w -> 1)')

for r in results:
    w_label = str(r["hidden"][0]) if r["num_layers"] == 1 else f"{r['hidden'][0]}x{r['hidden'][1]}"
    plt.annotate(f"w={w_label}\n({r['val_mae']:.2f}d)", 
                 (r["params"], r["val_mae"]), 
                 textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9, fontweight='bold')

plt.xscale('log')
plt.xlabel("Number of Parameters (Log Scale)", fontsize=12)
plt.ylabel("Validation MAE (Days)", fontsize=12)
plt.title("Model Capacity vs Validation MAE (Pareto Trade-off)", fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5, which='both')
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig("eda/plots/07_val_mae_vs_params.png", dpi=200)
plt.close()
print("Saved eda/plots/07_val_mae_vs_params.png")

# Plot 2: Train vs Val MAE (Overfitting diagnostic)
plt.figure(figsize=(10, 6))
labels = [r["architecture"].replace("5 -> ", "").replace(" -> 1", "") for r in results]
x_pos = np.arange(len(labels))
width = 0.35

plt.bar(x_pos - width/2, [r["train_mae"] for r in results], width, label='Train MAE', color='skyblue', edgecolor='black')
plt.bar(x_pos + width/2, [r["val_mae"] for r in results], width, label='Validation MAE', color='coral', edgecolor='black')

plt.xticks(x_pos, labels, rotation=35, ha='right', fontsize=10)
plt.ylabel("MAE (Days)", fontsize=12)
plt.title("Train MAE vs Validation MAE Across Architectures", fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5, axis='y')
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig("eda/plots/08_train_vs_val_overfitting.png", dpi=200)
plt.close()
print("Saved eda/plots/08_train_vs_val_overfitting.png")

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

# Compute mu and sd STRICTLY from training set
mu = X_tr.mean(axis=0)
sd = X_tr.std(axis=0)
# Safe division guard: if sd is zero, default to 1.0
sd = np.where(sd == 0, 1.0, sd).astype(np.float32)

print("=" * 60)
print("TRAINING SET NORMALIZATION PARAMETERS")
print("=" * 60)
for f, m, s in zip(FEATURES, mu, sd):
    print(f"  {f:15s} -> mu: {m:10.4f}, sd: {s:10.4f}")

epochs = 1500
batch_size = 128

# Function to train a model and record history
def train_experiment(normalize=True):
    torch.manual_seed(0)
    net = BatteryNet((32, 32))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    
    if normalize:
        Xt = torch.tensor((X_tr - mu) / sd)
        Xvt = torch.tensor((X_val - mu) / sd)
    else:
        Xt = torch.tensor(X_tr)
        Xvt = torch.tensor(X_val)
    yt = torch.tensor(y_tr)
    
    train_losses = []
    val_maes = []
    
    for ep in range(epochs):
        perm = torch.randperm(len(Xt))
        ep_loss = 0.0
        n_batches = 0
        for i in range(0, len(Xt), batch_size):
            b = perm[i:i + batch_size]
            loss = torch.nn.functional.mse_loss(net(Xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            n_batches += 1
        
        train_losses.append(ep_loss / n_batches)
        if ep % 50 == 0 or ep == epochs - 1:
            with torch.no_grad():
                v_mae = mae_fn(net(Xvt).numpy(), y_val)
                val_maes.append((ep, v_mae))
                
    with torch.no_grad():
        final_val_mae = mae_fn(net(Xvt).numpy(), y_val)
        final_tr_mae = mae_fn(net(Xt).numpy(), y_tr)
        
    return net, train_losses, val_maes, final_tr_mae, final_val_mae

print("\nTraining unnormalized model...")
net_raw, losses_raw, val_maes_raw, tr_mae_raw, val_mae_raw = train_experiment(normalize=False)
print(f"Unnormalized -> Train MAE: {tr_mae_raw:.2f} days, Val MAE: {val_mae_raw:.2f} days")

print("\nTraining normalized model...")
net_norm, losses_norm, val_maes_norm, tr_mae_norm, val_mae_norm = train_experiment(normalize=True)
print(f"Normalized   -> Train MAE: {tr_mae_norm:.2f} days, Val MAE: {val_mae_norm:.2f} days")

# Save comparison summary
norm_summary = {
    "normalization_stats": {f: {"mu": float(m), "sd": float(s)} for f, m, s in zip(FEATURES, mu, sd)},
    "unnormalized": {
        "train_mae": float(tr_mae_raw),
        "val_mae": float(val_mae_raw)
    },
    "normalized": {
        "train_mae": float(tr_mae_norm),
        "val_mae": float(val_mae_norm)
    }
}
with open("eda/normalization_summary.json", "w") as f:
    json.dump(norm_summary, f, indent=2)

# Plot comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Subplot 1: Training Loss (log scale)
ax1.plot(losses_raw, label="Unnormalized", color="crimson", lw=1.5, alpha=0.8)
ax1.plot(losses_norm, label="Normalized", color="teal", lw=1.5, alpha=0.8)
ax1.set_yscale("log")
ax1.set_xlabel("Epoch", fontsize=12)
ax1.set_ylabel("MSE Loss (Log Scale)", fontsize=12)
ax1.set_title("Training Loss Convergence", fontsize=13, fontweight='bold')
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(fontsize=11)

# Subplot 2: Validation MAE across epochs
ep_raw, mae_r = zip(*val_maes_raw)
ep_norm, mae_n = zip(*val_maes_norm)
ax2.plot(ep_raw, mae_r, label=f"Unnormalized (Final: {val_mae_raw:.2f}d)", color="crimson", lw=2, marker='o', markersize=3)
ax2.plot(ep_norm, mae_n, label=f"Normalized (Final: {val_mae_norm:.2f}d)", color="teal", lw=2, marker='s', markersize=3)
ax2.set_xlabel("Epoch", fontsize=12)
ax2.set_ylabel("Validation MAE (Days)", fontsize=12)
ax2.set_title("Validation MAE Progression", fontsize=13, fontweight='bold')
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(fontsize=11)

plt.tight_layout()
plt.savefig("eda/plots/06_normalization_comparison.png", dpi=200)
plt.close()
print("Saved eda/plots/06_normalization_comparison.png")

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("eda/plots", exist_ok=True)

# 1. Load data
df = pd.read_csv("data/team_01_train.csv")
bad = (df.battery_v <= 0) | (df.temp_c == -127) | (df.wifi_rssi_dbm == 0)
df = df[~bad]

# 2. Reproducible split (seed 0, 80/20)
FEATURES = ["battery_v", "temp_c", "rings_per_day", "wifi_rssi_dbm", "age_months"]
rng = np.random.default_rng(0)
idx = rng.permutation(len(df))
n_val = len(df) // 5
val, tr = df.iloc[idx[:n_val]], df.iloc[idx[n_val:]]

X_tr, y_tr = tr[FEATURES].to_numpy(np.float32), tr.days_left.to_numpy(np.float32)
X_val, y_val = val[FEATURES].to_numpy(np.float32), val.days_left.to_numpy(np.float32)

mae_fn = lambda p, t: float(np.mean(np.abs(p - t)))

print(f"Total clean rows: {len(df)}")
print(f"Training set: {len(tr)} rows (80%)")
print(f"Validation set: {len(val)} rows (20%)")

# Baseline 1: Predict Training Mean
y_mean_tr = y_tr.mean()
pred_b1 = np.full_like(y_val, y_mean_tr)
mae_b1 = mae_fn(pred_b1, y_val)
print(f"\n--- Baseline 1: Mean Predictor ---")
print(f"Training target mean: {y_mean_tr:.2f} days")
print(f"Validation MAE (Baseline 1): {mae_b1:.2f} days")

# Baseline 2: Linear Regression with np.linalg.lstsq
# Augment X with column of ones for intercept
A_tr = np.hstack([X_tr, np.ones((len(X_tr), 1), dtype=np.float32)])
A_val = np.hstack([X_val, np.ones((len(X_val), 1), dtype=np.float32)])

# Solve least squares: A * theta = y
theta, residuals, rank, s = np.linalg.lstsq(A_tr, y_tr, rcond=None)

pred_b2 = A_val @ theta
# Physical constraint: days_left >= 0
pred_b2_clipped = np.maximum(pred_b2, 0)
mae_b2 = mae_fn(pred_b2, y_val)
mae_b2_clipped = mae_fn(pred_b2_clipped, y_val)

print(f"\n--- Baseline 2: Linear Regression (np.linalg.lstsq) ---")
print("Learned coefficients:")
for f, w in zip(FEATURES, theta[:5]):
    print(f"  {f:15s}: {w:10.4f}")
print(f"  {'intercept':15s}: {theta[5]:10.4f}")
print(f"Validation MAE (Raw Linear): {mae_b2:.2f} days")
print(f"Validation MAE (Clipped >= 0): {mae_b2_clipped:.2f} days")

# Save baselines summary
baselines_res = {
    "train_rows": len(tr),
    "val_rows": len(val),
    "baseline1_mean_prediction": float(y_mean_tr),
    "baseline1_val_mae": float(mae_b1),
    "baseline2_coefficients": {f: float(w) for f, w in zip(FEATURES, theta[:5])},
    "baseline2_intercept": float(theta[5]),
    "baseline2_val_mae": float(mae_b2),
    "baseline2_val_mae_clipped": float(mae_b2_clipped)
}
with open("eda/baselines_summary.json", "w") as f:
    json.dump(baselines_res, f, indent=2)

# Plot Actual vs Predicted for Linear Regression
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Actual vs Predicted
ax1.scatter(y_val, pred_b2, alpha=0.5, color='darkblue', edgecolors='none', label='Predictions')
lims = [0, max(y_val.max(), pred_b2.max()) + 10]
ax1.plot(lims, lims, 'r--', label='Ideal 1:1 Line', lw=2)
ax1.set_xlim(lims)
ax1.set_ylim(lims)
ax1.set_xlabel("Actual days_left", fontsize=12)
ax1.set_ylabel("Predicted days_left (Linear)", fontsize=12)
ax1.set_title(f"Baseline 2: Linear Regression (Val MAE = {mae_b2:.2f} days)", fontsize=13, fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend()

# Subplot 2: Residuals vs Actual
residuals_val = pred_b2 - y_val
ax2.scatter(y_val, residuals_val, alpha=0.5, color='crimson', edgecolors='none')
ax2.axhline(0, color='black', linestyle='--', lw=1.5)
ax2.set_xlabel("Actual days_left", fontsize=12)
ax2.set_ylabel("Residual (Predicted - Actual)", fontsize=12)
ax2.set_title("Residual Plot (Linear Regression)", fontsize=13, fontweight='bold')
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("eda/plots/05_linear_baseline_actual_vs_pred.png", dpi=200)
plt.close()
print("Saved eda/plots/05_linear_baseline_actual_vs_pred.png")

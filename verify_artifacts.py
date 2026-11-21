import json
import numpy as np
import torch
from app.model import FEATURES, BatteryNet

print("=" * 60)
print("VERIFYING MODEL ARTIFACTS & INTEGRATION CONTRACT")
print("=" * 60)

# 1. Verify config.json
with open("model/config.json") as f:
    cfg = json.load(f)

print("config.json keys:", list(cfg.keys()))
assert cfg["team"] == 1, f"Expected team 1, got {cfg['team']}"
assert cfg["hidden"] == [32, 32], f"Expected hidden [32, 32], got {cfg['hidden']}"
assert cfg["features"] == FEATURES, f"Features mismatch: {cfg['features']} vs {FEATURES}"
assert len(cfg["mu"]) == 5, f"Expected 5 mu values, got {len(cfg['mu'])}"
assert len(cfg["sd"]) == 5, f"Expected 5 sd values, got {len(cfg['sd'])}"
print("config.json schema validation: PASSED")
print(f"Validation MAE recorded: {cfg['val_mae']} days")

# 2. Verify model.pt loading
net = BatteryNet(tuple(cfg["hidden"]))
state_dict = torch.load("model/model.pt", weights_only=True)
net.load_state_dict(state_dict)
net.eval()
print("model.pt weights loading: PASSED")

# 3. Verify parameter count
n_params = sum(p.numel() for p in net.parameters())
assert n_params == 1281, f"Expected 1281 params, got {n_params}"
print(f"Total parameters: {n_params} (Exact match: PASSED)")

# 4. Numerical test sample
test_reading = {
    "battery_v": 3.86,
    "temp_c": 28.0,
    "rings_per_day": 5,
    "wifi_rssi_dbm": -66,
    "age_months": 22
}
raw_x = np.array([[test_reading[f] for f in FEATURES]], dtype=np.float32)
mu = np.array(cfg["mu"], dtype=np.float32)
sd = np.array(cfg["sd"], dtype=np.float32)
norm_x = (raw_x - mu) / sd

with torch.no_grad():
    pred = net(torch.from_numpy(norm_x)).numpy()
pred_clipped = round(float(np.maximum(pred, 0)[0]), 2)
print(f"Sample prediction for {test_reading}:")
print(f"  -> Predicted days_left: {pred_clipped} days")
print("Integration contract verification: 100% READY FOR FASTAPI")

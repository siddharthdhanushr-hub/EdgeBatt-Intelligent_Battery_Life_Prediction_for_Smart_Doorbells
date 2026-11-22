import json
import matplotlib.pyplot as plt

data = json.load(open("repo/architecture_investigation_results.json"))

params = [d["params"] for d in data]
val_mae = [d["val_mae_days"] for d in data]

plt.figure(figsize=(11, 6))

# Single-layer points vs Two-layer points
single_p = [d["params"] for d in data if d["layers"] == 1]
single_mae = [d["val_mae_days"] for d in data if d["layers"] == 1]

double_p = [d["params"] for d in data if d["layers"] == 2]
double_mae = [d["val_mae_days"] for d in data if d["layers"] == 2]

plt.scatter(single_p, single_mae, color="#e63946", s=110, label="Single Hidden Layer (5 -> w -> 1)", zorder=5)
plt.scatter(double_p, double_mae, color="#2a9d8f", s=120, marker="^", label="Two Hidden Layers (5 -> w -> w -> 1)", zorder=5)

for d in data:
    lbl = f"{d['architecture']} ({d['val_mae_days']}d)"
    offset = (8, 4) if d["layers"] == 2 else (8, -10)
    plt.annotate(lbl, (d["params"], d["val_mae_days"]), textcoords="offset points", xytext=offset, fontsize=8, fontweight="bold" if d["params"] in [129, 385, 1281] else "normal")

plt.axhline(18.63, color="#457b9d", linestyle="--", linewidth=1.5, label="Baseline 2: Linear Regression (18.63d)")
plt.axhline(4.64, color="#e76f51", linestyle=":", linewidth=1.8, label="Assignment Target Baseline (5->32->32->1: 4.64d)")

plt.xscale("log")
plt.xlabel("Model Parameters (log scale)", fontsize=11, fontweight="bold")
plt.ylabel("Validation MAE (days left)", fontsize=11, fontweight="bold")
plt.title("Pareto Frontier: Neural Network Architecture & Hidden Bit Investigation\nAccuracy vs Model Complexity (Team 01)", fontsize=13, fontweight="bold")
plt.grid(True, which="both", linestyle="--", alpha=0.4)
plt.legend(frameon=True, facecolor="#f8f9fa")
plt.tight_layout()

out_path = "repo/eda/plots/09_architecture_comparison_pareto.png"
plt.savefig(out_path, dpi=200)
print(f"Pareto plot successfully saved to {out_path}")

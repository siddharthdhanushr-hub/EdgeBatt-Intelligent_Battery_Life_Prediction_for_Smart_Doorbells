import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("eda/plots", exist_ok=True)

# Load ONLY team_01_train.csv
csv_path = "data/team_01_train.csv"
df = pd.read_csv(csv_path)

print("=" * 60)
print("1. DATASET SHAPE & STRUCTURE")
print("=" * 60)
print(f"Dataset path: {csv_path}")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(f"Columns: {list(df.columns)}")
print("\nData Types:")
print(df.dtypes)
print("\nMissing Values per column:")
print(df.isnull().sum())
print(f"\nDuplicate Rows: {df.duplicated().sum()}")

print("\n" + "=" * 60)
print("2. DESCRIPTIVE STATISTICS (RAW DATA)")
print("=" * 60)
stats = df.describe().T
stats['median'] = df.median()
stats['IQR'] = stats['75%'] - stats['25%']
print(stats[['count', 'mean', 'std', 'min', '25%', 'median', '75%', 'max', 'IQR']])

print("\n" + "=" * 60)
print("3. SENSOR GLITCH INVESTIGATION")
print("=" * 60)
# Candidate glitch conditions
c_v0 = df['battery_v'] == 0
c_t127 = df['temp_c'] == -127
c_rssi0 = df['wifi_rssi_dbm'] == 0

print(f"Rows with battery_v == 0: {c_v0.sum()}")
print(f"Rows with temp_c == -127: {c_t127.sum()}")
print(f"Rows with wifi_rssi_dbm == 0: {c_rssi0.sum()}")

overlap_v_t = (c_v0 & c_t127).sum()
overlap_v_rssi = (c_v0 & c_rssi0).sum()
overlap_t_rssi = (c_t127 & c_rssi0).sum()
overlap_all = (c_v0 & c_t127 & c_rssi0).sum()

print(f"Overlap (battery_v == 0 AND temp_c == -127): {overlap_v_t}")
print(f"Overlap (battery_v == 0 AND wifi_rssi_dbm == 0): {overlap_v_rssi}")
print(f"Overlap (temp_c == -127 AND wifi_rssi_dbm == 0): {overlap_t_rssi}")
print(f"Overlap (All three simultaneously): {overlap_all}")

union_bad = c_v0 | c_t127 | c_rssi0
print(f"Total Unique Glitch Rows (union): {union_bad.sum()}")

# Print glitch sample rows
print("\nSample Glitch Rows:")
print(df[union_bad].head(10))

# Check legitimate edge cases:
print("\n" + "=" * 60)
print("4. LEGITIMATE EDGE CASES (NOT GLITCHES)")
print("=" * 60)
print(f"Rows with rings_per_day == 0: {(df['rings_per_day'] == 0).sum()} (Quiet day / vacation home)")
print(f"Rows with age_months == 0: {(df['age_months'] == 0).sum()} (Brand new battery)")
print(f"Rows with high temp_c (> 40C): {(df['temp_c'] > 40).sum()} (Hot summer sunny day)")
print(f"Rows with high rings_per_day (> 20): {(df['rings_per_day'] > 20).sum()} (Busy household / deliveries)")

# Create Cleaned Dataset
df_clean = df[~union_bad].copy()
print(f"\nCleaned dataset rows: {len(df_clean)} (Dropped {union_bad.sum()} rows)")

print("\n" + "=" * 60)
print("5. DESCRIPTIVE STATISTICS (CLEANED DATA)")
print("=" * 60)
clean_stats = df_clean.describe().T
clean_stats['median'] = df_clean.median()
clean_stats['IQR'] = clean_stats['75%'] - clean_stats['25%']
print(clean_stats[['count', 'mean', 'std', 'min', '25%', 'median', '75%', 'max', 'IQR']])

print("\n" + "=" * 60)
print("6. CORRELATION MATRIX (CLEANED DATA)")
print("=" * 60)
corr = df_clean.corr()
print(corr)

# Save summary stats to json
summary = {
    "raw_rows": int(len(df)),
    "glitch_rows": int(union_bad.sum()),
    "clean_rows": int(len(df_clean)),
    "glitch_breakdown": {
        "battery_v_zero": int(c_v0.sum()),
        "temp_c_minus_127": int(c_t127.sum()),
        "wifi_rssi_zero": int(c_rssi0.sum()),
        "unique_union": int(union_bad.sum())
    },
    "correlation_with_days_left": corr['days_left'].to_dict()
}
with open("eda/eda_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 60)
print("7. GENERATING VISUALIZATIONS")
print("=" * 60)
features = ['battery_v', 'temp_c', 'rings_per_day', 'wifi_rssi_dbm', 'age_months', 'days_left']

# 1. Histograms for raw vs cleaned
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
for i, col in enumerate(features):
    axes[i].hist(df_clean[col], bins=30, color='skyblue', edgecolor='black', alpha=0.7, label='Clean')
    axes[i].set_title(f"Distribution of {col}", fontsize=12, fontweight='bold')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Count")
    axes[i].grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("eda/plots/01_feature_histograms.png", dpi=200)
plt.close()
print("Saved eda/plots/01_feature_histograms.png")

# 2. Scatter plots vs days_left
inputs = ['battery_v', 'temp_c', 'rings_per_day', 'wifi_rssi_dbm', 'age_months']
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
for i, col in enumerate(inputs):
    axes[i].scatter(df_clean[col], df_clean['days_left'], alpha=0.35, edgecolors='none', color='teal')
    axes[i].set_title(f"{col} vs days_left", fontsize=12, fontweight='bold')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("days_left")
    axes[i].grid(True, linestyle='--', alpha=0.5)
# Hide the 6th subplot
axes[5].axis('off')
plt.tight_layout()
plt.savefig("eda/plots/02_scatter_vs_days_left.png", dpi=200)
plt.close()
print("Saved eda/plots/02_scatter_vs_days_left.png")

# 3. Correlation Heatmap
fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha='left')
ax.set_yticklabels(corr.columns)
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        text = f"{corr.iloc[i, j]:.2f}"
        ax.text(j, i, text, ha='center', va='center', color='black' if abs(corr.iloc[i, j]) < 0.6 else 'white', fontweight='bold')
plt.title("Correlation Matrix (Cleaned Data)", pad=20, fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("eda/plots/03_correlation_heatmap.png", dpi=200)
plt.close()
print("Saved eda/plots/03_correlation_heatmap.png")

# 4. Sensor Glitch Demonstration Plot
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
# battery_v with glitch
axes[0].scatter(range(len(df)), df['battery_v'], c=c_v0.map({True: 'red', False: 'blue'}), alpha=0.6, s=15)
axes[0].set_title("battery_v (Red = 0.0V Glitch)", fontweight='bold')
axes[0].set_ylabel("Volts")
axes[0].grid(True, linestyle='--', alpha=0.5)

# temp_c with glitch
axes[1].scatter(range(len(df)), df['temp_c'], c=c_t127.map({True: 'red', False: 'green'}), alpha=0.6, s=15)
axes[1].set_title("temp_c (Red = -127°C Glitch)", fontweight='bold')
axes[1].set_ylabel("°C")
axes[1].grid(True, linestyle='--', alpha=0.5)

# wifi_rssi with glitch
axes[2].scatter(range(len(df)), df['wifi_rssi_dbm'], c=c_rssi0.map({True: 'red', False: 'purple'}), alpha=0.6, s=15)
axes[2].set_title("wifi_rssi_dbm (Red = 0 dBm Glitch)", fontweight='bold')
axes[2].set_ylabel("dBm")
axes[2].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("eda/plots/04_sensor_glitches.png", dpi=200)
plt.close()
print("Saved eda/plots/04_sensor_glitches.png")
print("\nEDA script execution complete!")

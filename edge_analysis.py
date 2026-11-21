import json
import math

sram_limit_bytes = 300 * 1024 # 307,200 bytes
psram_limit_bytes = 8 * 1024 * 1024 # 8,388,608 bytes

widths = [32, 128, 256, 512, 2048]

def get_2layer_specs(w):
    # 5 -> w -> w -> 1
    # Layer 1: Linear(5, w): 5w + w
    # Layer 2: Linear(w, w): w^2 + w
    # Layer 3: Linear(w, 1): w + 1
    # Total params = w^2 + 8w + 1
    params = w**2 + 8*w + 1
    bytes_f32 = params * 4
    bytes_int8 = params * 1
    # MACs = 5w + w^2 + w = w^2 + 6w
    macs = w**2 + 6*w
    
    fits_sram_f32 = bytes_f32 <= sram_limit_bytes
    fits_sram_int8 = bytes_int8 <= sram_limit_bytes
    fits_psram = bytes_f32 <= psram_limit_bytes
    
    return {
        "w": w,
        "params": params,
        "bytes_f32": bytes_f32,
        "bytes_int8": bytes_int8,
        "fits_300kb_sram_f32": fits_sram_f32,
        "fits_300kb_sram_int8": fits_sram_int8,
        "fits_8mb_psram": fits_psram,
        "macs": macs
    }

table = [get_2layer_specs(w) for w in widths]

print("=" * 105)
print(f"{'Width (w)':10s} {'Params':>12s} {'F32 Bytes':>12s} {'Int8 Bytes':>12s} {'SRAM F32?':>11s} {'SRAM Int8?':>12s} {'PSRAM F32?':>12s} {'MACs':>12s}")
print("=" * 105)
for r in table:
    print(f"{r['w']:<10d} {r['params']:12d} {r['bytes_f32']:12d} {r['bytes_int8']:12d} {str(r['fits_300kb_sram_f32']):>11s} {str(r['fits_300kb_sram_int8']):>12s} {str(r['fits_8mb_psram']):>12s} {r['macs']:12d}")

# Find exact cutoff width where weights stop fitting in 300 KB SRAM:
# For Float32: (w^2 + 8w + 1) * 4 <= 307200 => w^2 + 8w + 1 <= 76800 => w^2 + 8w - 76799 <= 0
# w = (-8 + sqrt(64 - 4(1)(-76799))) / 2 = (-8 + sqrt(64 + 307196)) / 2 = (-8 + sqrt(307260)) / 2 = (-8 + 554.31) / 2 = 273.15
# So max integer width for Float32 in 300KB SRAM is w = 273! (At w=273, params = 76730, bytes = 306,920 <= 307,200; at w=274, bytes = 309,196 > 307,200)

w_f32_max = int((-8 + math.sqrt(64 + 4 * (sram_limit_bytes // 4 - 1))) / 2)
# For Int8: (w^2 + 8w + 1) * 1 <= 307200 => w^2 + 8w - 307199 <= 0
# w = (-8 + sqrt(64 + 4*307199)) / 2 = (-8 + sqrt(64 + 1228796)) / 2 = (-8 + 1108.54) / 2 = 550.27
# So max integer width for Int8 in 300KB SRAM is w = 550!
w_int8_max = int((-8 + math.sqrt(64 + 4 * (sram_limit_bytes - 1))) / 2)

print("\nExact cut-off widths for 300 KB SRAM:")
print(f"Float32 maximum width: w = {w_f32_max} (Weights: {(w_f32_max**2 + 8*w_f32_max + 1)*4:,} bytes <= 307,200 bytes)")
print(f"Int8 maximum width:    w = {w_int8_max} (Weights: {(w_int8_max**2 + 8*w_int8_max + 1):,} bytes <= 307,200 bytes)")

edge_summary = {
    "table": table,
    "max_width_sram_f32": w_f32_max,
    "max_width_sram_int8": w_int8_max,
    "selected_model": {
        "architecture": "5 -> 32 -> 32 -> 1",
        "params": 1281,
        "bytes_f32": 5124,
        "bytes_int8": 1281,
        "macs": 1216,
        "activation_memory_peak_bytes": 128,
        "fits_300kb_sram_f32": True,
        "fits_300kb_sram_int8": True
    }
}

with open("eda/edge_ai_hardware_analysis.json", "w") as f:
    json.dump(edge_summary, f, indent=2)
print("Saved eda/edge_ai_hardware_analysis.json")

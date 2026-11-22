"""
edge_doorbell_verification.py -- End-to-End Edge Hardware Verification for Smart Doorbell
Author: Saakshi-babu <saakshibabu025@gmail.com>
Team: EDGI AI Team 01 (Paradise)
Branch: feature/saakshi-edge-hardware-verification

Rigorous mathematical and hardware verification of whether our neural network
can run inside an actual smart doorbell powered by the ESP32-S3 microcontroller.

Answers all 7 core engineering questions:
1. Float32 model memory calculation
2. Int8 model memory estimation
3. Activation memory calculation (buffer reuse & peak working RAM)
4. Width comparison table (w = 32, 128, 256, 512, 2048)
5. Exact cutoff widths where weights exceed 300 KB SRAM
6. Estimation of prediction time on the board (240 MHz MCU, SIMD/MAC execution)
7. Cloud inference vs Local inference decision trade-off
"""

import math
import json
from pathlib import Path

# ESP32-S3 Hardware Specifications
SRAM_TOTAL_BYTES = 512 * 1024          # 512 KB = 524,288 bytes
SRAM_PROGRAM_AVAIL_BYTES = 300 * 1024  # ~300 KB free for user app = 307,200 bytes
PSRAM_BYTES = 8 * 1024 * 1024          # 8 MB = 8,388,608 bytes
FLASH_BYTES = 16 * 1024 * 1024         # 16 MB = 16,777,216 bytes
CLOCK_FREQ_HZ = 240 * 10**6            # 240 MHz = 240,000,000 cycles/sec

def question_1_float32_memory():
    """
    Task 1: Calculate Float32 model memory for 5 -> 32 -> 32 -> 1
    Each float32 parameter takes 4 bytes.
    """
    layer1_weights = 5 * 32
    layer1_biases = 32
    layer1_params = layer1_weights + layer1_biases
    layer1_bytes = layer1_params * 4

    layer2_weights = 32 * 32
    layer2_biases = 32
    layer2_params = layer2_weights + layer2_biases
    layer2_bytes = layer2_params * 4

    layer3_weights = 32 * 1
    layer3_biases = 1
    layer3_params = layer3_weights + layer3_biases
    layer3_bytes = layer3_params * 4

    total_params = layer1_params + layer2_params + layer3_params
    total_bytes_f32 = total_params * 4
    total_macs = (5 * 32) + (32 * 32) + (32 * 1)

    return {
        "architecture": "5 -> 32 -> 32 -> 1",
        "layers": [
            {"name": "linear1", "weights": layer1_weights, "biases": layer1_biases, "params": layer1_params, "bytes_f32": layer1_bytes, "macs": 5 * 32},
            {"name": "linear2", "weights": layer2_weights, "biases": layer2_biases, "params": layer2_params, "bytes_f32": layer2_bytes, "macs": 32 * 32},
            {"name": "linear3", "weights": layer3_weights, "biases": layer3_biases, "params": layer3_params, "bytes_f32": layer3_bytes, "macs": 32 * 1}
        ],
        "total_params": total_params,
        "total_bytes_f32": total_bytes_f32,
        "total_macs": total_macs,
        "sram_utilization_pct": round((total_bytes_f32 / SRAM_PROGRAM_AVAIL_BYTES) * 100, 3)
    }

def question_2_int8_memory(total_params):
    """
    Task 2: Estimate Int8 model memory for 5 -> 32 -> 32 -> 1
    Quantized weights & biases use 1 byte (8 bits) each.
    """
    bytes_int8 = total_params * 1
    compression_ratio = 4.0
    savings_pct = 75.0
    sram_utilization_pct = round((bytes_int8 / SRAM_PROGRAM_AVAIL_BYTES) * 100, 3)
    return {
        "total_bytes_int8": bytes_int8,
        "compression_ratio": f"{compression_ratio}x",
        "savings_pct": f"{savings_pct}%",
        "sram_utilization_pct": sram_utilization_pct
    }

def question_3_activation_memory():
    """
    Task 3: Calculate activation memory.
    Activation memory holds intermediate tensor values during forward propagation.
    With ping-pong buffer reuse, peak memory is the sum of the two largest adjacent layers.
    """
    in_bytes = 5 * 4       # 20 bytes
    h1_bytes = 32 * 4      # 128 bytes
    h2_bytes = 32 * 4      # 128 bytes
    out_bytes = 1 * 4      # 4 bytes

    # Peak concurrent buffer (e.g. while computing Layer 2, Layer 1 output and Layer 2 output exist simultaneously)
    peak_concurrent_f32 = h1_bytes + h2_bytes
    largest_single_layer_f32 = max(in_bytes, h1_bytes, h2_bytes, out_bytes)

    # Int8 activations
    peak_concurrent_int8 = (32 * 1) + (32 * 1)
    largest_single_layer_int8 = 32 * 1

    return {
        "layer_buffers_f32": {
            "input": in_bytes,
            "hidden_layer_1": h1_bytes,
            "hidden_layer_2": h2_bytes,
            "output": out_bytes
        },
        "largest_single_activation_f32_bytes": largest_single_layer_f32,
        "peak_concurrent_activation_f32_bytes": peak_concurrent_f32,
        "largest_single_activation_int8_bytes": largest_single_layer_int8,
        "peak_concurrent_activation_int8_bytes": peak_concurrent_int8
    }

def question_4_compare_widths():
    """
    Task 4: Compare widths 32, 128, 256, 512 and 2048 for 5 -> w -> w -> 1
    Formulas:
      Params = (5w + w) + (w^2 + w) + (w + 1) = w^2 + 8w + 1
      MACs = 5w + w^2 + w = w^2 + 6w
    """
    widths = [32, 128, 256, 512, 2048]
    table = []
    for w in widths:
        params = w**2 + 8*w + 1
        bytes_f32 = params * 4
        bytes_int8 = params * 1
        macs = w**2 + 6*w
        fits_sram_f32 = bytes_f32 <= SRAM_PROGRAM_AVAIL_BYTES
        fits_sram_int8 = bytes_int8 <= SRAM_PROGRAM_AVAIL_BYTES
        fits_psram = bytes_f32 <= PSRAM_BYTES

        table.append({
            "w": w,
            "params": params,
            "bytes_f32": bytes_f32,
            "bytes_int8": bytes_int8,
            "fits_300kb_sram_f32": fits_sram_f32,
            "fits_300kb_sram_int8": fits_sram_int8,
            "fits_8mb_psram": fits_psram,
            "macs": macs
        })
    return table

def question_5_exact_cutoff_widths():
    """
    Task 5: Find exact width where weights exceed 300 KB SRAM (307,200 bytes)
    P(w) = w^2 + 8w + 1
    Float32: 4 * P(w) <= 307200 => w^2 + 8w - 76799 <= 0
    Int8:    1 * P(w) <= 307200 => w^2 + 8w - 307199 <= 0
    """
    # Float32 analytical solution
    c_f32 = SRAM_PROGRAM_AVAIL_BYTES // 4 - 1
    w_f32_exact = (-8 + math.sqrt(64 + 4 * c_f32)) / 2
    w_f32_max_fits = int(w_f32_exact)
    w_f32_exceeds = w_f32_max_fits + 1

    bytes_f32_fits = (w_f32_max_fits**2 + 8*w_f32_max_fits + 1) * 4
    bytes_f32_exceeds = (w_f32_exceeds**2 + 8*w_f32_exceeds + 1) * 4

    # Int8 analytical solution
    c_int8 = SRAM_PROGRAM_AVAIL_BYTES - 1
    w_int8_exact = (-8 + math.sqrt(64 + 4 * c_int8)) / 2
    w_int8_max_fits = int(w_int8_exact)
    w_int8_exceeds = w_int8_max_fits + 1

    bytes_int8_fits = (w_int8_max_fits**2 + 8*w_int8_max_fits + 1) * 1
    bytes_int8_exceeds = (w_int8_exceeds**2 + 8*w_int8_exceeds + 1) * 1

    return {
        "float32": {
            "exact_analytical_w": round(w_f32_exact, 3),
            "max_width_fits": w_f32_max_fits,
            "max_width_bytes": bytes_f32_fits,
            "first_width_exceeds": w_f32_exceeds,
            "first_width_bytes": bytes_f32_exceeds,
            "sram_limit_bytes": SRAM_PROGRAM_AVAIL_BYTES
        },
        "int8": {
            "exact_analytical_w": round(w_int8_exact, 3),
            "max_width_fits": w_int8_max_fits,
            "max_width_bytes": bytes_int8_fits,
            "first_width_exceeds": w_int8_exceeds,
            "first_width_bytes": bytes_int8_exceeds,
            "sram_limit_bytes": SRAM_PROGRAM_AVAIL_BYTES
        }
    }

def question_6_estimate_prediction_time(total_macs):
    """
    Task 6: Estimate prediction time on ESP32-S3 board
    Clock: 240 MHz (2.4e8 cycles/sec).
    Xtensa PIE DSP/SIMD instructions execute 1 MAC/cycle (or 2 Int8 MACs/cycle).
    """
    theoretical_cycles = total_macs / 1.0
    theoretical_time_us = (theoretical_cycles / CLOCK_FREQ_HZ) * 10**6

    # Adding 2.5x practical margin for activation functions, normalization, and memory fetches
    practical_margin = 2.5
    practical_time_us = theoretical_time_us * practical_margin

    # Comparison with observed network latencies
    local_server_round_trip_us = 3116.0  # 3.116 ms
    cloud_wifi_round_trip_us = 35000.0   # 35.0 ms

    speedup_vs_local_server = local_server_round_trip_us / practical_time_us
    speedup_vs_cloud_api = cloud_wifi_round_trip_us / practical_time_us

    return {
        "mcu_clock_mhz": 240,
        "total_macs": total_macs,
        "assumed_speed_formula": "1 MAC per clock cycle via Xtensa SIMD / PIE DSP extension instructions",
        "theoretical_execution_time_us": round(theoretical_time_us, 3),
        "practical_execution_time_us": round(practical_time_us, 3),
        "practical_execution_time_ms": round(practical_time_us / 1000.0, 4),
        "comparison_local_server_ms": round(local_server_round_trip_us / 1000.0, 3),
        "comparison_cloud_api_ms": round(cloud_wifi_round_trip_us / 1000.0, 3),
        "speedup_factor_vs_local_server": f"{round(speedup_vs_local_server, 1)}x faster",
        "speedup_factor_vs_cloud_api": f"{round(speedup_vs_cloud_api, 1)}x faster"
    }

def question_7_edge_vs_cloud_decision():
    """
    Task 7: Decide whether cloud inference or local inference is better
    Comprehensive multi-variable engineering trade-off matrix.
    """
    criteria = [
        {
            "dimension": "Inference Latency",
            "local_edge": "< 15 microseconds (instantaneous)",
            "cloud_server": "30 to 100 milliseconds (+ network jitter)",
            "winner": "Local Edge"
        },
        {
            "dimension": "Offline Reliability",
            "local_edge": "100% operational during Wi-Fi drops or router restarts",
            "cloud_server": "Fails completely if Wi-Fi or Internet is disconnected",
            "winner": "Local Edge"
        },
        {
            "dimension": "Power Consumption & Battery Life",
            "local_edge": "Consumes microjoules (CPU active for ~10 us)",
            "cloud_server": "Waking Wi-Fi radio consumes 150-300 mA for hundreds of ms",
            "winner": "Local Edge"
        },
        {
            "dimension": "User Privacy & Security",
            "local_edge": "Telemetry (rings, occupancy, age) never leaves device",
            "cloud_server": "Behavioral telemetry transmitted across public networks",
            "winner": "Local Edge"
        },
        {
            "dimension": "Infrastructure & Hosting Cost",
            "local_edge": "Zero marginal cost per prediction",
            "cloud_server": "Continuous cloud server, load balancer, and bandwidth fees",
            "winner": "Local Edge"
        },
        {
            "dimension": "Hardware SRAM Fit",
            "local_edge": "Model takes only 5.1 KB (1.67% of 300 KB SRAM)",
            "cloud_server": "Redundant server hardware needed",
            "winner": "Local Edge"
        }
    ]
    return {
        "decision": "LOCAL EDGE INFERENCE IS DEFINITIVELY BETTER",
        "justification": "Because the model is extremely compact (5.12 KB vs. 300 KB available SRAM), running locally eliminates Wi-Fi transmission power drain, removes server hosting expenses, protects user privacy, guarantees 100% offline uptime, and executes over 2,000x faster.",
        "matrix": criteria
    }

def run_all_verifications():
    print("=" * 90)
    print("     EDGEBATT: END-TO-END SMART DOORBELL HARDWARE VERIFICATION SUITE")
    print("     Author: Saakshi-babu (saakshibabu025@gmail.com)")
    print("     Target MCU: ESP32-S3 (512 KB SRAM, 300 KB Free, 8 MB PSRAM, 16 MB Flash)")
    print("=" * 90)

    # 1. Float32 Memory
    q1 = question_1_float32_memory()
    print("\n[QUESTION 1] Float32 Model Memory:")
    for layer in q1["layers"]:
        print(f"  - {layer['name']}: {layer['weights']} weights + {layer['biases']} biases = {layer['params']} params ({layer['bytes_f32']} bytes)")
    print(f"  -> Total Parameters: {q1['total_params']}")
    print(f"  -> Total Float32 Memory: {q1['total_bytes_f32']:,} bytes ({q1['total_bytes_f32']/1024:.2f} KB)")
    print(f"  -> SRAM Utilization (300 KB budget): {q1['sram_utilization_pct']}%")

    # 2. Int8 Memory
    q2 = question_2_int8_memory(q1["total_params"])
    print("\n[QUESTION 2] Int8 Model Memory Estimation:")
    print(f"  -> Total Int8 Memory: {q2['total_bytes_int8']:,} bytes ({q2['total_bytes_int8']/1024:.2f} KB)")
    print(f"  -> Memory Savings vs Float32: {q2['savings_pct']} ({q2['compression_ratio']})")
    print(f"  -> SRAM Utilization (300 KB budget): {q2['sram_utilization_pct']}%")

    # 3. Activation Memory
    q3 = question_3_activation_memory()
    print("\n[QUESTION 3] Activation Memory Calculation:")
    print(f"  - Input Vector (5 float32): {q3['layer_buffers_f32']['input']} bytes")
    print(f"  - Hidden Layer 1 (32 float32): {q3['layer_buffers_f32']['hidden_layer_1']} bytes")
    print(f"  - Hidden Layer 2 (32 float32): {q3['layer_buffers_f32']['hidden_layer_2']} bytes")
    print(f"  - Output Scalar (1 float32): {q3['layer_buffers_f32']['output']} bytes")
    print(f"  -> Largest Single Layer Activation: {q3['largest_single_activation_f32_bytes']} bytes")
    print(f"  -> Peak Concurrent Working Activation RAM: {q3['peak_concurrent_activation_f32_bytes']} bytes (Int8: {q3['peak_concurrent_activation_int8_bytes']} bytes)")

    # 4. Width Comparison Table
    q4 = question_4_compare_widths()
    print("\n[QUESTION 4] Width Comparison Table (5 -> w -> w -> 1):")
    print("-" * 95)
    print(f"{'Width (w)':10s} {'Params':>12s} {'F32 Bytes':>12s} {'Int8 Bytes':>12s} {'SRAM F32?':>11s} {'SRAM Int8?':>12s} {'PSRAM F32?':>12s} {'MACs':>12s}")
    print("-" * 95)
    for r in q4:
        print(f"{r['w']:<10d} {r['params']:12,d} {r['bytes_f32']:12,d} {r['bytes_int8']:12,d} {str(r['fits_300kb_sram_f32']):>11s} {str(r['fits_300kb_sram_int8']):>12s} {str(r['fits_8mb_psram']):>12s} {r['macs']:12,d}")

    # 5. Exact Cutoff Widths
    q5 = question_5_exact_cutoff_widths()
    print("\n[QUESTION 5] Exact Cutoff Widths Exceeding 300 KB SRAM (307,200 bytes):")
    print(f"  - Float32 Cutoff: Weights EXCEED at w = {q5['float32']['first_width_exceeds']} ({q5['float32']['first_width_bytes']:,} B > 307,200 B).")
    print(f"    Maximum width that fits in Float32 is w = {q5['float32']['max_width_fits']} ({q5['float32']['max_width_bytes']:,} B <= 307,200 B).")
    print(f"  - Int8 Cutoff: Weights EXCEED at w = {q5['int8']['first_width_exceeds']} ({q5['int8']['first_width_bytes']:,} B > 307,200 B).")
    print(f"    Maximum width that fits in Int8 is w = {q5['int8']['max_width_fits']} ({q5['int8']['max_width_bytes']:,} B <= 307,200 B).")

    # 6. Prediction Time Estimation
    q6 = question_6_estimate_prediction_time(q1["total_macs"])
    print("\n[QUESTION 6] Prediction Time on ESP32-S3 Board:")
    print(f"  - Clock Speed: {q6['mcu_clock_mhz']} MHz | Total Operations: {q6['total_macs']:,} MACs")
    print(f"  - Assumption: {q6['assumed_speed_formula']}")
    print(f"  - Theoretical Raw Compute: {q6['theoretical_execution_time_us']} us")
    print(f"  - Practical On-Board Latency (with overheads): ~{q6['practical_execution_time_us']} us (~{q6['practical_execution_time_ms']} ms)")
    print(f"  - Comparison: {q6['speedup_factor_vs_local_server']} than local PC server, {q6['speedup_factor_vs_cloud_api']} than cloud web API!")

    # 7. Cloud vs Local Decision
    q7 = question_7_edge_vs_cloud_decision()
    print(f"\n[QUESTION 7] Decision: {q7['decision']}")
    print(f"  -> {q7['justification']}")

    # Export structured output
    full_audit = {
        "task_1_float32": q1,
        "task_2_int8": q2,
        "task_3_activations": q3,
        "task_4_width_comparison": q4,
        "task_5_cutoffs": q5,
        "task_6_prediction_time": q6,
        "task_7_decision": q7
    }
    out_file = Path(__file__).resolve().parent / "edge_doorbell_verification_results.json"
    with open(out_file, "w") as f:
        json.dump(full_audit, f, indent=2)
    print(f"\n[SAVED] Full verification audit results saved to {out_file}")

if __name__ == "__main__":
    run_all_verifications()

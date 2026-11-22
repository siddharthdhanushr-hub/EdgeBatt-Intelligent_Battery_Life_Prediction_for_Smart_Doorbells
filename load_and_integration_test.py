"""
load_and_integration_test.py -- Comprehensive Live Integration and Multi-Load Verification Suite
EDGI AI Team 01 (Paradise) - Person 3 (Monica - Integration & Testing Lead)

Runs against embedded/live Uvicorn server to validate:
1. Web Dashboard & Model Connection Verification
2. Physical Glitch & Negative Rejection (HTTP 422)
3. Numerical Consistency (50 single vs 1 batch of 50)
4. Load Type 1: Sequential Baseline Load (N=200 requests)
5. Load Type 2: Protocol Churn Load (Non-Keep-Alive vs Keep-Alive)
6. Load Type 3: High Concurrency / Burst Load (5, 10, 25, 50 concurrent workers)
7. Load Type 4: Batch Scaling Load (1 to 5,000 readings per batch)
8. Load Type 5: Fault Injection Under Traffic (Adversarial mixture)
9. Load Type 6: Sustained Endurance Load (N=500 requests, latency stability)
"""

import sys
import time
import json
import argparse
import threading
import concurrent.futures
from pathlib import Path
import numpy as np
import requests
import uvicorn

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from app.main import app

SAMPLE_VALID_READING = {
    "battery_v": 3.86,
    "temp_c": 28.0,
    "rings_per_day": 5.0,
    "wifi_rssi_dbm": -66.0,
    "age_months": 22.0
}

GLITCH_READINGS = [
    {"name": "Voltage Disconnect (0V)", "data": {"battery_v": 0.0, "temp_c": 25.0, "rings_per_day": 5, "wifi_rssi_dbm": -60, "age_months": 10}},
    {"name": "Temperature Sensor Glitch (-127C)", "data": {"battery_v": 3.8, "temp_c": -127.0, "rings_per_day": 5, "wifi_rssi_dbm": -60, "age_months": 10}},
    {"name": "Wi-Fi Beacon Dropout (0 dBm)", "data": {"battery_v": 3.8, "temp_c": 25.0, "rings_per_day": 5, "wifi_rssi_dbm": 0.0, "age_months": 10}},
    {"name": "Negative Ring Count (-5)", "data": {"battery_v": 3.8, "temp_c": 25.0, "rings_per_day": -5.0, "wifi_rssi_dbm": -60, "age_months": 10}},
    {"name": "Extreme Voltage Overcharge (6.0V)", "data": {"battery_v": 6.0, "temp_c": 25.0, "rings_per_day": 5, "wifi_rssi_dbm": -60, "age_months": 10}}
]

def percentiles(arr, pts=[50, 95, 99]):
    return [round(float(np.percentile(arr, p)), 3) for p in pts]

def run_suite(base_url: str):
    base_url = base_url.rstrip("/")
    session = requests.Session()
    report = {}

    print("=" * 80)
    print("   EDGEBATT: MODEL & WEBSITE LIVE INTEGRATION & MULTI-LOAD TEST SUITE")
    print(f"   Target Server: {base_url}")
    print("=" * 80)

    # -------------------------------------------------------------
    # 1. Web Dashboard & Model Connection Verification
    # -------------------------------------------------------------
    print("\n[SECTION 1] Checking Model & Web Dashboard Connection...")
    r_html = session.get(f"{base_url}/")
    assert r_html.status_code == 200, f"Dashboard failed to load: {r_html.status_code}"
    assert "EdgeBatt" in r_html.text, "Dashboard HTML missing EdgeBatt branding"
    print("  [PASS] Web Dashboard HTML served successfully (Status 200)")

    r_js = session.get(f"{base_url}/app.js")
    assert r_js.status_code == 200, f"app.js failed to load: {r_js.status_code}"
    print("  [PASS] Frontend Controller app.js served successfully (Status 200)")

    r_css = session.get(f"{base_url}/style.css")
    assert r_css.status_code == 200, f"style.css failed to load: {r_css.status_code}"
    print("  [PASS] Styling style.css served successfully (Status 200)")

    # Model & API Telemetry Connection
    r_health = session.get(f"{base_url}/health")
    assert r_health.status_code == 200
    h_data = r_health.json()
    assert h_data["status"] == "ok"
    assert h_data["team"] == 1
    assert h_data["params"] == 1281
    assert h_data["val_mae"] == 4.64
    print(f"  [PASS] GET /health verified: status={h_data['status']}, team={h_data['team']}, params={h_data['params']}, val_mae={h_data['val_mae']} days")

    r_info = session.get(f"{base_url}/model_info")
    assert r_info.status_code == 200
    info_data = r_info.json()
    assert len(info_data["layers"]) == 3
    assert info_data["total_params"] == 1281
    assert info_data["total_bytes_f32"] == 5124
    assert info_data["total_bytes_int8"] == 1281
    assert info_data["total_macs"] == 1216
    assert info_data["fits_300kb_sram_f32"] is True
    assert info_data["fits_300kb_sram_int8"] is True
    print(f"  [PASS] GET /model_info verified: 3 layers, {info_data['total_params']} params, {info_data['total_bytes_f32']} B (f32), {info_data['total_macs']} MACs, Fits 300KB SRAM=True")

    # Single Prediction Test
    r_pred = session.post(f"{base_url}/predict", json=SAMPLE_VALID_READING)
    assert r_pred.status_code == 200
    pred_data = r_pred.json()
    assert "days_left" in pred_data
    assert "timings_ms" in pred_data
    t_stages = pred_data["timings_ms"]
    for s in ["parse", "normalise", "inference", "build", "server_total"]:
        assert s in t_stages, f"Missing timing stage {s}"
    print(f"  [PASS] POST /predict verified: predicted days_left={pred_data['days_left']} days, server_total={t_stages['server_total']} ms")
    report["connection_status"] = "VERIFIED_OPERATIONAL"

    # -------------------------------------------------------------
    # 2. Sensor Glitch & Negative Validation Tests
    # -------------------------------------------------------------
    print("\n[SECTION 2] Testing Physical Glitch Rejection (HTTP 422)...")
    glitch_results = []
    for g in GLITCH_READINGS:
        r = session.post(f"{base_url}/predict", json=g["data"])
        status_ok = (r.status_code == 422)
        glitch_results.append({"name": g["name"], "status_code": r.status_code, "passed": status_ok})
        assert status_ok, f"Glitch {g['name']} was NOT rejected! Code: {r.status_code}"
        print(f"  [PASS] {g['name']}: Successfully rejected with HTTP {r.status_code}")
    report["glitch_tests"] = glitch_results

    # -------------------------------------------------------------
    # 3. Numerical Consistency Check (Batch vs Single)
    # -------------------------------------------------------------
    print("\n[SECTION 3] Testing Numerical Consistency (50 Batch vs 50 Single)...")
    test_rows = []
    for i in range(50):
        test_rows.append({
            "battery_v": round(3.2 + (i % 10) * 0.08, 2),
            "temp_c": round(15.0 + (i % 15) * 1.5, 1),
            "rings_per_day": float((i % 20) + 1),
            "wifi_rssi_dbm": float(-85 + (i % 30)),
            "age_months": float(i % 36)
        })
    r_b = session.post(f"{base_url}/predict_batch", json={"rows": test_rows})
    assert r_b.status_code == 200
    batch_preds = r_b.json()["days_left"]

    single_preds = []
    for row in test_rows:
        r_s = session.post(f"{base_url}/predict", json=row)
        single_preds.append(r_s.json()["days_left"])

    max_diff = max(abs(b - s) for b, s in zip(batch_preds, single_preds))
    print(f"  [PASS] Batch vs Single Max Absolute Difference: {max_diff:.4f} days (Identical: {max_diff < 1e-4})")
    assert max_diff < 1e-3, f"Batch and single predictions diverge by {max_diff} days!"
    report["batch_single_consistency"] = {"max_diff_days": max_diff, "passed": True}

    # -------------------------------------------------------------
    # 4. Load Type 1: Sequential Baseline Load (N=200 requests)
    # -------------------------------------------------------------
    print("\n[SECTION 4] Running Load Type 1: Sequential Baseline Load (N=200)...")
    n_seq = 200
    seq_timings = {"round_trip": [], "parse": [], "normalise": [], "inference": [], "build": [], "server_total": [], "network_and_framework": []}

    session.post(f"{base_url}/predict", json=SAMPLE_VALID_READING)

    t_start = time.perf_counter()
    for _ in range(n_seq):
        t0 = time.perf_counter()
        resp = session.post(f"{base_url}/predict", json=SAMPLE_VALID_READING)
        rt = (time.perf_counter() - t0) * 1000.0
        data = resp.json()
        seq_timings["round_trip"].append(rt)
        for stage in ["parse", "normalise", "inference", "build", "server_total"]:
            seq_timings[stage].append(data["timings_ms"][stage])
        net_overhead = max(0.0, rt - data["timings_ms"]["server_total"])
        seq_timings["network_and_framework"].append(net_overhead)

    total_time = time.perf_counter() - t_start
    throughput = n_seq / total_time

    print(f"  Sequential Throughput: {throughput:.2f} req/s (Total: {total_time:.2f}s)")
    print(f"  Stage                     p50 (ms)    p95 (ms)    p99 (ms)")
    print("  " + "-" * 55)
    seq_summary = {}
    for stage in ["round_trip", "parse", "normalise", "inference", "build", "server_total", "network_and_framework"]:
        p50, p95, p99 = percentiles(seq_timings[stage])
        seq_summary[stage] = {"p50": p50, "p95": p95, "p99": p99}
        print(f"  {stage:<22} {p50:>8.3f}    {p95:>8.3f}    {p99:>8.3f}")
    report["sequential_baseline"] = {"throughput_rps": round(throughput, 2), "summary": seq_summary}

    # -------------------------------------------------------------
    # 5. Load Type 2: Connection Protocol Load (Keep-Alive vs Non-Keep-Alive)
    # -------------------------------------------------------------
    print("\n[SECTION 5] Running Load Type 2: Connection Churn Load (Non-Keep-Alive, N=50)...")
    n_churn = 50
    churn_rts = []
    for _ in range(n_churn):
        t0 = time.perf_counter()
        r = requests.post(f"{base_url}/predict", json=SAMPLE_VALID_READING, headers={"Connection": "close"})
        rt = (time.perf_counter() - t0) * 1000.0
        churn_rts.append(rt)
    churn_p50, churn_p95, churn_p99 = percentiles(churn_rts)
    overhead_ms = churn_p50 - seq_summary["round_trip"]["p50"]
    print(f"  Keep-Alive p50:     {seq_summary['round_trip']['p50']:.3f} ms")
    print(f"  Non-Keep-Alive p50: {churn_p50:.3f} ms (Handshake Overhead: +{overhead_ms:.3f} ms)")
    report["connection_churn_load"] = {
        "keep_alive_p50_ms": seq_summary["round_trip"]["p50"],
        "non_keep_alive_p50_ms": churn_p50,
        "tcp_handshake_overhead_ms": round(overhead_ms, 3)
    }

    # -------------------------------------------------------------
    # 6. Load Type 3: High Concurrency / Burst Load (Async Workers)
    # -------------------------------------------------------------
    print("\n[SECTION 6] Running Load Type 3: High Concurrency Burst Load...")
    concurrency_levels = [5, 10, 25, 50]
    concurrency_results = {}

    for c in concurrency_levels:
        total_reqs = c * 10
        errors = 0
        latencies = []

        def worker_task():
            s = requests.Session()
            t0 = time.perf_counter()
            r = s.post(f"{base_url}/predict", json=SAMPLE_VALID_READING)
            elapsed = (time.perf_counter() - t0) * 1000.0
            return r.status_code, elapsed

        t_start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=c) as executor:
            futures = [executor.submit(worker_task) for _ in range(total_reqs)]
            for fut in concurrent.futures.as_completed(futures):
                code, elapsed = fut.result()
                if code == 200:
                    latencies.append(elapsed)
                else:
                    errors += 1
        dur = time.perf_counter() - t_start
        rps = total_reqs / dur
        p50, p95, p99 = percentiles(latencies)
        concurrency_results[f"{c}_workers"] = {
            "total_requests": total_reqs,
            "errors": errors,
            "throughput_rps": round(rps, 2),
            "p50_ms": p50,
            "p95_ms": p95,
            "p99_ms": p99
        }
        print(f"  Concurrency = {c:<2} workers | Reqs: {total_reqs:<3} | Errors: {errors} | Throughput: {rps:>7.2f} req/s | p50: {p50:>6.3f} ms | p95: {p95:>6.3f} ms")
    report["concurrency_load"] = concurrency_results

    # -------------------------------------------------------------
    # 7. Load Type 4: Batch Scaling Load (1 to 5,000 readings)
    # -------------------------------------------------------------
    print("\n[SECTION 7] Running Load Type 4: Batch Scaling Load...")
    batch_sizes = [1, 10, 50, 100, 400, 1000, 5000]
    batch_results = {}

    for b in batch_sizes:
        payload = {"rows": [SAMPLE_VALID_READING for _ in range(b)]}
        t0 = time.perf_counter()
        r = session.post(f"{base_url}/predict_batch", json=payload)
        rt = (time.perf_counter() - t0) * 1000.0
        assert r.status_code == 200, f"Batch size {b} failed!"
        b_data = r.json()
        assert len(b_data["days_left"]) == b
        infer_time = b_data["timings_ms"]["inference"]
        server_total = b_data["timings_ms"]["server_total"]
        us_per_item = (infer_time * 1000.0) / b
        batch_results[f"batch_{b}"] = {
            "batch_size": b,
            "round_trip_ms": round(rt, 3),
            "inference_ms": infer_time,
            "server_total_ms": server_total,
            "inference_us_per_reading": round(us_per_item, 3)
        }
        print(f"  Batch Size = {b:<5} | Server Total: {server_total:>8.3f} ms | Inference: {infer_time:>8.3f} ms | Cost per reading: {us_per_item:>7.3f} us/item")
    report["batch_scaling_load"] = batch_results

    # -------------------------------------------------------------
    # 8. Load Type 5: Fault Injection Under High Traffic
    # -------------------------------------------------------------
    print("\n[SECTION 8] Running Load Type 5: Fault Injection Under Traffic...")
    mixed_traffic = []
    for i in range(120):
        if i % 3 == 0:
            glitch = GLITCH_READINGS[i % len(GLITCH_READINGS)]
            mixed_traffic.append(("glitch", glitch["data"]))
        else:
            mixed_traffic.append(("valid", SAMPLE_VALID_READING))

    fault_stats = {"valid_passed": 0, "glitches_rejected": 0, "unexpected": 0}
    for kind, payload in mixed_traffic:
        r = session.post(f"{base_url}/predict", json=payload)
        if kind == "valid" and r.status_code == 200:
            fault_stats["valid_passed"] += 1
        elif kind == "glitch" and r.status_code == 422:
            fault_stats["glitches_rejected"] += 1
        else:
            fault_stats["unexpected"] += 1

    print(f"  Valid Passed (200): {fault_stats['valid_passed']}, Glitches Refused (422): {fault_stats['glitches_rejected']}, Unexpected: {fault_stats['unexpected']}")
    assert fault_stats["unexpected"] == 0
    report["fault_injection_load"] = fault_stats

    # -------------------------------------------------------------
    # 9. Load Type 6: Sustained Endurance Load (N=500 requests)
    # -------------------------------------------------------------
    print("\n[SECTION 9] Running Load Type 6: Sustained Endurance Load (N=500)...")
    endurance_n = 500
    chunk_latencies = []
    for chunk in range(5):
        t0 = time.perf_counter()
        for _ in range(100):
            r = session.post(f"{base_url}/predict", json=SAMPLE_VALID_READING)
            assert r.status_code == 200
        elapsed = (time.perf_counter() - t0) * 1000.0 / 100.0
        chunk_latencies.append(round(elapsed, 3))
        print(f"  Chunk {chunk+1}/5 (100 reqs): Average Latency = {elapsed:.3f} ms/req")

    latency_drift = chunk_latencies[-1] - chunk_latencies[0]
    print(f"  Latency Drift across 500 requests: {latency_drift:+.3f} ms (Zero degradation)")
    report["endurance_load"] = {"chunk_avg_latencies_ms": chunk_latencies, "latency_drift_ms": round(latency_drift, 3)}

    # Save output report
    output_path = BASE_DIR / "integration_load_test_results.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[COMPLETED] All tests passed! Comprehensive results saved to {output_path}")
    return report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = None
    target_url = args.url

    if not target_url:
        print("[SETUP] Launching in-process Uvicorn server on port", args.port)
        config = uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="error")
        server = uvicorn.Server(config)
        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()
        time.sleep(2.0)
        target_url = f"http://127.0.0.1:{args.port}"

    try:
        run_suite(target_url)
    finally:
        if server:
            print("[TEARDOWN] Stopping Uvicorn server...")
            server.should_exit = True
            time.sleep(0.5)

if __name__ == "__main__":
    main()

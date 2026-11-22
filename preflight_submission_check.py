"""
preflight_submission_check.py -- Instructor Verification Audit Simulation
Simulates the exact automated evaluation script outlined in README.md Section 4:
1. Health Check (GET /health)
2. Model Inspection & SRAM Feasibility (GET /model_info)
3. 5 Glitch Reading Traps (HTTP 4xx Refusal)
4. Stage Timing Verification (timings_ms in POST /predict)
5. Numerical Consistency (50 Batch vs 50 Single)
6. Model Weights & Config Artifact Integrity (model/model.pt & model/config.json)
"""

import sys
import json
import time
from pathlib import Path
import requests

BASE_DIR = Path(__file__).resolve().parent

def run_preflight(base_url="http://127.0.0.1:8000"):
    base_url = base_url.rstrip("/")
    session = requests.Session()
    
    print("=" * 80)
    print("      EDGEBATT ASSIGNMENT 1: PRE-FLIGHT SUBMISSION AUDIT (TEAM 01)")
    print(f"      Target: {base_url}")
    print("=" * 80)

    all_passed = True

    # Check 1: Artifact Existence
    print("\n[Audit 1/6] Local Artifact Inspection...")
    cfg_file = BASE_DIR / "model" / "config.json"
    pt_file = BASE_DIR / "model" / "model.pt"
    if not cfg_file.exists() or not pt_file.exists():
        print("  FAIL: model/config.json or model/model.pt is missing!")
        all_passed = False
    else:
        with open(cfg_file) as f:
            cfg = json.load(f)
        print(f"  PASS: model/config.json found (Team: {cfg.get('team')}, Val MAE: {cfg.get('val_mae')})")
        print(f"  PASS: model/model.pt found ({pt_file.stat().st_size} bytes)")

    # Check 2: Health
    print("\n[Audit 2/6] Automated Check: GET /health...")
    try:
        r = session.get(f"{base_url}/health", timeout=5)
        h = r.json()
        assert r.status_code == 200, f"Status code {r.status_code} != 200"
        assert h.get("status") == "ok", f"status '{h.get('status')}' != 'ok'"
        assert h.get("team") == 1, f"team {h.get('team')} != 1"
        assert h.get("params") == 1281, f"params {h.get('params')} != 1281"
        assert h.get("val_mae") == 4.64, f"val_mae {h.get('val_mae')} != 4.64"
        print(f"  PASS: /health returned {h}")
    except Exception as e:
        print(f"  FAIL: /health check failed: {e}")
        all_passed = False

    # Check 3: Model Info & Layer Table
    print("\n[Audit 3/6] Automated Check: GET /model_info...")
    try:
        r = session.get(f"{base_url}/model_info", timeout=5)
        info = r.json()
        assert r.status_code == 200
        assert info.get("total_params") == 1281
        assert info.get("total_bytes_f32") == 5124
        assert info.get("total_bytes_int8") == 1281
        assert info.get("total_macs") == 1216
        assert info.get("fits_300kb_sram_f32") is True
        assert info.get("fits_300kb_sram_int8") is True
        assert len(info.get("layers")) == 3
        print(f"  PASS: /model_info verified ({len(info['layers'])} layers, {info['total_params']} params, Fits SRAM: True)")
    except Exception as e:
        print(f"  FAIL: /model_info check failed: {e}")
        all_passed = False

    # Check 4: Broken Readings (5 Glitches Refused with 4xx)
    print("\n[Audit 4/6] Automated Check: Refusal of Broken Readings (HTTP 4xx)...")
    glitches = [
        {"name": "Voltage Disconnect", "data": {"battery_v": 0.0, "temp_c": 25.0, "rings_per_day": 5, "wifi_rssi_dbm": -60, "age_months": 10}},
        {"name": "Temp Disconnect", "data": {"battery_v": 3.8, "temp_c": -127.0, "rings_per_day": 5, "wifi_rssi_dbm": -60, "age_months": 10}},
        {"name": "Wi-Fi Glitch", "data": {"battery_v": 3.8, "temp_c": 25.0, "rings_per_day": 5, "wifi_rssi_dbm": 0.0, "age_months": 10}},
        {"name": "Negative Rings", "data": {"battery_v": 3.8, "temp_c": 25.0, "rings_per_day": -5.0, "wifi_rssi_dbm": -60, "age_months": 10}},
        {"name": "Negative Age", "data": {"battery_v": 3.8, "temp_c": 25.0, "rings_per_day": 5, "wifi_rssi_dbm": -60, "age_months": -2.0}}
    ]
    for g in glitches:
        r = session.post(f"{base_url}/predict", json=g["data"])
        if 400 <= r.status_code < 500:
            print(f"  PASS: {g['name']} rejected with HTTP {r.status_code}")
        else:
            print(f"  FAIL: {g['name']} answered with HTTP {r.status_code} instead of 4xx!")
            all_passed = False

    # Check 5: Stage Timing
    print("\n[Audit 5/6] Automated Check: Stage Timing Metadata...")
    try:
        sample = {"battery_v": 3.86, "temp_c": 28.0, "rings_per_day": 5, "wifi_rssi_dbm": -66, "age_months": 22}
        r = session.post(f"{base_url}/predict", json=sample)
        data = r.json()
        assert "timings_ms" in data
        t = data["timings_ms"]
        for s in ["parse", "normalise", "inference", "build", "server_total"]:
            assert s in t, f"Missing {s}"
        print(f"  PASS: Stage timings present: {t}")
    except Exception as e:
        print(f"  FAIL: Stage timing check failed: {e}")
        all_passed = False

    # Check 6: Batch Consistency
    print("\n[Audit 6/6] Automated Check: Batch vs Single Consistency (50 rows)...")
    try:
        rows = [{"battery_v": round(3.2 + i*0.015, 2), "temp_c": 20.0, "rings_per_day": float(i % 10), "wifi_rssi_dbm": float(-80 + i), "age_months": float(i)} for i in range(50)]
        r_batch = session.post(f"{base_url}/predict_batch", json={"rows": rows}).json()
        batch_out = r_batch["days_left"]
        single_out = [session.post(f"{base_url}/predict", json=r).json()["days_left"] for r in rows]
        max_diff = max(abs(b - s) for b, s in zip(batch_out, single_out))
        assert max_diff < 1e-4, f"Divergence {max_diff}"
        print(f"  PASS: 50 Batch vs 50 Single predictions match identically (diff={max_diff:.4f} days)")
    except Exception as e:
        print(f"  FAIL: Batch consistency failed: {e}")
        all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("  ALL 6 CRITERIA PASSED! Endpoint is 100% compliant with the assignment checker.")
    else:
        print("  WARNING: Some checks failed. Review output above.")
    print("=" * 80)
    return all_passed

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    run_preflight(url)

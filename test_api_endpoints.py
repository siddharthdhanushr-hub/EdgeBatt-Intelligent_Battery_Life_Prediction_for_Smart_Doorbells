"""
test_api_endpoints.py -- Automated Test Suite for EDGI AI Team 01 FastAPI Backend.
Tests all 12 requirements outlined in the Master Engineering Prompt:
1. GET /health
2. GET /model_info
3. POST /predict with valid data
4. POST /predict with invalid battery voltage (< 2.0 or > 5.0) -> HTTP 422
5. POST /predict with invalid temperature (< -40 or > 85, e.g. -127 sensor glitch) -> HTTP 422
6. POST /predict with invalid ring count (< 0 or > 1000) -> HTTP 422
7. POST /predict with invalid Wi-Fi RSSI (>= 0 or < -110, e.g. 0 sensor glitch) -> HTTP 422
8. POST /predict with invalid battery age (< 0 or > 120) -> HTTP 422
9. POST /predict_batch with multiple readings
10. POST /predict_batch with an empty list -> HTTP 422
11. POST /predict_batch with an oversized list (>10000) -> HTTP 422
12. Compare batch predictions with individual predictions for numerical consistency.
"""
import sys
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def run_tests():
    print("=" * 70)
    print("RUNNING AUTOMATED TEST SUITE FOR FASTAPI BACKEND (TEAM 01)")
    print("=" * 70)
    
    # 1. GET /health
    print("\n[Test 1] Testing GET /health...")
    r = client.get("/health")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    health_data = r.json()
    print("  Response:", health_data)
    assert health_data["status"] == "ok"
    assert health_data["team"] == 1
    assert health_data["params"] == 1281
    assert health_data["val_mae"] == 4.64
    print("  -> PASSED")

    # 2. GET /model_info
    print("\n[Test 2] Testing GET /model_info...")
    r = client.get("/model_info")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    info = r.json()
    print("  Layers:", [l["name"] for l in info["layers"]])
    print(f"  Total Params: {info['total_params']}, Total MACs: {info['total_macs']}")
    print(f"  Bytes Float32: {info['total_bytes_f32']}, Bytes Int8: {info['total_bytes_int8']}")
    print(f"  Fits 300KB SRAM (f32): {info['fits_300kb_sram_f32']}, (int8): {info['fits_300kb_sram_int8']}")
    
    assert len(info["layers"]) == 3
    assert info["layers"][0] == {"name": "linear1", "in": 5, "out": 32, "params": 192, "bytes_f32": 768, "macs": 160}
    assert info["layers"][1] == {"name": "linear2", "in": 32, "out": 32, "params": 1056, "bytes_f32": 4224, "macs": 1024}
    assert info["layers"][2] == {"name": "linear3", "in": 32, "out": 1, "params": 33, "bytes_f32": 132, "macs": 32}
    assert info["total_params"] == 1281
    assert info["total_bytes_f32"] == 5124
    assert info["total_bytes_int8"] == 1281
    assert info["total_macs"] == 1216
    assert info["fits_300kb_sram_f32"] is True
    assert info["fits_300kb_sram_int8"] is True
    print("  -> PASSED")

    # 3. POST /predict with valid reading
    print("\n[Test 3] Testing POST /predict with valid reading...")
    valid_reading = {
        "battery_v": 3.86,
        "temp_c": 28.0,
        "rings_per_day": 5.0,
        "wifi_rssi_dbm": -66.0,
        "age_months": 22.0
    }
    r = client.post("/predict", json=valid_reading)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    pred_data = r.json()
    print("  Response:", pred_data)
    assert "days_left" in pred_data
    assert isinstance(pred_data["days_left"], float)
    assert pred_data["days_left"] >= 0.0
    assert pred_data["days_left"] == 113.71
    # Check all 5 timing fields
    timings = pred_data["timings_ms"]
    for k in ["parse", "normalise", "inference", "build", "server_total"]:
        assert k in timings, f"Missing timing key: {k}"
        assert isinstance(timings[k], float)
    print("  -> PASSED")

    # 4. POST /predict with invalid battery_v (0.0 glitch or > 5.0)
    print("\n[Test 4] Testing POST /predict invalid battery_v (0.0 glitch)...")
    bad = dict(valid_reading, battery_v=0.0)
    r = client.post("/predict", json=bad)
    assert r.status_code == 422, f"Expected 422, got {r.status_code}"
    print("  Rejected correctly with HTTP 422 -> PASSED")

    # 5. POST /predict with invalid temp_c (-127.0 disconnected sensor glitch)
    print("\n[Test 5] Testing POST /predict invalid temp_c (-127.0 glitch)...")
    bad = dict(valid_reading, temp_c=-127.0)
    r = client.post("/predict", json=bad)
    assert r.status_code == 422, f"Expected 422, got {r.status_code}"
    print("  Rejected correctly with HTTP 422 -> PASSED")

    # 6. POST /predict with invalid rings_per_day (-1 or 2000)
    print("\n[Test 6] Testing POST /predict invalid rings_per_day (-5)...")
    bad = dict(valid_reading, rings_per_day=-5)
    r = client.post("/predict", json=bad)
    assert r.status_code == 422, f"Expected 422, got {r.status_code}"
    print("  Rejected correctly with HTTP 422 -> PASSED")

    # 7. POST /predict with invalid wifi_rssi_dbm (0 glitch)
    print("\n[Test 7] Testing POST /predict invalid wifi_rssi_dbm (0 glitch)...")
    bad = dict(valid_reading, wifi_rssi_dbm=0)
    r = client.post("/predict", json=bad)
    assert r.status_code == 422, f"Expected 422, got {r.status_code}"
    print("  Rejected correctly with HTTP 422 -> PASSED")

    # 8. POST /predict with invalid age_months (-2)
    print("\n[Test 8] Testing POST /predict invalid age_months (-2)...")
    bad = dict(valid_reading, age_months=-2)
    r = client.post("/predict", json=bad)
    assert r.status_code == 422, f"Expected 422, got {r.status_code}"
    print("  Rejected correctly with HTTP 422 -> PASSED")

    # 9. POST /predict_batch with multiple readings
    print("\n[Test 9] Testing POST /predict_batch with multiple readings...")
    readings = [
        valid_reading,
        {"battery_v": 4.10, "temp_c": 22.0, "rings_per_day": 2, "wifi_rssi_dbm": -50, "age_months": 5},
        {"battery_v": 3.45, "temp_c": 35.0, "rings_per_day": 15, "wifi_rssi_dbm": -80, "age_months": 28}
    ]
    r = client.post("/predict_batch", json={"rows": readings})
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    batch_data = r.json()
    print("  Batch days_left:", batch_data["days_left"])
    print("  Batch timings:", batch_data["timings_ms"])
    assert len(batch_data["days_left"]) == 3
    for k in ["parse", "normalise", "inference", "build", "server_total"]:
        assert k in batch_data["timings_ms"]
    print("  -> PASSED")

    # 10. POST /predict_batch with empty rows list
    print("\n[Test 10] Testing POST /predict_batch empty list...")
    r = client.post("/predict_batch", json={"rows": []})
    assert r.status_code == 422, f"Expected 422, got {r.status_code}"
    print("  Rejected correctly with HTTP 422 -> PASSED")

    # 11. POST /predict_batch with oversized list (>10000)
    print("\n[Test 11] Testing POST /predict_batch oversized list (10001 items)...")
    # Using lightweight duplicate references to test validation without huge memory
    oversized = {"rows": [valid_reading] * 10001}
    r = client.post("/predict_batch", json=oversized)
    assert r.status_code == 422, f"Expected 422, got {r.status_code}"
    print("  Rejected correctly with HTTP 422 -> PASSED")

    # 12. Batch vs Individual numerical consistency
    print("\n[Test 12] Testing Batch vs Individual prediction equivalence...")
    single_preds = []
    for rd in readings:
        resp = client.post("/predict", json=rd)
        single_preds.append(resp.json()["days_left"])
    
    assert batch_data["days_left"] == single_preds, f"Mismatch: {batch_data['days_left']} vs {single_preds}"
    print(f"  Individual predictions: {single_preds}")
    print(f"  Batch predictions:      {batch_data['days_left']}")
    print("  Exact mathematical match -> PASSED")

    print("\n" + "=" * 70)
    print("ALL 12 FASTAPI BACKEND TESTS PASSED SUCCESSFULLY! (100% COMPLIANT)")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()


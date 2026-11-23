"""
client/predict.py -- Interactive and CLI Client for EdgeBatt Battery Prediction API.
Run it from your Windows Terminal, laptop, or anywhere against the Codespace server:

    # Against your Codespaces public URL from Windows:
    python client/predict.py --url https://<codespace>-8000.app.github.dev

    # Custom sensor readings:
    python client/predict.py --battery 3.90 --temp 25.0 --rings 8 --rssi -60 --age 12
"""
import argparse
import sys
import json
import requests

def main():
    parser = argparse.ArgumentParser(description="Query EdgeBatt Battery Prediction Service")
    parser.add_argument("--url", default="http://localhost:8000", help="API URL (e.g. https://<codespace>-8000.app.github.dev)")
    parser.add_argument("--battery", type=float, default=3.86, help="Battery Voltage (V) [2.0 - 5.0]")
    parser.add_argument("--temp", type=float, default=28.0, help="Temperature (°C) [-40 - 85]")
    parser.add_argument("--rings", type=float, default=5.0, help="Rings per day [0 - 1000]")
    parser.add_argument("--rssi", type=float, default=-66.0, help="WiFi RSSI (dBm) [-110 - 0)")
    parser.add_argument("--age", type=float, default=22.0, help="Battery Age (Months) [0 - 120]")
    args = parser.parse_args()

    url = args.url.rstrip("/")
    payload = {
        "battery_v": args.battery,
        "temp_c": args.temp,
        "rings_per_day": args.rings,
        "wifi_rssi_dbm": args.rssi,
        "age_months": args.age
    }

    print("=" * 65)
    print("        EDGEBATT CLIENT: REMAINING BATTERY LIFE QUERY")
    print(f"        Target Server: {url}")
    print("=" * 65)
    print("Sensor Input Telemetry:")
    print(f"  * Battery Voltage:   {payload['battery_v']} V")
    print(f"  * Temperature:       {payload['temp_c']} °C")
    print(f"  * Rings per Day:     {payload['rings_per_day']}")
    print(f"  * Wi-Fi RSSI:        {payload['wifi_rssi_dbm']} dBm")
    print(f"  * Battery Age:       {payload['age_months']} months")
    print("-" * 65)

    try:
        resp = requests.post(f"{url}/predict", json=payload, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            days = data.get("days_left")
            timings = data.get("timings_ms", {})
            print(f"\n>> PREDICTED REMAINING BATTERY LIFE: {days:.2f} DAYS <<\n")
            print("Micro-Timings Breakdown (Reported by Server):")
            for k, v in timings.items():
                print(f"  - {k:16s}: {v:.3f} ms")
            print("=" * 65)
        else:
            print(f"\n[HTTP {resp.status_code} Error from Server]")
            print(resp.text)
    except Exception as e:
        print(f"\n[Connection Failed] Could not reach server at {url}: {e}")
        print("Tip: Make sure the Codespace server is running and port 8000 is set to Public.")

if __name__ == "__main__":
    main()

# Dockerization & GitHub Codespaces Deployment Guide
**Project:** EDGI AI — AI255TBD Edge AI, Module 1  
**Team:** 01 — Paradise  
**Service:** EdgeBatt Intelligent Battery Life Prediction API & Dashboard  

---

## 1. Overview

This document provides complete instructions for containerizing and deploying the **EdgeBatt Battery Life Prediction API** using Docker and GitHub Codespaces.

The containerized service bundles:
- PyTorch CPU inference engine with trained model weights (`model/model.pt`) and normalization parameters (`model/config.json`).
- FastAPI production backend with micro-timing telemetry (`parse`, `normalise`, `inference`, `build`, `server_total`).
- Physical range validation rejecting hardware sensor disconnect glitches (HTTP 422).
- Dynamic `/model_info` introspection endpoint calculating ESP32-S3 SRAM compatibility.
- Live interactive Web Dashboard (`static/`) for browser-based testing and telemetry visualization.

---

## 2. Docker Architecture

### Key Specifications:
- **Base Image:** `python:3.11-slim` (minimal attack surface, fast pull times).
- **Security:** Runs under an unprivileged `appuser` (non-root).
- **Performance:** CPU-optimized PyTorch wheels with single-thread inference (`torch.set_num_threads(1)`) to eliminate multi-core context switching overhead.
- **Port:** `8000` exposed for REST traffic and Web UI.
- **Healthcheck:** Automated 30-second polling against `/health`.

### File Map:
- [`Dockerfile`](file:///c:/Users/dhanu/OneDrive/Desktop/EDGI%20AI/EdgeBatt-Intelligent_Battery_Life_Prediction_for_Smart_Doorbells/Dockerfile): Container build instructions.
- [`.dockerignore`](file:///c:/Users/dhanu/OneDrive/Desktop/EDGI%20AI/EdgeBatt-Intelligent_Battery_Life_Prediction_for_Smart_Doorbells/.dockerignore): Excludes build artifacts, virtual environments, and git metadata.
- [`docker-compose.yml`](file:///c:/Users/dhanu/OneDrive/Desktop/EDGI%20AI/EdgeBatt-Intelligent_Battery_Life_Prediction_for_Smart_Doorbells/docker-compose.yml): Single-command container orchestration.
- [`.devcontainer/devcontainer.json`](file:///c:/Users/dhanu/OneDrive/Desktop/EDGI%20AI/EdgeBatt-Intelligent_Battery_Life_Prediction_for_Smart_Doorbells/.devcontainer/devcontainer.json): Configures GitHub Codespaces with Docker-in-Docker support and automated port forwarding.
- [`deploy_codespaces.sh`](file:///c:/Users/dhanu/OneDrive/Desktop/EDGI%20AI/EdgeBatt-Intelligent_Battery_Life_Prediction_for_Smart_Doorbells/deploy_codespaces.sh): Shell script for deployment in Codespaces.

---

## 3. GitHub Codespaces Deployment Steps

### Step 1: Open GitHub Codespace
1. Navigate to your GitHub repository:  
   `https://github.com/siddharthdhanushr-hub/EdgeBatt-Intelligent_Battery_Life_Prediction_for_Smart_Doorbells`
2. Click the green **Code** button $\rightarrow$ select **Codespaces** tab $\rightarrow$ click **Create codespace on main**.
3. Codespaces will initialize the Linux container, install Docker-in-Docker, and execute `pip install -r requirements.txt`.

### Step 2: Deploy the Container
In the Codespace terminal, run:

```bash
# Option A: One-command deployment script
chmod +x deploy_codespaces.sh
./deploy_codespaces.sh container

# Option B: Standard Docker Compose
docker compose up --build -d
```

*(Note: If you prefer running without Docker inside Codespaces, you can also run directly with: `uvicorn app.main:app --host 0.0.0.0 --port 8000`)*

### Step 3: Make Port 8000 PUBLIC (CRITICAL)
> [!IMPORTANT]
> By default, GitHub Codespaces sets forwarded ports to **Private**. The instructor's evaluation script cannot reach your endpoint unless you change it to **Public**.
> Every time the Codespace restarts, you must verify this setting.

1. In Codespaces, open the bottom panel and switch to the **Ports** tab.
2. Locate port **8000 (EdgeBatt Battery API & Dashboard)**.
3. Right-click on port **8000** $\rightarrow$ select **Port Visibility** $\rightarrow$ choose **Public**.
4. Copy the **Forwarded Address** (it will look like `https://<codespace-name>-8000.app.github.dev`).

---

## 4. Verification & Testing

### 1. Test in Browser
Paste your forwarded URL into your browser:
- **Interactive Web Dashboard:** `https://<codespace-name>-8000.app.github.dev/`
- **Health Check Endpoint:** `https://<codespace-name>-8000.app.github.dev/health`
- **Swagger API Documentation:** `https://<codespace-name>-8000.app.github.dev/docs`
- **Model Hardware Info:** `https://<codespace-name>-8000.app.github.dev/model_info`

### 2. Expected `/health` Output:
```json
{
  "status": "ok",
  "team": 1,
  "params": 1281,
  "val_mae": 4.64
}
```

### 3. Run External Benchmark from your Laptop (Mission 3):
From your local laptop terminal:
```bash
python client/bench.py --url https://<codespace-name>-8000.app.github.dev --n 200
```
This sends 200 live readings and measures round-trip time, server stages (`parse`, `normalise`, `inference`, `build`), and network overhead.

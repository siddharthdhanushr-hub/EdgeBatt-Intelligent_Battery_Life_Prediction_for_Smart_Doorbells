#!/usr/bin/env bash
# ==============================================================================
# deploy_codespaces.sh -- Deployment & Runner Script for GitHub Codespaces
# ==============================================================================
set -e

echo "=================================================================="
echo "    EdgeBatt: Intelligent Battery Life Prediction (Team 01)"
echo "            GitHub Codespaces Deployment Script"
echo "=================================================================="

MODE="${1:-auto}"

# Check if Docker is available in the current environment
if ! command -v docker &> /dev/null; then
    echo "[Notice] 'docker' command is not installed in this Codespace environment."
    echo "[Notice] GitHub Codespaces itself IS already a secure Linux container!"
    echo "[Notice] Falling back to direct native execution via Uvicorn on port 8000..."
    MODE="local"
fi

if [ "$MODE" = "container" ] || [ "$MODE" = "docker" ] || [ "$MODE" = "auto" ]; then
    echo "[1/3] Building and starting Docker container..."
    docker compose up --build -d
    
    echo "[2/3] Waiting for container health check..."
    sleep 5
    docker compose ps
    
    echo "[3/3] Probing /health endpoint..."
    curl -s http://localhost:8000/health || true
    echo ""
    echo "Container is running at http://localhost:8000"
    echo "IMPORTANT: In the Codespace 'Ports' tab, right-click Port 8000 and select 'Port Visibility -> Public'!"
elif [ "$MODE" = "local" ] || [ "$MODE" = "direct" ]; then
    echo "[1/2] Detecting Python environment..."
    
    # Check which Python binary has pip or is available
    if python -m pip --version &> /dev/null; then
        PY_BIN="python"
    elif python3 -m pip --version &> /dev/null; then
        PY_BIN="python3"
    elif command -v python &> /dev/null; then
        PY_BIN="python"
    else
        PY_BIN="python3"
    fi
    echo "Using Python: $($PY_BIN --version 2>&1) ($PY_BIN)"

    # Check and bootstrap pip if missing
    if ! $PY_BIN -m pip --version &> /dev/null; then
        echo "[Notice] Bootstrapping pip via get-pip.py..."
        curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
        $PY_BIN /tmp/get-pip.py || python3 /tmp/get-pip.py
    fi

    # Install dependencies from requirements.txt
    echo "[Notice] Installing dependencies from requirements.txt..."
    $PY_BIN -m pip install --no-cache-dir -r requirements.txt

    echo "[2/2] Starting Uvicorn server on http://0.0.0.0:8000..."
    echo "Press Ctrl+C to stop the server."
    echo ""
    $PY_BIN -m uvicorn app.main:app --host 0.0.0.0 --port 8000


else
    echo "Usage: ./deploy_codespaces.sh [container|local]"
    exit 1
fi

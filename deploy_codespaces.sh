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
    echo "[1/2] Setting up Python environment..."
    
    if [ ! -d ".venv" ]; then
        echo "[Notice] Creating virtual environment (.venv)..."
        python3 -m venv .venv || python -m venv .venv
    fi
    source .venv/bin/activate

    if [ -f "$HOME/.local/bin/uv" ]; then
        UV_CMD="$HOME/.local/bin/uv"
    elif command -v uv &> /dev/null; then
        UV_CMD="uv"
    else
        UV_CMD=""
    fi

    echo "[Notice] Installing dependencies via high-speed mirror..."
    if [ -n "$UV_CMD" ]; then
        $UV_CMD pip install -i https://mirrors.aliyun.com/pypi/simple/ --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
    else
        pip install -i https://mirrors.aliyun.com/pypi/simple/ --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
    fi

    echo "[2/2] Starting Uvicorn server on http://0.0.0.0:8000..."
    echo "Press Ctrl+C to stop the server."
    echo ""
    uvicorn app.main:app --host 0.0.0.0 --port 8000





else
    echo "Usage: ./deploy_codespaces.sh [container|local]"
    exit 1
fi

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
    
    # Install uv for ultra-fast, SSL-resilient package installation
    if ! command -v uv &> /dev/null && [ ! -f "$HOME/.local/bin/uv" ]; then
        echo "[Notice] Installing ultra-fast, SSL-resilient installer (uv)..."
        curl -LsSf https://astral.sh/uv/install.sh | sh 2>/dev/null || true
    fi

    if [ -f "$HOME/.local/bin/uv" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    fi

    if command -v uv &> /dev/null; then
        echo "[Notice] Using uv for fast installation..."
        if [ ! -d ".venv" ]; then
            uv venv .venv
        fi
        source .venv/bin/activate
        uv pip install -r requirements.txt
    else
        # Fallback to standard venv and pip
        if [ ! -d ".venv" ]; then
            python3 -m venv .venv || python -m venv .venv
        fi
        source .venv/bin/activate
        pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
    fi

    echo "[2/2] Starting Uvicorn server on http://0.0.0.0:8000..."
    echo "Press Ctrl+C to stop the server."
    echo ""
    uvicorn app.main:app --host 0.0.0.0 --port 8000




else
    echo "Usage: ./deploy_codespaces.sh [container|local]"
    exit 1
fi

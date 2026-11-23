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

# 1. Critical Cloud VM Network Fix:
# In Azure/Codespaces, default MTU 1500 causes TLS packet fragmentation drops
# (SSLEOFError / connection timeout to CDN servers). MTU 1350 solves this completely.
echo "[Notice] Applying network optimizations (MTU 1350 & IPv4 priority)..."
sudo ip link set dev eth0 mtu 1350 2>/dev/null || true
for iface in $(ip -o link show 2>/dev/null | awk -F': ' '{print $2}' | grep -v lo); do
    sudo ip link set dev "$iface" mtu 1350 2>/dev/null || true
done
sudo sysctl -w net.ipv6.conf.all.disable_ipv6=1 2>/dev/null || true
sudo sysctl -w net.ipv6.conf.default.disable_ipv6=1 2>/dev/null || true

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

    echo "[Notice] Installing PyTorch (CPU-only, lightweight)..."
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

    echo "[Notice] Installing FastAPI server runtime..."
    pip install --no-cache-dir fastapi "uvicorn[standard]" pydantic numpy requests

    echo "[2/2] Starting Uvicorn server on http://0.0.0.0:8000..."
    echo "=================================================================="
    echo ">> SERVER IS LIVE AT http://0.0.0.0:8000 <<"
    echo ">> IN CODESPACES: GO TO 'PORTS' TAB -> PORT 8000 -> VISIBILITY: PUBLIC <<"
    echo "=================================================================="
    echo ""
    exec uvicorn app.main:app --host 0.0.0.0 --port 8000

else
    echo "Usage: ./deploy_codespaces.sh [container|local]"
    exit 1
fi


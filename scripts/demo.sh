#!/bin/bash
# Start the S4 Research Intelligence live demo stack.
#
# This brings up Ollama + FastAPI + Streamlit via Docker Compose,
# then opens a Cloudflare Tunnel exposing only port 8501 (Streamlit).
#
# Prerequisites:
#   - Docker with GPU support (nvidia-container-toolkit)
#   - cloudflared CLI installed
#   - Mistral 7B model pulled in Ollama

set -euo pipefail

echo "=== S4 Research Intelligence — Live Demo ==="
echo ""

# Start Docker Compose stack
echo "[1/3] Starting Docker Compose stack..."
docker compose -f docker/docker-compose.demo.yml up -d

# Wait for services
echo "[2/3] Waiting for services to start..."
for i in {1..30}; do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "  Backend is healthy."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "  WARNING: Backend health check timed out after 30s."
    fi
    sleep 1
done

# Check if Streamlit is up
for i in {1..15}; do
    if curl -sf http://localhost:8501 > /dev/null 2>&1; then
        echo "  Streamlit is running."
        break
    fi
    sleep 1
done

# Start Cloudflare Tunnel (only port 8501)
echo "[3/3] Starting Cloudflare Tunnel (port 8501 only)..."
echo "  Share the URL below with your interviewer."
echo ""
cloudflared tunnel --url http://localhost:8501

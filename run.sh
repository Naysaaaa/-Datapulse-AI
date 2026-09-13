#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "============================================="
echo "⚡ Starting DataPulse AI Analytics Platform"
echo "============================================="

# Resolve absolute path of directory containing this script
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$PROJECT_DIR"

# Check if virtual environment exists, activate it
if [ -d ".venv" ]; then
    echo "✅ Found .venv, activating virtual environment..."
    source .venv/bin/activate
else
    echo "⚠️ Warning: .venv not found. Attempting to run with system python3..."
fi

# Ensure data directory exists
mkdir -p "$PROJECT_DIR/data"

# Setup trap to terminate background backend server on exit (Ctrl+C)
trap 'echo "Stopping servers..."; kill $(jobs -p) 2>/dev/null || true' EXIT

# Start FastAPI Backend in the background
echo "🚀 Starting FastAPI Backend Service (http://localhost:8000)..."
python3 -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 &

# Wait for FastAPI to initialize
echo "⏱️ Waiting for backend service to bind to port 8000..."
sleep 3

# Start Streamlit Dashboard in the foreground
echo "🖥️ Starting Streamlit Dashboard (http://localhost:8501)..."
python3 -m streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0

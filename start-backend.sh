#!/bin/bash
cd "$(dirname "$0")/backend"
echo "Installing Python dependencies..."
pip3 install -r requirements.txt
echo ""
echo "Starting backend on http://localhost:8000"
echo "Make sure ANTHROPIC_API_KEY is set in your environment."
echo ""
uvicorn main:app --reload --host 0.0.0.0 --port 8000

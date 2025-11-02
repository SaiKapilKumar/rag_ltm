#!/bin/bash

# RAG with Long-Term Memory - Startup Script

echo "Starting RAG with Long-Term Memory System..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Start API server in background
echo "Starting FastAPI server..."
python -m uvicorn src.api.main:app --reload --port 8000 &
API_PID=$!

# Wait for API to start
echo "Waiting for API to initialize..."
sleep 5

# Check if API is running
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "Warning: API may not be fully initialized yet"
fi

# Start Streamlit
echo "Starting Streamlit UI..."
streamlit run src/ui/streamlit_app.py --server.port 8501 &
STREAMLIT_PID=$!

echo ""
echo "System is starting up!"
echo "API: http://localhost:8000"
echo "UI: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop both services"

# Wait for interrupt
trap "kill $API_PID $STREAMLIT_PID 2>/dev/null; exit" INT TERM

wait

#!/bin/bash

# Setup script for RAG with Long-Term Memory

echo "Setting up RAG with Long-Term Memory..."

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
required_version="3.9"

if (( $(echo "$python_version < $required_version" | bc -l) )); then
    echo "Python 3.9+ is required. Current version: $python_version"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create data directories
echo "Creating data directories..."
mkdir -p data/memory
mkdir -p data/embeddings
mkdir -p data/documents

# Copy environment file
if [ ! -f "../.env" ]; then
    echo "Warning: .env file not found in parent directory"
    echo "Please ensure your Azure OpenAI credentials are configured"
else
    echo ".env file found in parent directory"
fi

echo ""
echo "Setup complete!"
echo "To start the system, run: ./run.sh"
echo "Make sure your Azure OpenAI credentials are configured in the .env file"

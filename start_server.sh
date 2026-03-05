#!/bin/bash
# Start Staff Badge Detection Server

echo "=========================================="
echo "Staff Badge Detection System"
echo "=========================================="
echo ""

# Activate virtual environment if not already activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Check if dependencies are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Start server
echo ""
echo "Starting server..."
python server.py

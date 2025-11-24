#!/bin/bash

# Setup script for The Maze and The Map: Understanding Transformers
# This script sets up the Python environment and installs all dependencies

set -e  # Exit on error

echo "=================================="
echo "Transformer Maze Setup"
echo "=================================="
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.8"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "Error: Python 3.8 or higher is required"
    echo "Current version: $PYTHON_VERSION"
    exit 1
fi

echo "✓ Python version $PYTHON_VERSION is compatible"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
echo "✓ pip upgraded"
echo ""

# Install requirements
echo "Installing dependencies..."
echo "This may take a few minutes..."
pip install -r requirements.txt

echo "✓ All dependencies installed"
echo ""

# Install package in development mode
echo "Installing transformer-maze package..."
pip install -e .
echo "✓ Package installed in development mode"
echo ""

# Test imports
echo "Testing imports..."
python3 -c "
import torch
import numpy as np
import matplotlib
import seaborn
import jupyter
print('✓ All core imports successful')
"
echo ""

# Create output directories
echo "Creating output directories..."
mkdir -p outputs
mkdir -p outputs/models
mkdir -p outputs/figures
mkdir -p outputs/data
echo "✓ Output directories created"
echo ""

# Print success message
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "To get started:"
echo "  1. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Launch Jupyter:"
echo "     jupyter notebook"
echo ""
echo "  3. Open notebooks in order:"
echo "     - notebooks/01_the_mouse_rnn_maze.ipynb"
echo "     - notebooks/02_the_map_attention_basics.ipynb"
echo "     - etc."
echo ""
echo "For more information, see README.md"
echo ""

# Check for GPU support
echo "Checking for GPU support..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✓ GPU available: {torch.cuda.get_device_name(0)}')
    print(f'  CUDA version: {torch.version.cuda}')
else:
    print('ℹ No GPU detected - training will use CPU')
    print('  (This is fine for the educational examples)')
"
echo ""

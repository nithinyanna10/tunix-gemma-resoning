#!/bin/bash

# Setup script for Tunix Gemma Reasoning project

set -e

echo "=========================================="
echo "Tunix Gemma Reasoning - Setup"
echo "=========================================="

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r scripts/requirements.txt

# Install Tunix from GitHub
echo ""
echo "Installing Tunix from GitHub..."
pip install git+https://github.com/google/tunix.git

# Setup JAX for CPU/GPU
echo ""
echo "Setting up JAX..."
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected, installing JAX with CUDA support..."
    pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
else
    echo "No GPU detected, using CPU version..."
    pip install --upgrade jax jaxlib
fi

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p model_checkpoints
mkdir -p logs
mkdir -p gemma

# Generate sample data if needed
echo ""
echo "Generating sample data..."
if [ ! -f "data/synthetic_reasoning_set.jsonl" ] || [ ! -s "data/synthetic_reasoning_set.jsonl" ]; then
    python scripts/generate_synthetic_data.py
fi

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "import jax; print(f'JAX version: {jax.__version__}')"
python3 -c "import jax; print(f'JAX devices: {jax.devices()}')"
python3 -c "import flax; print(f'Flax version: {flax.__version__}')"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Download Gemma model (using Ollama or other method)"
echo "3. Update configs/training_config.yaml with your model path"
echo "4. Run training: python scripts/trainer.py --config configs/training_config.yaml"
echo ""
echo "For Kaggle TPU training, use the notebook: notebook/tunix_gemma_train.ipynb"
echo ""


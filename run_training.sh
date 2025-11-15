#!/bin/bash

# Quick training script - shows everything in terminal

cd "$(dirname "$0")"
source venv/bin/activate

echo "🚀 Starting Tunix Gemma Reasoning Training..."
echo ""

python scripts/run_training_verbose.py --config configs/training_config.yaml --model gemma3:1b


# Quick Start Guide

## Prerequisites

- Python 3.8+
- pip
- (Optional) CUDA-capable GPU for faster training

## Setup

### 1. Clone/Download the Project

```bash
cd tunix-gemma-reasoning
```

### 2. Run Setup Script

```bash
chmod +x setup.sh
./setup.sh
```

Or manually:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r scripts/requirements.txt
pip install git+https://github.com/google/tunix.git

# Setup JAX (CPU or GPU)
# For CPU:
pip install jax jaxlib

# For GPU (CUDA 12):
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### 3. Download Gemma Model

#### Option A: Using Ollama (Recommended for local)

```bash
# Install Ollama if not already installed
# Visit: https://ollama.ai

# Pull Gemma model
ollama pull gemma2:2b
# or
ollama pull gemma3:1b

# Convert to format compatible with Tunix
# (You may need a conversion script - see scripts/convert_ollama_model.py)
```

#### Option B: Download from Hugging Face

```bash
# Install huggingface-hub
pip install huggingface-hub

# Download Gemma2 2B
python -c "from huggingface_hub import snapshot_download; snapshot_download('google/gemma-2-2b', local_dir='gemma/')"
```

### 4. Prepare Training Data

```bash
# Generate synthetic data (if needed)
python scripts/generate_synthetic_data.py

# Or use your own data in JSONL format:
# {"prompt": "Question", "reasoning": "Step-by-step", "answer": "Answer"}
```

### 5. Configure Training

Edit `configs/training_config.yaml`:

```yaml
model:
  base_model: "gemma2-2b"
  model_path: "gemma/"  # Update this path
  tokenizer_path: "gemma/"

data:
  train_file: "data/synthetic_reasoning_set.jsonl"
  eval_file: "data/synthetic_reasoning_set.jsonl"
```

## Training

### Local Training (CPU/GPU)

```bash
python scripts/trainer.py --config configs/training_config.yaml
```

### Kaggle TPU Training

1. Upload `notebook/tunix_gemma_train.ipynb` to Kaggle
2. Enable TPU in notebook settings
3. Upload your training data
4. Run the notebook

## Inference

### Command Line

```bash
# Single question
python scripts/inference.py \
    --model_path model_checkpoints/final_model \
    --prompt "If a train travels 120 miles in 2 hours, what is its average speed?"

# Interactive mode
python scripts/inference.py \
    --model_path model_checkpoints/final_model \
    --interactive

# Batch inference
python scripts/inference.py \
    --model_path model_checkpoints/final_model \
    --questions_file questions.txt \
    --output_file results.json
```

### Jupyter Notebook

```bash
jupyter notebook notebook/tunix_gemma_inference.ipynb
```

## Project Structure

```
tunix-gemma-reasoning/
├── configs/              # Configuration files
├── data/                 # Training data
├── scripts/              # Python scripts
├── notebook/             # Jupyter notebooks
├── model_checkpoints/    # Saved models
├── gemma/                # Gemma model files
└── writeup/              # Kaggle writeup materials
```

## Troubleshooting

### JAX Installation Issues

- **CPU**: `pip install jax jaxlib`
- **GPU**: Install CUDA toolkit first, then `pip install "jax[cuda12]"`

### Out of Memory

- Reduce `batch_size` in config
- Enable `gradient_checkpointing`
- Set `offload_to_cpu: true`

### Model Loading Issues

- Ensure model files are in correct format
- Check `model_path` in config
- Verify tokenizer files are present

## Next Steps

1. Review `README.md` for detailed documentation
2. Check `writeup/kaggle_writeup.md` for submission details
3. See `configs/` for all configuration options
4. Review `scripts/` for implementation details

## Resources

- [Tunix GitHub](https://github.com/google/tunix/)
- [Tunix Documentation](https://tunix.readthedocs.io/)
- [JAX Documentation](https://jax.dev)
- [Gemma Models](https://ai.google.dev/gemma)


# Tunix Gemma Reasoning Model

Train a Gemma model to show its work using Google's Tunix library for LLM post-training.

## Overview

This project fine-tunes Google's Gemma model (Gemma2 2B or Gemma3 1B) using Tunix to teach it step-by-step reasoning. The model learns to output reasoning traces in the format:

```
<reasoning>model_thinking_trace</reasoning>
<answer>model_answer</answer>
```

## Project Structure

```
tunix-gemma-reasoning/
├── README.md
├── configs/              # Configuration files
├── data/                 # Training data and rubrics
├── notebook/             # Jupyter notebooks for Kaggle
├── scripts/              # Training and utility scripts
├── video/                # Video materials
├── writeup/              # Kaggle writeup materials
├── model_checkpoints/    # Saved model checkpoints
└── gemma/                # Gemma model files (from Ollama)
```

## Setup

### Prerequisites

```bash
# Install Python dependencies
pip install jax jaxlib flax optax transformers datasets numpy pandas
pip install tunix  # Install from GitHub: https://github.com/google/tunix/

# For CPU/GPU (non-TPU) usage
export JAX_PLATFORMS=cpu  # or 'gpu' if you have CUDA
```

### Download Gemma Model

```bash
# Using Ollama
ollama pull gemma2:2b
# or
ollama pull gemma3:1b

# Convert to format compatible with Tunix (see scripts/convert_ollama_model.py)
```

## Training

### Single Session Training

```bash
python scripts/trainer.py --config configs/training_config.yaml
```

### Multi-Session Training

```bash
# Session 1
python scripts/trainer.py --config configs/training_config.yaml --checkpoint_dir model_checkpoints/session1

# Session 2 (resume from checkpoint)
python scripts/trainer.py --config configs/training_config.yaml --checkpoint_dir model_checkpoints/session2 --resume_from model_checkpoints/session1/checkpoint_epoch_5
```

## Inference

```bash
python scripts/inference.py --model_path model_checkpoints/final_model --prompt "Your question here"
```

Or use the Jupyter notebook:
```bash
jupyter notebook notebook/tunix_gemma_inference.ipynb
```

## Data Format

Training data should be in JSONL format:
```json
{"prompt": "Question here", "reasoning": "Step-by-step reasoning", "answer": "Final answer"}
```

## Reward Function

The model uses rubric-based rewards that evaluate:
- Reasoning quality
- Answer correctness
- Trace coherence
- Step-by-step clarity

## Evaluation

The model is evaluated on:
- Creative writing
- Creative ideation
- Summarization
- Math
- Coding
- Basic science
- Other domains

## References

- [Tunix GitHub](https://github.com/google/tunix/)
- [Tunix Documentation](https://tunix.readthedocs.io/en/latest/index.html)
- [JAX Documentation](https://jax.dev)
- [Flax Documentation](https://flax.readthedocs.io/)

## License

CC BY-SA 4.0


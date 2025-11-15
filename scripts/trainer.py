"""
Main training script for Tunix GRPO fine-tuning.
Adapted for CPU/GPU (non-TPU) usage.
"""

import argparse
import yaml
import os
from pathlib import Path
from datetime import datetime
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

# Note: This is a simplified trainer. In practice, you would use Tunix's actual API
# For full implementation, refer to: https://github.com/google/tunix/

from dataset_loader import ReasoningDataset, create_dataloader
from rubric_reward import RubricReward

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_jax(device: str = "cpu"):
    """Setup JAX for CPU/GPU."""
    if device == "cpu":
        os.environ['JAX_PLATFORMS'] = 'cpu'
    elif device == "gpu":
        os.environ['JAX_PLATFORMS'] = 'gpu'
    
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX platform: {jax.default_backend()}")

def initialize_model(config: dict):
    """
    Initialize Gemma model.
    In actual implementation, this would load from Tunix model registry.
    """
    model_name = config['model']['base_model']
    model_path = config['model']['model_path']
    
    print(f"Loading model: {model_name} from {model_path}")
    
    # Placeholder: Actual implementation would use Tunix model loading
    # from tunix import models
    # model = models.load_gemma(model_name, model_path)
    
    # For now, return None as placeholder
    # In real code, return the loaded model
    return None

def train_step(model, batch, optimizer_state, reward_fn, config):
    """
    Single training step using GRPO.
    
    This is a simplified version. Actual Tunix GRPO implementation
    would handle policy optimization, KL penalties, etc.
    """
    # Placeholder implementation
    # Actual GRPO would:
    # 1. Generate multiple responses per prompt (group_size)
    # 2. Compute rewards for each response
    # 3. Compute relative rewards within groups
    # 4. Update policy using PPO-style objective
    
    prompts = [item['prompt'] for item in batch]
    questions = [item['question'] for item in batch]
    
    # Generate responses (placeholder - would use actual model)
    responses = []
    for prompt in prompts:
        # In real code: response = model.generate(prompt, **generation_params)
        response = "<reasoning>\nPlaceholder reasoning\n</reasoning>\n<answer>\nPlaceholder answer\n</answer>"
        responses.append(response)
    
    # Compute rewards
    rewards = []
    for question, response in zip(questions, responses):
        reward_dict = reward_fn.evaluate(question, response)
        rewards.append(reward_dict['total'])
    
    # Convert to JAX array
    rewards = jnp.array(rewards)
    
    # Placeholder loss computation
    # Actual GRPO would compute policy loss, KL penalty, etc.
    loss = -jnp.mean(rewards)  # Negative because we maximize reward
    
    # Placeholder gradient update
    # In real code, would use Tunix's training utilities
    grads = jax.grad(lambda x: x)(loss)
    
    return loss, rewards, optimizer_state

def train(config: dict, resume_from: str = None):
    """
    Main training loop.
    
    Args:
        config: Training configuration dictionary
        resume_from: Path to checkpoint to resume from
    """
    print("=" * 50)
    print("Starting Tunix GRPO Training")
    print("=" * 50)
    
    # Setup JAX
    device = config['hardware']['device']
    setup_jax(device)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = ReasoningDataset(
        config['data']['train_file'],
        max_samples=config['data'].get('max_train_samples'),
        shuffle=True
    )
    
    eval_dataset = ReasoningDataset(
        config['data']['eval_file'],
        max_samples=config['data'].get('max_eval_samples'),
        shuffle=False
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    
    # Initialize reward function
    reward_fn = RubricReward(config['reward']['config_file'])
    
    # Initialize model
    model = initialize_model(config)
    
    # Training parameters
    batch_size = config['training']['batch_size']
    gradient_accumulation_steps = config['training']['gradient_accumulation_steps']
    num_epochs = config['training']['num_epochs']
    save_steps = config['training']['save_steps']
    eval_steps = config['training']['eval_steps']
    logging_steps = config['training']['logging_steps']
    
    # Create output directory
    output_dir = Path(config['output']['output_dir'])
    run_name = config['output']['run_name'].format(
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {run_dir}")
    
    # Save config
    with open(run_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # Training loop
    print("\nStarting training...")
    global_step = 0
    optimizer_state = None  # Would initialize actual optimizer state
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        dataloader = create_dataloader(train_dataset, batch_size, shuffle=True)
        accumulated_loss = 0.0
        accumulated_rewards = []
        
        for step, batch in enumerate(dataloader):
            # Training step
            loss, rewards, optimizer_state = train_step(
                model, batch, optimizer_state, reward_fn, config
            )
            
            accumulated_loss += float(loss)
            accumulated_rewards.extend(rewards.tolist())
            
            # Gradient accumulation
            if (step + 1) % gradient_accumulation_steps == 0:
                global_step += 1
                
                # Logging
                if global_step % logging_steps == 0:
                    avg_loss = accumulated_loss / gradient_accumulation_steps
                    avg_reward = sum(accumulated_rewards) / len(accumulated_rewards)
                    print(
                        f"Step {global_step}: "
                        f"Loss={avg_loss:.4f}, "
                        f"Avg Reward={avg_reward:.4f}"
                    )
                    accumulated_loss = 0.0
                    accumulated_rewards = []
                
                # Evaluation
                if global_step % eval_steps == 0:
                    print("Running evaluation...")
                    # Placeholder: would run actual evaluation
                    eval_reward = 0.0  # Placeholder
                    print(f"Eval reward: {eval_reward:.4f}")
                
                # Save checkpoint
                if global_step % save_steps == 0:
                    checkpoint_dir = run_dir / f"checkpoint_step_{global_step}"
                    checkpoint_dir.mkdir(exist_ok=True)
                    print(f"Saving checkpoint to {checkpoint_dir}")
                    # Placeholder: would save actual model checkpoint
                    # model.save_checkpoint(checkpoint_dir)
    
    # Final save
    final_checkpoint_dir = run_dir / "final_checkpoint"
    final_checkpoint_dir.mkdir(exist_ok=True)
    print(f"\nSaving final checkpoint to {final_checkpoint_dir}")
    # model.save_checkpoint(final_checkpoint_dir)
    
    print("\nTraining completed!")

def main():
    parser = argparse.ArgumentParser(description="Train Gemma with Tunix GRPO")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training config file"
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Custom checkpoint directory"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override checkpoint dir if provided
    if args.checkpoint_dir:
        config['output']['output_dir'] = args.checkpoint_dir
    
    # Train
    train(config, resume_from=args.resume_from)

if __name__ == "__main__":
    main()


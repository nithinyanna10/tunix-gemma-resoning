"""
Training script that uses Ollama for model inference during training.
This is a hybrid approach: uses Ollama API for generation, but we still need
actual model weights for Tunix training. For now, this demonstrates the pipeline.
"""

import argparse
import json
import yaml
from pathlib import Path
from datetime import datetime
import time
from typing import List, Dict

from ollama_client import OllamaClient
from dataset_loader import ReasoningDataset, create_dataloader
from reward_functions import compute_reward, extract_reasoning_and_answer
from rubric_reward import RubricReward

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def format_prompt(question: str, prompt_template: str = None) -> str:
    """Format question into prompt."""
    if prompt_template is None:
        prompt_template = (
            "Answer the following question. Show your reasoning step by step, "
            "then provide your final answer.\n\n"
            "Question: {question}\n\n"
            "Format your response as:\n"
            "<reasoning>\n"
            "Your step-by-step reasoning here\n"
            "</reasoning>\n"
            "<answer>\n"
            "Your final answer here\n"
            "</answer>"
        )
    return prompt_template.format(question=question)

def generate_with_ollama(
    client: OllamaClient,
    model: str,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 512
) -> str:
    """Generate response using Ollama."""
    return client.generate(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens
    )

def evaluate_batch(
    client: OllamaClient,
    model: str,
    batch: List[Dict],
    reward_fn: RubricReward,
    config: dict,
    group_size: int = 4
) -> Dict:
    """
    Evaluate a batch of prompts using Ollama.
    Generates multiple responses per prompt for GRPO.
    """
    all_responses = []
    all_rewards = []
    
    for item in batch:
        prompt = format_prompt(item['question'], config['data'].get('prompt_template'))
        question = item['question']
        
        # Generate group_size responses per prompt
        group_responses = []
        group_rewards = []
        
        for _ in range(group_size):
            response = generate_with_ollama(
                client,
                model,
                prompt,
                temperature=config['model'].get('temperature', 0.7),
                max_tokens=config['model'].get('max_new_tokens', 512)
            )
            
            # Compute reward
            reward_dict = reward_fn.evaluate(question, response)
            reward = reward_dict['total']
            
            group_responses.append(response)
            group_rewards.append(reward)
        
        all_responses.append(group_responses)
        all_rewards.append(group_rewards)
    
    # Compute relative rewards (GRPO style)
    # In GRPO, we compare within groups
    relative_rewards = []
    for group_rewards in all_rewards:
        # Normalize rewards within group
        if len(group_rewards) > 1:
            mean_reward = sum(group_rewards) / len(group_rewards)
            relative = [r - mean_reward for r in group_rewards]
        else:
            relative = group_rewards
        relative_rewards.append(relative)
    
    return {
        'responses': all_responses,
        'rewards': all_rewards,
        'relative_rewards': relative_rewards,
        'avg_reward': sum([sum(r) for r in all_rewards]) / sum([len(r) for r in all_rewards])
    }

def train_with_ollama(config: dict, ollama_model: str = "gemma3:1b"):
    """
    Training loop using Ollama for generation.
    Note: This is a demonstration. Actual Tunix training requires model weights.
    """
    print("=" * 60)
    print("Tunix Gemma Reasoning - Training with Ollama")
    print("=" * 60)
    
    # Initialize Ollama client
    client = OllamaClient()
    
    print("\nChecking Ollama connection...")
    if not client.check_connection():
        print("ERROR: Cannot connect to Ollama on port 11434")
        print("Make sure Ollama is running: ollama serve")
        return
    
    print("✓ Ollama connected")
    
    # Check if model is available
    models = client.list_models()
    if ollama_model not in models:
        print(f"\nModel {ollama_model} not found. Available models: {models}")
        print(f"Pulling {ollama_model}...")
        if not client.pull_model(ollama_model):
            print(f"Failed to pull {ollama_model}")
            return
    else:
        print(f"✓ Model {ollama_model} is available")
    
    # Load datasets
    print("\nLoading datasets...")
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
    
    # Training parameters
    batch_size = config['training']['batch_size']
    num_epochs = config['training']['num_epochs']
    group_size = config['training'].get('group_size', 4)
    eval_steps = config['training'].get('eval_steps', 250)
    logging_steps = config['training'].get('logging_steps', 50)
    
    # Create output directory
    output_dir = Path(config['output']['output_dir'])
    run_name = config['output']['run_name'].format(
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {run_dir}")
    
    # Save config
    with open(run_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting Training Loop")
    print("=" * 60)
    print("\nNOTE: This is a demonstration using Ollama API.")
    print("For actual Tunix training, you need model weights in JAX format.")
    print("Consider downloading from HuggingFace and converting to JAX/Flax format.\n")
    
    global_step = 0
    all_metrics = []
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")
        
        dataloader = create_dataloader(train_dataset, batch_size, shuffle=True)
        
        for step, batch in enumerate(dataloader):
            # Evaluate batch
            results = evaluate_batch(
                client,
                ollama_model,
                batch,
                reward_fn,
                config,
                group_size
            )
            
            global_step += 1
            
            # Logging
            if global_step % logging_steps == 0:
                avg_reward = results['avg_reward']
                print(f"\nStep {global_step}:")
                print(f"  Average Reward: {avg_reward:.4f}")
                print(f"  Format Compliance: {sum([1 for r in results['responses'][0] if '<reasoning>' in r and '<answer>' in r]) / len(results['responses'][0]):.2%}")
                
                # Show example
                if results['responses']:
                    example = results['responses'][0][0]
                    reasoning, answer = extract_reasoning_and_answer(example)
                    print(f"\n  Example Response:")
                    print(f"  Reasoning: {reasoning[:100]}...")
                    print(f"  Answer: {answer[:50]}...")
                
                all_metrics.append({
                    'step': global_step,
                    'avg_reward': avg_reward,
                    'epoch': epoch + 1
                })
            
            # Evaluation
            if global_step % eval_steps == 0 and global_step > 0:
                print("\nRunning evaluation...")
                eval_results = evaluate_batch(
                    client,
                    ollama_model,
                    eval_dataset.get_batch(list(range(min(10, len(eval_dataset))))),
                    reward_fn,
                    config,
                    group_size=2  # Smaller for eval
                )
                print(f"Eval Average Reward: {eval_results['avg_reward']:.4f}")
            
            # Save metrics
            if global_step % 100 == 0:
                metrics_file = run_dir / "metrics.json"
                with open(metrics_file, 'w') as f:
                    json.dump(all_metrics, f, indent=2)
        
        # Small delay to avoid overwhelming Ollama
        time.sleep(0.1)
    
    # Final save
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"\nMetrics saved to: {run_dir / 'metrics.json'}")
    print(f"Config saved to: {run_dir / 'config.yaml'}")
    print("\nNOTE: This was a demonstration using Ollama API.")
    print("For actual model fine-tuning with Tunix, you need:")
    print("1. Model weights in JAX/Flax format")
    print("2. Tunix library properly installed")
    print("3. TPU or GPU for training")

def main():
    parser = argparse.ArgumentParser(description="Train with Ollama API")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training config"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemma3:1b",
        help="Ollama model name"
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama API URL"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Train
    train_with_ollama(config, args.model)

if __name__ == "__main__":
    main()


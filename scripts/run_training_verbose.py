"""
Verbose training script that shows detailed progress in terminal.
"""

import sys
import time
from train_with_ollama import *
from ollama_client import OllamaClient
from dataset_loader import ReasoningDataset, create_dataloader
from reward_functions import extract_reasoning_and_answer
from rubric_reward import RubricReward

def print_separator(char="=", length=70):
    """Print a separator line."""
    print(char * length)

def print_step_header(step, total_steps=None):
    """Print step header."""
    if total_steps:
        print(f"\n{'='*70}")
        print(f"STEP {step}/{total_steps}")
        print(f"{'='*70}")
    else:
        print(f"\n{'='*70}")
        print(f"STEP {step}")
        print(f"{'='*70}")

def train_verbose(config: dict, ollama_model: str = "gemma3:1b"):
    """Verbose training with detailed terminal output."""
    print_separator("=", 70)
    print("TUNIX GEMMA REASONING - TRAINING WITH OLLAMA")
    print_separator("=", 70)
    
    # Initialize Ollama client
    print("\n[1/5] Initializing Ollama client...")
    client = OllamaClient()
    
    if not client.check_connection():
        print("❌ ERROR: Cannot connect to Ollama on port 11434")
        print("   Make sure Ollama is running: ollama serve")
        return
    print("   ✓ Ollama connected")
    
    # Check model
    print(f"\n[2/5] Checking model: {ollama_model}...")
    models = client.list_models()
    if ollama_model not in models:
        print(f"   ⚠ Model {ollama_model} not found. Available: {models}")
        print(f"   Pulling {ollama_model}...")
        if not client.pull_model(ollama_model):
            print(f"   ❌ Failed to pull {ollama_model}")
            return
    print(f"   ✓ Model {ollama_model} available")
    
    # Load datasets
    print(f"\n[3/5] Loading datasets...")
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
    print(f"   ✓ Train samples: {len(train_dataset)}")
    print(f"   ✓ Eval samples: {len(eval_dataset)}")
    
    # Initialize reward function
    print(f"\n[4/5] Initializing reward function...")
    reward_fn = RubricReward(config['reward']['config_file'])
    print("   ✓ Reward function ready")
    
    # Training parameters
    batch_size = config['training']['batch_size']
    num_epochs = config['training']['num_epochs']
    group_size = config['training'].get('group_size', 4)
    eval_steps = config['training'].get('eval_steps', 10)
    logging_steps = config['training'].get('logging_steps', 5)
    
    # Output directory
    from pathlib import Path
    from datetime import datetime
    output_dir = Path(config['output']['output_dir'])
    run_name = config['output']['run_name'].format(
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[5/5] Setting up output directory...")
    print(f"   ✓ Output: {run_dir}")
    
    # Save config
    import yaml
    with open(run_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    print_separator("=", 70)
    print("STARTING TRAINING LOOP")
    print_separator("=", 70)
    print(f"\nConfiguration:")
    print(f"  - Model: {ollama_model}")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Group size: {group_size}")
    print(f"  - Logging every: {logging_steps} steps")
    print(f"  - Eval every: {eval_steps} steps")
    print()
    
    global_step = 0
    all_metrics = []
    
    for epoch in range(num_epochs):
        print_separator("=", 70)
        print(f"EPOCH {epoch + 1}/{num_epochs}")
        print_separator("=", 70)
        
        dataloader = create_dataloader(train_dataset, batch_size, shuffle=True)
        total_batches = len(list(create_dataloader(train_dataset, batch_size, shuffle=False)))
        
        for batch_idx, batch in enumerate(dataloader):
            print(f"\n📦 Processing batch {batch_idx + 1}/{total_batches}...")
            
            # Show questions
            for i, item in enumerate(batch):
                print(f"   Q{i+1}: {item['question'][:60]}...")
            
            # Generate responses
            print(f"\n   🤖 Generating {group_size} responses per question...")
            all_responses = []
            all_rewards = []
            
            for item_idx, item in enumerate(batch):
                prompt = format_prompt(item['question'], config['data'].get('prompt_template'))
                question = item['question']
                
                group_responses = []
                group_rewards = []
                
                for group_idx in range(group_size):
                    print(f"      Generating response {group_idx + 1}/{group_size} for Q{item_idx + 1}...", end=" ")
                    sys.stdout.flush()
                    
                    response = generate_with_ollama(
                        client,
                        ollama_model,
                        prompt,
                        temperature=config['model'].get('temperature', 0.7),
                        max_tokens=config['model'].get('max_new_tokens', 512)
                    )
                    
                    # Compute reward
                    reward_dict = reward_fn.evaluate(question, response)
                    reward = reward_dict['total']
                    
                    group_responses.append(response)
                    group_rewards.append(reward)
                    print(f"✓ (reward: {reward:.3f})")
                
                all_responses.append(group_responses)
                all_rewards.append(group_rewards)
            
            # Compute relative rewards
            relative_rewards = []
            for group_rewards in all_rewards:
                if len(group_rewards) > 1:
                    mean_reward = sum(group_rewards) / len(group_rewards)
                    relative = [r - mean_reward for r in group_rewards]
                else:
                    relative = group_rewards
                relative_rewards.append(relative)
            
            avg_reward = sum([sum(r) for r in all_rewards]) / sum([len(r) for r in all_rewards])
            global_step += 1
            
            # Logging
            if global_step % logging_steps == 0:
                print(f"\n   📊 Step {global_step} Metrics:")
                print(f"      Average Reward: {avg_reward:.4f}")
                
                # Format compliance
                format_count = 0
                total_responses = 0
                for group in all_responses:
                    for resp in group:
                        total_responses += 1
                        if '<reasoning>' in resp and '<answer>' in resp:
                            format_count += 1
                format_pct = (format_count / total_responses * 100) if total_responses > 0 else 0
                print(f"      Format Compliance: {format_pct:.1f}% ({format_count}/{total_responses})")
                
                # Show best response
                best_idx = (0, 0)
                best_reward = all_rewards[0][0]
                for i, group in enumerate(all_rewards):
                    for j, r in enumerate(group):
                        if r > best_reward:
                            best_reward = r
                            best_idx = (i, j)
                
                best_response = all_responses[best_idx[0]][best_idx[1]]
                reasoning, answer = extract_reasoning_and_answer(best_response)
                
                print(f"\n   🏆 Best Response (reward: {best_reward:.3f}):")
                print(f"      Question: {batch[best_idx[0]]['question'][:50]}...")
                if reasoning:
                    print(f"      Reasoning: {reasoning[:100]}...")
                if answer:
                    print(f"      Answer: {answer[:50]}...")
                else:
                    print(f"      Answer: (in reasoning)")
                
                all_metrics.append({
                    'step': global_step,
                    'avg_reward': avg_reward,
                    'format_compliance': format_pct,
                    'epoch': epoch + 1
                })
            
            # Evaluation
            if global_step % eval_steps == 0 and global_step > 0:
                print(f"\n   🔍 Running evaluation...")
                eval_batch = eval_dataset.get_batch(list(range(min(5, len(eval_dataset)))))
                eval_results = evaluate_batch(
                    client,
                    ollama_model,
                    eval_batch,
                    reward_fn,
                    config,
                    group_size=2
                )
                print(f"      Eval Average Reward: {eval_results['avg_reward']:.4f}")
            
            # Small delay
            time.sleep(0.1)
    
    # Final save
    import json
    metrics_file = run_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print_separator("=", 70)
    print("TRAINING COMPLETE!")
    print_separator("=", 70)
    print(f"\n📁 Results saved to: {run_dir}")
    print(f"   - Config: {run_dir / 'config.yaml'}")
    print(f"   - Metrics: {metrics_file}")
    print(f"\n📈 Final Metrics:")
    if all_metrics:
        final_metric = all_metrics[-1]
        print(f"   - Final Reward: {final_metric['avg_reward']:.4f}")
        print(f"   - Format Compliance: {final_metric['format_compliance']:.1f}%")
        print(f"   - Total Steps: {len(all_metrics)}")
    print()

if __name__ == "__main__":
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="Verbose training with Ollama")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--model", type=str, default="gemma3:1b")
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    train_verbose(config, args.model)


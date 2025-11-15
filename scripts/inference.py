"""
Inference script for trained reasoning model.
Loads model and generates reasoning traces for new questions.
"""

import argparse
import yaml
from pathlib import Path
import jax

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model(model_path: str, config: dict):
    """
    Load trained model from checkpoint.
    
    In actual implementation, this would use Tunix's model loading utilities.
    """
    print(f"Loading model from {model_path}")
    
    # Placeholder: Actual implementation would use Tunix
    # from tunix import models
    # model = models.load_from_checkpoint(model_path, config)
    
    # For now, return None as placeholder
    return None

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

def generate_response(model, prompt: str, config: dict) -> str:
    """
    Generate response from model.
    
    Args:
        model: Loaded model
        prompt: Input prompt
        config: Generation configuration
    
    Returns:
        Generated text
    """
    generation_params = {
        'max_new_tokens': config['model'].get('max_new_tokens', 512),
        'temperature': config['model'].get('temperature', 0.7),
        'top_p': config['model'].get('top_p', 0.9),
        'top_k': config['model'].get('top_k', 50),
        'do_sample': config['model'].get('do_sample', True)
    }
    
    # Placeholder: Actual implementation would use model.generate()
    # response = model.generate(prompt, **generation_params)
    
    # For demonstration, return placeholder
    response = (
        "<reasoning>\n"
        "This is a placeholder response. In actual implementation, "
        "the model would generate reasoning here.\n"
        "</reasoning>\n"
        "<answer>\n"
        "Placeholder answer\n"
        "</answer>"
    )
    
    return response

def extract_reasoning_and_answer(text: str) -> tuple[str, str]:
    """Extract reasoning and answer from model output."""
    import re
    
    reasoning_pattern = r'<reasoning>(.*?)</reasoning>'
    answer_pattern = r'<answer>(.*?)</answer>'
    
    reasoning_match = re.search(reasoning_pattern, text, re.DOTALL)
    answer_match = re.search(answer_pattern, text, re.DOTALL)
    
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    answer = answer_match.group(1).strip() if answer_match else ""
    
    return reasoning, answer

def interactive_inference(model, config: dict):
    """Interactive inference loop."""
    print("=" * 50)
    print("Interactive Inference Mode")
    print("Type 'quit' to exit")
    print("=" * 50)
    
    while True:
        question = input("\nEnter your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if not question:
            continue
        
        # Format prompt
        prompt = format_prompt(question, config['data'].get('prompt_template'))
        
        # Generate response
        print("\nGenerating response...")
        response = generate_response(model, prompt, config)
        
        # Extract and display
        reasoning, answer = extract_reasoning_and_answer(response)
        
        print("\n" + "=" * 50)
        print("REASONING:")
        print("-" * 50)
        print(reasoning)
        print("\n" + "=" * 50)
        print("ANSWER:")
        print("-" * 50)
        print(answer)
        print("=" * 50)

def batch_inference(model, questions: list, config: dict):
    """Run inference on a batch of questions."""
    results = []
    
    for question in questions:
        prompt = format_prompt(question, config['data'].get('prompt_template'))
        response = generate_response(model, prompt, config)
        reasoning, answer = extract_reasoning_and_answer(response)
        
        results.append({
            'question': question,
            'reasoning': reasoning,
            'answer': answer,
            'full_response': response
        })
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run inference with trained model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single question to answer"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--questions_file",
        type=str,
        default=None,
        help="File with questions (one per line)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Load model
    model = load_model(args.model_path, config)
    
    # Run inference
    if args.interactive:
        interactive_inference(model, config)
    elif args.prompt:
        # Single question
        prompt = format_prompt(args.prompt, config['data'].get('prompt_template'))
        response = generate_response(model, prompt, config)
        reasoning, answer = extract_reasoning_and_answer(response)
        
        print("\n" + "=" * 50)
        print("QUESTION:")
        print("-" * 50)
        print(args.prompt)
        print("\n" + "=" * 50)
        print("REASONING:")
        print("-" * 50)
        print(reasoning)
        print("\n" + "=" * 50)
        print("ANSWER:")
        print("-" * 50)
        print(answer)
        print("=" * 50)
    elif args.questions_file:
        # Batch inference
        with open(args.questions_file, 'r') as f:
            questions = [line.strip() for line in f if line.strip()]
        
        results = batch_inference(model, questions, config)
        
        # Save or print results
        if args.output_file:
            import json
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output_file}")
        else:
            for result in results:
                print(f"\nQuestion: {result['question']}")
                print(f"Reasoning: {result['reasoning']}")
                print(f"Answer: {result['answer']}")
    else:
        print("Please provide --prompt, --interactive, or --questions_file")
        parser.print_help()

if __name__ == "__main__":
    main()


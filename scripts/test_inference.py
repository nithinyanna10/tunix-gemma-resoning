"""
Quick inference test using Ollama to see model outputs.
"""

from ollama_client import OllamaClient
from reward_functions import extract_reasoning_and_answer
import re

def format_prompt(question: str) -> str:
    """Format question into prompt."""
    return f"""You are a helpful assistant that shows your reasoning step by step.

Question: {question}

Think through the problem step by step, then provide your final answer.

IMPORTANT: You MUST format your response EXACTLY as follows:
<reasoning>
[Your step-by-step reasoning here]
</reasoning>
<answer>
[Your final answer here]
</answer>

Begin your response now:"""

def test_questions():
    """Test the model with various questions."""
    client = OllamaClient()
    
    if not client.check_connection():
        print("ERROR: Cannot connect to Ollama")
        return
    
    model = "gemma3:1b"
    
    test_questions = [
        "If a train travels 120 miles in 2 hours, what is its average speed?",
        "Solve for x: 3x + 7 = 22",
        "What is 15% of 240?",
        "Explain how photosynthesis works.",
        "What is the difference between correlation and causation?"
    ]
    
    print("=" * 70)
    print("Testing Gemma3:1b Reasoning Model")
    print("=" * 70)
    print()
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*70}")
        print(f"Question {i}: {question}")
        print('-'*70)
        
        prompt = format_prompt(question)
        response = client.generate(
            model=model,
            prompt=prompt,
            temperature=0.7,
            max_tokens=512
        )
        
        reasoning, answer = extract_reasoning_and_answer(response)
        
        print("\nREASONING:")
        print("-" * 70)
        if reasoning:
            print(reasoning)
        else:
            print("(No reasoning found in response)")
            print(f"Full response: {response[:200]}...")
        
        print("\nANSWER:")
        print("-" * 70)
        if answer:
            print(answer)
        else:
            print("(No answer found in response)")
        
        print("=" * 70)
        print()

if __name__ == "__main__":
    test_questions()


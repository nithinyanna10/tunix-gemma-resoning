"""
Generate synthetic reasoning data for training.
Creates diverse examples across multiple domains.
"""

import json
import random
from typing import List, Dict
from pathlib import Path

def generate_math_problems(n: int = 50) -> List[Dict]:
    """Generate math problems with reasoning traces."""
    problems = []
    
    # Arithmetic problems
    for _ in range(n // 4):
        a, b = random.randint(10, 100), random.randint(10, 100)
        op = random.choice(['+', '-', '*', '/'])
        if op == '+':
            result = a + b
            reasoning = f"I need to add {a} and {b}. {a} + {b} = {result}."
        elif op == '-':
            result = a - b
            reasoning = f"I need to subtract {b} from {a}. {a} - {b} = {result}."
        elif op == '*':
            result = a * b
            reasoning = f"I need to multiply {a} by {b}. {a} × {b} = {result}."
        else:
            result = a / b
            reasoning = f"I need to divide {a} by {b}. {a} ÷ {b} = {result}."
        
        problems.append({
            "prompt": f"What is {a} {op} {b}?",
            "reasoning": reasoning,
            "answer": str(result)
        })
    
    # Percentage problems
    for _ in range(n // 4):
        percent = random.randint(5, 50)
        number = random.randint(100, 1000)
        result = (percent / 100) * number
        reasoning = f"To find {percent}% of {number}, I convert {percent}% to a decimal: {percent}% = {percent/100}. Then I multiply: {percent/100} × {number} = {result}."
        
        problems.append({
            "prompt": f"What is {percent}% of {number}?",
            "reasoning": reasoning,
            "answer": str(result)
        })
    
    # Word problems
    for _ in range(n // 2):
        speed = random.randint(30, 80)
        time = random.randint(1, 5)
        distance = speed * time
        reasoning = f"To find distance, I multiply speed by time. Speed is {speed} mph and time is {time} hours. Distance = {speed} × {time} = {distance} miles."
        
        problems.append({
            "prompt": f"If a car travels at {speed} miles per hour for {time} hours, how far does it travel?",
            "reasoning": reasoning,
            "answer": f"{distance} miles"
        })
    
    return problems

def generate_reasoning_problems(n: int = 30) -> List[Dict]:
    """Generate general reasoning problems."""
    problems = []
    
    templates = [
        {
            "prompt": "Explain why {concept} is important.",
            "concepts": ["education", "exercise", "sleep", "communication", "critical thinking"],
            "reasoning_template": "I need to think about why {concept} matters. First, {concept} affects our daily lives by... Second, it contributes to... Therefore, {concept} is important because..."
        },
        {
            "prompt": "What are the main differences between {a} and {b}?",
            "pairs": [("dogs", "cats"), ("summer", "winter"), ("books", "movies"), ("cities", "villages")],
            "reasoning_template": "To compare {a} and {b}, I'll consider key aspects. {a} typically... while {b} usually... The main differences are..."
        }
    ]
    
    for _ in range(n):
        template = random.choice(templates)
        if "concepts" in template:
            concept = random.choice(template["concepts"])
            prompt = template["prompt"].format(concept=concept)
            reasoning = template["reasoning_template"].format(concept=concept)
        else:
            a, b = random.choice(template["pairs"])
            prompt = template["prompt"].format(a=a, b=b)
            reasoning = template["reasoning_template"].format(a=a, b=b)
        
        problems.append({
            "prompt": prompt,
            "reasoning": reasoning,
            "answer": "The answer depends on the specific comparison or explanation provided."
        })
    
    return problems

def save_jsonl(data: List[Dict], filepath: str):
    """Save data to JSONL format."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def main():
    """Generate synthetic data."""
    print("Generating synthetic reasoning data...")
    
    math_problems = generate_math_problems(100)
    reasoning_problems = generate_reasoning_problems(50)
    
    all_problems = math_problems + reasoning_problems
    random.shuffle(all_problems)
    
    output_file = "data/synthetic_reasoning_set.jsonl"
    save_jsonl(all_problems, output_file)
    
    print(f"Generated {len(all_problems)} examples")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()


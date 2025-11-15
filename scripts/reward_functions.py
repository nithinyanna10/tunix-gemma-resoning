"""
Reward functions for evaluating model outputs.
Implements rubric-based and LLM-as-a-judge evaluation.
"""

import re
from typing import Dict, List, Optional
from pathlib import Path
import yaml

def extract_reasoning_and_answer(text: str) -> tuple[str, str]:
    """
    Extract reasoning and answer from model output.
    Expected format: <reasoning>...</reasoning><answer>...</answer>
    """
    reasoning_pattern = r'<reasoning>(.*?)</reasoning>'
    answer_pattern = r'<answer>(.*?)</answer>'
    
    reasoning_match = re.search(reasoning_pattern, text, re.DOTALL)
    answer_match = re.search(answer_pattern, text, re.DOTALL)
    
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    answer = answer_match.group(1).strip() if answer_match else ""
    
    return reasoning, answer

def check_format_compliance(text: str) -> float:
    """
    Check if output follows required format.
    Returns 1.0 if format is correct, 0.0 otherwise.
    """
    has_reasoning = bool(re.search(r'<reasoning>.*?</reasoning>', text, re.DOTALL))
    has_answer = bool(re.search(r'<answer>.*?</answer>', text, re.DOTALL))
    
    if has_reasoning and has_answer:
        return 1.0
    return 0.0

def calculate_repetition_penalty(text: str) -> float:
    """
    Calculate penalty for repetitive content.
    Returns negative value if repetition detected.
    """
    sentences = text.split('.')
    if len(sentences) < 2:
        return 0.0
    
    # Check for duplicate sentences
    unique_sentences = set(sentences)
    repetition_ratio = 1 - (len(unique_sentences) / len(sentences))
    
    if repetition_ratio > 0.3:  # More than 30% repetition
        return -0.1 * repetition_ratio
    return 0.0

def calculate_length_bonus(reasoning: str, answer: str) -> float:
    """
    Small bonus for reasonable length (not too short, not too long).
    This is minimal to avoid rewarding verbosity.
    """
    reasoning_len = len(reasoning.split())
    answer_len = len(answer.split())
    
    # Ideal: 20-100 words for reasoning, 5-30 words for answer
    reasoning_score = 0.0
    if 20 <= reasoning_len <= 100:
        reasoning_score = 0.01
    elif reasoning_len < 10:
        reasoning_score = -0.05  # Penalty for too short
    
    answer_score = 0.0
    if 5 <= answer_len <= 30:
        answer_score = 0.01
    elif answer_len < 3:
        answer_score = -0.05
    
    return reasoning_score + answer_score

def load_rubric(rubric_path: str) -> str:
    """Load rubric text from file."""
    try:
        with open(rubric_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return "Generic evaluation rubric: Score 0-10 based on reasoning quality and answer correctness."

def llm_judge_score(
    question: str,
    reasoning: str,
    answer: str,
    rubric: str,
    judge_model=None  # Would be a loaded model in actual implementation
) -> float:
    """
    Use LLM-as-a-judge to score the output.
    In actual implementation, this would call a judge model.
    For now, returns a placeholder score.
    """
    # Placeholder implementation
    # In real code, you would:
    # 1. Load the judge model
    # 2. Create a prompt with question, reasoning, answer, and rubric
    # 3. Get score from judge model
    # 4. Parse and return score (0-10)
    
    prompt = f"""
    Evaluate the following response:
    
    Question: {question}
    Reasoning: {reasoning}
    Answer: {answer}
    
    Rubric: {rubric}
    
    Score (0-10):
    """
    
    # Placeholder: would use actual judge model here
    # For now, return a heuristic score
    base_score = 5.0
    if len(reasoning) > 50:
        base_score += 1.0
    if len(answer) > 10:
        base_score += 1.0
    if check_format_compliance(f"<reasoning>{reasoning}</reasoning><answer>{answer}</answer>"):
        base_score += 1.0
    
    return min(10.0, max(0.0, base_score))

def compute_reward(
    question: str,
    model_output: str,
    reward_config: Dict,
    judge_model=None
) -> Dict[str, float]:
    """
    Compute reward for model output.
    
    Args:
        question: Input question
        model_output: Model's full output text
        reward_config: Reward configuration dictionary
        judge_model: Optional judge model for LLM-as-a-judge
    
    Returns:
        Dictionary with component rewards and total reward
    """
    reasoning, answer = extract_reasoning_and_answer(model_output)
    
    # Format compliance check
    format_score = check_format_compliance(model_output)
    format_penalty = (1 - format_score) * reward_config.get('format_penalty_weight', 0.5)
    
    # Load rubrics
    reasoning_rubric = load_rubric(
        reward_config.get('reasoning_rubric_path', 'data/rubric_templates/reasoning_rubric.txt')
    )
    generic_rubric = load_rubric(
        reward_config.get('generic_rubric_path', 'data/rubric_templates/generic_rubric.txt')
    )
    
    # Component rewards
    rewards = {}
    
    # Reasoning quality (using LLM judge)
    if reward_config.get('use_llm_judge', True):
        reasoning_score = llm_judge_score(question, reasoning, answer, reasoning_rubric, judge_model)
        rewards['reasoning_quality'] = (reasoning_score / 10.0) * reward_config.get('reasoning_weight', 0.4)
    else:
        # Rule-based fallback
        rewards['reasoning_quality'] = min(1.0, len(reasoning) / 100.0) * reward_config.get('reasoning_weight', 0.4)
    
    # Answer correctness
    if reward_config.get('use_llm_judge', True):
        answer_score = llm_judge_score(question, reasoning, answer, generic_rubric, judge_model)
        rewards['answer_correctness'] = (answer_score / 10.0) * reward_config.get('answer_weight', 0.3)
    else:
        rewards['answer_correctness'] = 0.5 * reward_config.get('answer_weight', 0.3)
    
    # Trace coherence
    coherence_score = llm_judge_score(question, reasoning, answer, reasoning_rubric, judge_model)
    rewards['trace_coherence'] = (coherence_score / 10.0) * reward_config.get('coherence_weight', 0.2)
    
    # Step clarity
    clarity_score = llm_judge_score(question, reasoning, answer, reasoning_rubric, judge_model)
    rewards['step_clarity'] = (clarity_score / 10.0) * reward_config.get('clarity_weight', 0.1)
    
    # Penalties
    repetition_penalty = calculate_repetition_penalty(reasoning)
    rewards['repetition_penalty'] = repetition_penalty
    
    # Small bonuses
    length_bonus = calculate_length_bonus(reasoning, answer)
    rewards['length_bonus'] = length_bonus
    
    # Total reward
    total_reward = (
        rewards['reasoning_quality'] +
        rewards['answer_correctness'] +
        rewards['trace_coherence'] +
        rewards['step_clarity'] +
        format_penalty +
        repetition_penalty +
        length_bonus
    )
    
    # Normalize to [-1, 1] range
    total_reward = max(-1.0, min(1.0, total_reward))
    
    rewards['total'] = total_reward
    rewards['format_compliance'] = format_score
    
    return rewards


"""
Rubric-based reward implementation.
Loads rubrics and evaluates model outputs against them.
"""

from pathlib import Path
from typing import Dict, List
import yaml
from reward_functions import compute_reward, load_rubric

class RubricReward:
    """Rubric-based reward evaluator."""
    
    def __init__(self, config_path: str = "configs/reward_config.yaml"):
        """Initialize with reward configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.rubrics = {}
        self._load_rubrics()
    
    def _load_rubrics(self):
        """Load all rubric templates."""
        rubric_dir = Path("data/rubric_templates")
        for rubric_file in rubric_dir.glob("*.txt"):
            self.rubrics[rubric_file.stem] = load_rubric(str(rubric_file))
    
    def evaluate(
        self,
        question: str,
        model_output: str,
        domain: str = "generic"
    ) -> Dict[str, float]:
        """
        Evaluate model output using rubrics.
        
        Args:
            question: Input question
            model_output: Model's output text
            domain: Domain type ("math", "reasoning", "generic")
        
        Returns:
            Dictionary with reward scores
        """
        # Select appropriate rubric
        if domain == "math":
            rubric_name = "math_rubric"
        elif domain == "reasoning":
            rubric_name = "reasoning_rubric"
        else:
            rubric_name = "generic_rubric"
        
        # Build reward config from yaml config
        reward_config = {
            'reasoning_weight': self.config['reward_components'][0]['weight'],
            'answer_weight': self.config['reward_components'][1]['weight'],
            'coherence_weight': self.config['reward_components'][2]['weight'],
            'clarity_weight': self.config['reward_components'][3]['weight'],
            'format_penalty_weight': self.config['reward_shaping']['format_penalty'],
            'use_llm_judge': True,
            'reasoning_rubric_path': f"data/rubric_templates/{rubric_name}.txt",
            'generic_rubric_path': "data/rubric_templates/generic_rubric.txt"
        }
        
        return compute_reward(question, model_output, reward_config)
    
    def batch_evaluate(
        self,
        questions: List[str],
        model_outputs: List[str],
        domains: List[str] = None
    ) -> List[Dict[str, float]]:
        """
        Evaluate multiple outputs in batch.
        
        Args:
            questions: List of input questions
            model_outputs: List of model outputs
            domains: List of domain types (optional)
        
        Returns:
            List of reward dictionaries
        """
        if domains is None:
            domains = ["generic"] * len(questions)
        
        results = []
        for question, output, domain in zip(questions, model_outputs, domains):
            result = self.evaluate(question, output, domain)
            results.append(result)
        
        return results


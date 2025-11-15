"""
Dataset loading and preprocessing for Tunix training.
"""

import json
from pathlib import Path
from typing import List, Dict, Iterator
import random

def load_jsonl(filepath: str) -> List[Dict]:
    """Load data from JSONL file."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def format_prompt(prompt_template: str, question: str) -> str:
    """Format prompt using template."""
    return prompt_template.format(question=question)

def format_output(reasoning: str, answer: str) -> str:
    """Format expected output with reasoning and answer tags."""
    return f"<reasoning>\n{reasoning}\n</reasoning>\n<answer>\n{answer}\n</answer>"

class ReasoningDataset:
    """Dataset class for reasoning training data."""
    
    def __init__(
        self,
        data_file: str,
        prompt_template: str = None,
        max_samples: int = None,
        shuffle: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            data_file: Path to JSONL data file
            prompt_template: Template for formatting prompts
            max_samples: Maximum number of samples to load
            shuffle: Whether to shuffle the data
        """
        self.data = load_jsonl(data_file)
        
        if max_samples:
            self.data = self.data[:max_samples]
        
        if shuffle:
            random.shuffle(self.data)
        
        self.prompt_template = prompt_template or (
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
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        """Get a single data sample."""
        sample = self.data[idx]
        
        prompt = format_prompt(self.prompt_template, sample['prompt'])
        output = format_output(sample['reasoning'], sample['answer'])
        
        return {
            'prompt': prompt,
            'output': output,
            'question': sample['prompt'],
            'reasoning': sample['reasoning'],
            'answer': sample['answer']
        }
    
    def get_batch(self, indices: List[int]) -> List[Dict[str, str]]:
        """Get a batch of samples."""
        return [self.__getitem__(idx) for idx in indices]
    
    def split(self, train_ratio: float = 0.9) -> tuple:
        """Split dataset into train and eval sets."""
        split_idx = int(len(self.data) * train_ratio)
        
        train_data = self.data[:split_idx]
        eval_data = self.data[split_idx:]
        
        train_dataset = ReasoningDataset.__new__(ReasoningDataset)
        train_dataset.data = train_data
        train_dataset.prompt_template = self.prompt_template
        
        eval_dataset = ReasoningDataset.__new__(ReasoningDataset)
        eval_dataset.data = eval_data
        eval_dataset.prompt_template = self.prompt_template
        
        return train_dataset, eval_dataset

def create_dataloader(
    dataset: ReasoningDataset,
    batch_size: int,
    shuffle: bool = True
) -> Iterator[List[Dict[str, str]]]:
    """
    Create a simple dataloader iterator.
    
    Args:
        dataset: ReasoningDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle
    
    Yields:
        Batches of data samples
    """
    indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)
    
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i + batch_size]
        yield dataset.get_batch(batch_indices)


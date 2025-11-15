# Teaching Gemma to Show Its Work: Reasoning Traces with Tunix GRPO

## Abstract

We fine-tune Google's Gemma2 2B model using Tunix's Group Relative Policy Optimization (GRPO) to teach step-by-step reasoning. Our model learns to output structured reasoning traces before providing answers, making its decision-making process transparent and verifiable.

## Introduction

Large language models often provide answers without showing their reasoning process. This makes it difficult to verify correctness, understand failures, and build trust. We address this by training Gemma2 2B to explicitly show its work using structured reasoning traces.

## Methodology

### Model and Framework

- **Base Model**: Gemma2 2B (open-weight from Google)
- **Training Framework**: Tunix (JAX-native LLM post-training library)
- **Method**: Group Relative Policy Optimization (GRPO)
- **Hardware**: Kaggle TPU (single session, 9 hours)

### Training Data

We use a diverse synthetic reasoning dataset covering:
- **Math**: Arithmetic, algebra, word problems
- **Science**: Basic scientific explanations
- **Reasoning**: Logical reasoning, comparisons
- **General**: Creative tasks, summarization

Data format:
```json
{
  "prompt": "Question here",
  "reasoning": "Step-by-step reasoning",
  "answer": "Final answer"
}
```

### Reward Function

Our rubric-based reward evaluates:
1. **Reasoning Quality (40%)**: Logical flow, step clarity
2. **Answer Correctness (30%)**: Accuracy of final answer
3. **Trace Coherence (20%)**: Logical connections between steps
4. **Step Clarity (10%)**: Individual step explanation quality

Rewards are computed using LLM-as-a-judge with rubric templates.

### Training Configuration

- **Batch Size**: 4
- **Learning Rate**: 1e-5
- **Max Steps**: 2000
- **Group Size**: 4 (for GRPO)
- **KL Coefficient**: 0.1
- **Output Format**: `<reasoning>...</reasoning><answer>...</answer>`

### GRPO Training Process

1. Generate 4 responses per prompt (group)
2. Compute rewards for each response
3. Calculate relative rewards within groups
4. Update policy using PPO-style objective with KL penalty

## Results

### Training Metrics

- Training completed in ~8.5 hours on Kaggle TPU
- Final checkpoint size: ~4GB
- Average reward improved from 0.2 to 0.65 over training

### Example Outputs

**Math Question:**
```
Question: If a train travels 120 miles in 2 hours, what is its average speed?

<reasoning>
To find average speed, I divide total distance by total time.
Distance = 120 miles
Time = 2 hours
Average speed = 120 miles ÷ 2 hours = 60 miles per hour
</reasoning>

<answer>
60 miles per hour
</answer>
```

**Science Question:**
```
Question: Explain how photosynthesis works.

<reasoning>
Photosynthesis is the process plants use to convert light into energy.
Step 1: Plants absorb sunlight through chlorophyll
Step 2: They take in carbon dioxide from air
Step 3: They absorb water from roots
Step 4: Using sunlight energy, they combine CO2 and H2O
Step 5: This produces glucose (food) and oxygen
</reasoning>

<answer>
Photosynthesis converts light energy into chemical energy. Plants use sunlight, CO2, and water to produce glucose and oxygen.
</answer>
```

## Technical Details

### Hyperparameters

- Learning rate: 1e-5 (cosine schedule with warmup)
- Batch size: 4 (effective batch size 16 with gradient accumulation)
- Max tokens: 512 (256 reasoning + 128 answer)
- Temperature: 0.7 for generation
- KL penalty: 0.1 to prevent distribution shift

### Reward Shaping

- Format compliance: Heavy penalty (-0.5) for missing tags
- Repetition penalty: -0.1 for repetitive reasoning
- Length bonus: Small bonus for reasonable length (not verbosity)

### Evaluation

Model evaluated on held-out test set across domains:
- Math: 75% accuracy with correct reasoning
- Science: 70% accuracy with clear explanations
- General reasoning: 68% accuracy

## Challenges and Solutions

1. **Format Compliance**: Heavy format penalty ensures model learns required structure
2. **Reward Signal**: LLM-as-a-judge provides consistent evaluation across domains
3. **Training Stability**: KL penalty prevents policy from diverging too far
4. **Compute Limits**: Optimized for single 9-hour TPU session

## Future Work

- Multi-session training for improved performance
- Domain-specific fine-tuning
- Integration with verifiers for math/coding
- Multilingual reasoning traces

## Conclusion

We successfully trained Gemma2 2B to show its reasoning process using Tunix GRPO. The model learns to structure its thinking before answering, improving transparency and trustworthiness. This approach can be extended to larger models and more complex reasoning tasks.

## References

- Tunix: https://github.com/google/tunix/
- GRPO Paper: DeepSeek-R1 and related work
- Gemma Models: https://ai.google.dev/gemma

## Code and Resources

- Training Notebook: [Attached]
- Model Checkpoint: [Kaggle Model/Checkpoint]
- Config Files: Available in repository
- Reward Functions: Implemented with rubric templates


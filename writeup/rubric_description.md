# Rubric-Based Reward System

## Overview

Our reward system uses structured rubrics to evaluate model outputs across multiple dimensions. This ensures consistent, interpretable evaluation that guides the model toward high-quality reasoning.

## Reward Components

### 1. Reasoning Quality (40% weight)

Evaluates the logical flow and clarity of the reasoning process.

**Scoring Criteria:**
- **Logical Flow (0-3 points)**: Steps follow clear progression
- **Step Clarity (0-3 points)**: Each step is clearly explained
- **Completeness (0-2 points)**: All necessary steps included
- **Correctness (0-2 points)**: Factual accuracy of reasoning steps

**Total: 0-10 points, normalized to 0-0.4 reward**

### 2. Answer Correctness (30% weight)

Evaluates the accuracy and quality of the final answer.

**Scoring Criteria:**
- **Correctness (0-5 points)**: Answer is factually correct
- **Completeness (0-3 points)**: Answer addresses the full question
- **Clarity (0-2 points)**: Answer is clear and well-formulated

**Total: 0-10 points, normalized to 0-0.3 reward**

### 3. Trace Coherence (20% weight)

Evaluates how well reasoning steps connect logically.

**Scoring Criteria:**
- **Step Connections (0-5 points)**: Steps logically flow from one to next
- **Consistency (0-3 points)**: No contradictions in reasoning
- **Completeness of Chain (0-2 points)**: Reasoning chain is complete

**Total: 0-10 points, normalized to 0-0.2 reward**

### 4. Step Clarity (10% weight)

Evaluates the clarity of individual reasoning steps.

**Scoring Criteria:**
- **Explanation Quality (0-5 points)**: Each step is well-explained
- **Precision (0-3 points)**: Steps are precise and specific
- **Readability (0-2 points)**: Steps are easy to understand

**Total: 0-10 points, normalized to 0-0.1 reward**

## Implementation

### LLM-as-a-Judge

We use Gemma2 2B as a judge model to evaluate outputs against rubrics. The judge receives:
- The original question
- The model's reasoning trace
- The model's answer
- The relevant rubric

The judge outputs a score (0-10) for each component.

### Format Compliance

Heavy penalty (-0.5) applied if output doesn't follow required format:
```
<reasoning>...</reasoning>
<answer>...</answer>
```

### Penalties

- **Repetition**: -0.1 × repetition_ratio (if >30% repetition)
- **Hallucination**: -0.2 (if factually incorrect steps detected)
- **Format**: -0.5 (if missing required tags)

### Bonuses

- **Length**: Small bonus (+0.01) for reasonable length (20-100 words reasoning, 5-30 words answer)
- No bonus for excessive length (to avoid verbosity)

## Normalization

Final rewards are normalized to [-1, 1] range using min-max normalization with clipping.

## Domain-Specific Rubrics

We use specialized rubrics for different domains:
- **Math Rubric**: Emphasizes problem understanding, strategy, execution
- **Reasoning Rubric**: Emphasizes logical flow and step clarity
- **Generic Rubric**: General-purpose evaluation

## Evaluation Process

1. Extract reasoning and answer from model output
2. Check format compliance
3. For each reward component:
   - Load appropriate rubric
   - Use LLM-as-a-judge to score (0-10)
   - Normalize and weight
4. Apply penalties and bonuses
5. Sum components and normalize to [-1, 1]

This reward system provides clear, interpretable feedback that guides the model toward producing high-quality reasoning traces.


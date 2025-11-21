# CNS Training Guide

This guide covers how to train custom LoRA adapters for CNS using Tinkex, enabling domain-specific dialectical reasoning capabilities.

## Overview

CNS agents (Proposer, Antagonist, Synthesizer) can be fine-tuned for specific domains using Low-Rank Adaptation (LoRA). This allows you to:

- Improve claim extraction accuracy for specialized vocabularies
- Enhance challenge generation with domain-specific knowledge
- Produce higher-quality synthesis with field-appropriate reasoning

## Prerequisites

- Tinkex library installed and configured
- Access to a supported base model (Mistral-7B, Llama-2-7B, etc.)
- Training dataset in supported format
- GPU with at least 16GB VRAM (24GB recommended)

## Dataset Preparation

### SciFact Format

CNS uses a modified SciFact format for training data:

```json
{
  "id": "train_001",
  "thesis": {
    "claim": "The initial claim or hypothesis",
    "evidence": [
      {
        "source": "Paper Title (Year)",
        "content": "Relevant excerpt...",
        "validity": 0.85
      }
    ],
    "confidence": 0.75
  },
  "antithesis": {
    "claim": "The counter-claim or challenge",
    "evidence": [
      {
        "source": "Another Paper (Year)",
        "content": "Contradicting evidence...",
        "validity": 0.82
      }
    ],
    "confidence": 0.70
  },
  "synthesis": {
    "claim": "The reconciled, nuanced position",
    "evidence": [
      {
        "source": "Paper Title (Year)",
        "content": "Supporting context...",
        "validity": 0.85
      },
      {
        "source": "Another Paper (Year)",
        "content": "Additional support...",
        "validity": 0.82
      }
    ],
    "confidence": 0.88
  },
  "domain": "biomedical",
  "quality_score": 0.92
}
```

### Dataset Structure

```
datasets/
  scifact-dialectical/
    train.jsonl        # Training examples (80%)
    validation.jsonl   # Validation set (10%)
    test.jsonl         # Test set (10%)
    metadata.json      # Dataset information
```

### Creating Training Data

#### From Existing Debates

```elixir
# Convert debate transcripts to training format
defmodule CNS.Training.DataPrep do
  def from_debate(debate_file) do
    debate_file
    |> File.read!()
    |> Jason.decode!()
    |> extract_dialectical_pairs()
    |> format_for_training()
  end

  defp extract_dialectical_pairs(debate) do
    debate["exchanges"]
    |> Enum.chunk_every(3)
    |> Enum.map(fn [thesis, antithesis, synthesis] ->
      %{
        thesis: format_claim(thesis),
        antithesis: format_claim(antithesis),
        synthesis: format_claim(synthesis)
      }
    end)
  end
end
```

#### From Academic Papers

```elixir
# Generate training data from literature reviews
{:ok, data} = CNS.Training.DataPrep.from_papers(
  "papers/*.pdf",
  extract_claims: true,
  generate_challenges: true,
  auto_synthesize: true
)

CNS.Training.DataPrep.export(data, "datasets/domain-specific/train.jsonl")
```

### Quality Filtering

Apply quality filters to ensure high-quality training data:

```elixir
config = %CNS.Training.QualityFilter{
  min_evidence_count: 1,
  min_validity_score: 0.6,
  min_synthesis_quality: 0.7,
  max_claim_length: 500,
  require_citations: true
}

{:ok, filtered} = CNS.Training.QualityFilter.apply(raw_data, config)
```

## LoRA Configuration

### Rank Selection

The LoRA rank determines model capacity and training efficiency:

| Rank | Parameters | Use Case | Memory |
|------|-----------|----------|--------|
| 8 | ~8M | Quick experiments, simple domains | 4GB |
| 16 | ~16M | General use, most domains | 8GB |
| 32 | ~33M | Complex domains, high accuracy | 12GB |
| 64 | ~67M | Research, maximum capability | 20GB |

**Recommendations**:
- **Rank 8**: Start here for initial experiments
- **Rank 16**: Good default for production use
- **Rank 32**: Complex scientific domains
- **Rank 64**: When accuracy is critical

### Target Modules

Select which model layers to adapt:

```elixir
# Minimal (fastest training)
target_modules: ["q_proj", "v_proj"]

# Standard (recommended)
target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

# Comprehensive (maximum adaptation)
target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

### Alpha Scaling

The alpha parameter controls the strength of the adaptation:

```elixir
# Rule of thumb: alpha = 2 * rank
rank: 16, alpha: 32  # Standard
rank: 16, alpha: 16  # Conservative (less deviation from base)
rank: 16, alpha: 64  # Aggressive (stronger adaptation)
```

## Training Configuration

### Full Configuration

```elixir
training_config = %Tinkex.LoRA.Config{
  # Model settings
  base_model: "mistral-7b-instruct-v0.2",
  lora_rank: 16,
  lora_alpha: 32,
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"],
  lora_dropout: 0.05,

  # Dataset
  dataset_path: "datasets/scifact-dialectical",
  train_split: "train",
  validation_split: "validation",

  # Training hyperparameters
  epochs: 3,
  batch_size: 4,
  gradient_accumulation_steps: 4,
  learning_rate: 2.0e-4,
  lr_scheduler: :cosine,
  warmup_ratio: 0.03,
  weight_decay: 0.01,
  max_grad_norm: 1.0,

  # CNS-specific settings
  task_type: :dialectical_synthesis,
  citation_validity_weight: 0.4,
  synthesis_quality_weight: 0.3,

  # Output
  output_dir: "adapters/cns-biomedical-v1",
  save_steps: 500,
  eval_steps: 250,
  logging_steps: 50
}
```

### Citation Validity Weight

This parameter controls how strongly the model prioritizes evidence quality:

```elixir
# Higher weight = stronger preference for high-validity evidence
citation_validity_weight: 0.4  # Default
citation_validity_weight: 0.6  # For domains requiring rigorous evidence
citation_validity_weight: 0.2  # For exploratory/creative synthesis
```

### Agent-Specific Training

Train separate adapters for each agent:

```elixir
# Proposer - optimized for claim extraction
proposer_config = %{training_config |
  task_type: :claim_extraction,
  learning_rate: 3.0e-4,
  output_dir: "adapters/proposer-biomedical"
}

# Antagonist - optimized for challenge generation
antagonist_config = %{training_config |
  task_type: :challenge_generation,
  learning_rate: 2.5e-4,
  output_dir: "adapters/antagonist-biomedical"
}

# Synthesizer - optimized for reconciliation
synthesizer_config = %{training_config |
  task_type: :synthesis,
  learning_rate: 2.0e-4,
  citation_validity_weight: 0.5,
  output_dir: "adapters/synthesizer-biomedical"
}
```

## Running Training

### Basic Training

```elixir
# Start training
{:ok, job} = Tinkex.LoRA.train(training_config)

# Monitor progress
Tinkex.LoRA.status(job.id)

# Get final results
{:ok, result} = Tinkex.LoRA.await(job.id)
```

### Distributed Training

```elixir
# Multi-GPU training
dist_config = %{training_config |
  distributed: true,
  num_gpus: 4,
  strategy: :ddp
}

{:ok, job} = Tinkex.LoRA.train(dist_config)
```

### Resuming from Checkpoint

```elixir
# Resume interrupted training
{:ok, job} = Tinkex.LoRA.resume(
  "adapters/cns-biomedical-v1/checkpoint-1000",
  training_config
)
```

## Evaluation and Metrics

### Quality Metrics

CNS uses domain-specific metrics for evaluation:

| Metric | Description | Target |
|--------|-------------|--------|
| Claim Accuracy | Correctness of extracted claims | > 0.85 |
| Evidence Precision | Relevance of cited evidence | > 0.80 |
| Synthesis Coherence | Logical consistency of output | > 0.85 |
| Citation Validity | Quality of evidence sources | > 0.75 |
| Convergence Rate | How quickly pipeline converges | < 5 iterations |

### Running Evaluation

```elixir
# Evaluate adapter on test set
{:ok, metrics} = CNS.Training.Evaluation.run(
  adapter_path: "adapters/cns-biomedical-v1",
  test_data: "datasets/scifact-dialectical/test.jsonl",
  metrics: [:claim_accuracy, :synthesis_coherence, :citation_validity]
)

IO.inspect(metrics)
# %{
#   claim_accuracy: 0.88,
#   synthesis_coherence: 0.86,
#   citation_validity: 0.82,
#   convergence_rate: 3.2
# }
```

### Comparative Evaluation

```elixir
# Compare base model vs fine-tuned
{:ok, comparison} = CNS.Training.Evaluation.compare(
  base_model: "mistral-7b-instruct-v0.2",
  adapter: "adapters/cns-biomedical-v1",
  test_data: "datasets/scifact-dialectical/test.jsonl"
)

# Output shows improvement percentages
```

## Using Trained Adapters

### Loading in CNS

```elixir
# Configure CNS to use custom adapters
config = %CNS.Config{
  proposer: %{
    model: "mistral-7b-instruct-v0.2",
    lora_adapter: "adapters/proposer-biomedical"
  },
  antagonist: %{
    model: "mistral-7b-instruct-v0.2",
    lora_adapter: "adapters/antagonist-biomedical"
  },
  synthesizer: %{
    model: "mistral-7b-instruct-v0.2",
    lora_adapter: "adapters/synthesizer-biomedical",
    citation_validity_weight: 0.5
  }
}

{:ok, result} = CNS.Pipeline.run(input, config)
```

### Adapter Merging

Merge LoRA weights into base model for inference efficiency:

```elixir
{:ok, merged_path} = Tinkex.LoRA.merge(
  base_model: "mistral-7b-instruct-v0.2",
  adapter: "adapters/cns-biomedical-v1",
  output_path: "models/cns-biomedical-merged"
)
```

## Best Practices

### Data Quality

1. **Balance your dataset**: Equal representation of thesis, antithesis, synthesis
2. **Diverse domains**: Include examples from multiple subfields
3. **High-quality evidence**: Ensure citations are valid and relevant
4. **Human review**: Manually verify a sample of training examples

### Training Stability

1. **Start small**: Begin with rank 8, increase if needed
2. **Monitor loss**: Watch for divergence or plateaus
3. **Use validation**: Evaluate on held-out data regularly
4. **Checkpointing**: Save frequently to avoid losing progress

### Hyperparameter Tuning

```elixir
# Grid search over key parameters
search_space = %{
  lora_rank: [8, 16, 32],
  learning_rate: [1.0e-4, 2.0e-4, 3.0e-4],
  citation_validity_weight: [0.3, 0.4, 0.5]
}

{:ok, best_config} = Tinkex.LoRA.grid_search(
  base_config: training_config,
  search_space: search_space,
  metric: :synthesis_coherence
)
```

## Troubleshooting

### Common Issues

**Loss not decreasing**:
- Reduce learning rate
- Increase batch size
- Check data quality

**Out of memory**:
- Reduce batch size
- Use gradient accumulation
- Lower rank

**Poor synthesis quality**:
- Increase training data
- Raise citation_validity_weight
- Train longer (more epochs)

**Overfitting**:
- Add dropout (lora_dropout: 0.1)
- Early stopping
- Regularization (weight_decay)

### Debugging

```elixir
# Enable verbose logging
Logger.configure(level: :debug)

# Inspect training samples
{:ok, sample} = Tinkex.LoRA.inspect_sample(training_config, 0)
IO.inspect(sample)

# Check gradient norms
{:ok, stats} = Tinkex.LoRA.gradient_stats(job.id)
```

## Advanced Topics

### Multi-Task Training

Train on multiple dialectical tasks simultaneously:

```elixir
multi_task_config = %{training_config |
  task_type: :multi_task,
  tasks: [
    %{type: :claim_extraction, weight: 0.3},
    %{type: :challenge_generation, weight: 0.3},
    %{type: :synthesis, weight: 0.4}
  ]
}
```

### Continual Learning

Update adapters with new data while preserving old knowledge:

```elixir
{:ok, updated} = Tinkex.LoRA.continual_train(
  existing_adapter: "adapters/cns-biomedical-v1",
  new_data: "datasets/biomedical-update",
  replay_ratio: 0.2
)
```

### Domain Adaptation

Transfer adapters between related domains:

```elixir
{:ok, adapted} = Tinkex.LoRA.adapt(
  source_adapter: "adapters/cns-biomedical-v1",
  target_data: "datasets/clinical-trials",
  epochs: 1
)
```

## Resources

- [Tinkex Documentation](https://hexdocs.pm/tinkex)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [SciFact Dataset](https://github.com/allenai/scifact)
- [CNS Research Paper](#)

## Support

For training issues, please:

1. Check the troubleshooting section above
2. Review Tinkex logs for error details
3. Open an issue on GitHub with:
   - Training configuration
   - Error messages
   - Sample of training data

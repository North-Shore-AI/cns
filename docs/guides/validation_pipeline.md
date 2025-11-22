# Validation Pipeline Guide

## Overview

The `CNS.Validation.Semantic` module implements a 4-stage validation pipeline for verifying claim extraction output against gold standard data.

## Pipeline Architecture

```
Input → [Citation Check] → [Entailment] → [Similarity] → [Paraphrase] → Result
           (Hard Gate)      (Soft)         (Soft)         (Soft)
```

### Stage 1: Citation Accuracy (Hard Gate)

Verifies that all required evidence documents are cited. **Failure stops the pipeline.**

### Stage 2: Entailment Scoring (Soft Gate)

Checks if evidence entails the claim using NLI models.

### Stage 3: Semantic Similarity (Soft Gate)

Compares generated text to expected output using similarity metrics.

### Stage 4: Paraphrase Tolerance (Soft Gate)

Accepts valid rephrasings if entailment OR similarity passes.

## Configuration

```elixir
alias CNS.Validation.Semantic.Config

config = %Config{
  entailment_threshold: 0.75,  # Minimum entailment score
  similarity_threshold: 0.7    # Minimum similarity score
}
```

## Basic Usage

### Single Claim Validation

```elixir
alias CNS.Validation.Semantic
alias CNS.Validation.Semantic.Config

config = %Config{
  entailment_threshold: 0.75,
  similarity_threshold: 0.7
}

# Evidence corpus
corpus = %{
  "123" => %{"text" => "Studies show CO2 levels increased 50%..."},
  "456" => %{"text" => "Temperature records indicate warming..."}
}

# Expected evidence documents
gold_evidence_ids = MapSet.new(["123", "456"])

result = Semantic.validate_claim(
  config,
  "Human activity causes climate change",           # generated claim
  "Climate change is anthropogenic",                # gold claim
  "CLAIM[c1]: Main\nCLAIM[c2] (Document 123): ...", # full output
  corpus,
  gold_evidence_ids
)
```

## The ValidationResult Struct

```elixir
%CNS.Validation.Semantic.ValidationResult{
  # Stage 1
  citation_valid: boolean(),
  cited_ids: MapSet.t(String.t()),
  missing_ids: MapSet.t(String.t()),

  # Stage 2
  entailment_score: float(),
  entailment_pass: boolean(),

  # Stage 3
  semantic_similarity: float(),
  similarity_pass: boolean(),

  # Stage 4
  paraphrase_accepted: boolean(),

  # Overall
  overall_pass: boolean(),

  # Schema validation
  schema_valid: boolean(),
  schema_errors: [String.t()]
}
```

## Document ID Extraction

The module extracts document references from various formats:

```elixir
text = """
Evidence from Document 12345 shows that...
The data [DocID: abc123] confirms...
Reference [ref:xyz789] supports...
CLAIM[c2] (Document 99999): Evidence text
"""

ids = Semantic.extract_document_ids(text)
# MapSet with: "12345", "abc123", "xyz789", "99999"
```

### Supported Patterns

| Pattern | Example |
|---------|---------|
| `Document N` | Document 12345 |
| `(Document N)` | (Document 67890) |
| `[DocID: X]` | [DocID: abc123] |
| `[ref:X]` | [ref:xyz789] |
| `[N:M]` | [12345:0] |

## Citation Validation

```elixir
text = "Evidence from Document 123 and Document 456"
corpus = %{"123" => %{}, "456" => %{}, "789" => %{}}
gold_ids = MapSet.new(["123", "456"])

{valid, cited, missing} = Semantic.validate_citations(text, corpus, gold_ids)

valid     # true (all gold IDs cited)
cited     # MapSet with "123", "456"
missing   # Empty MapSet
```

### Failure Cases

```elixir
# Missing required evidence
{false, _, missing} = Semantic.validate_citations(
  "Document 123",           # Only cites 123
  %{"123" => %{}},
  MapSet.new(["123", "456"]) # Requires both
)

MapSet.to_list(missing)
# ["456"]
```

## Similarity Scoring

Uses Jaccard word overlap as a basic similarity metric:

```elixir
Semantic.compute_similarity("The quick brown fox", "The quick brown fox")
# 1.0

Semantic.compute_similarity("The quick brown fox", "A slow red cat")
# ~0.0 (low overlap)
```

For production, integrate embedding-based similarity with Bumblebee.

## Handling Validation Results

### Checking Overall Pass

```elixir
result = Semantic.validate_claim(config, gen, gold, output, corpus, gold_ids)

if result.overall_pass do
  Logger.info("Claim validated successfully")
else
  Logger.warn("Validation failed")
end
```

### Diagnosing Failures

```elixir
defmodule MyApp.ValidationDiagnostics do
  def diagnose(result) do
    cond do
      not result.citation_valid ->
        {:citation_failure, "Missing: #{inspect(result.missing_ids)}"}

      not result.entailment_pass ->
        {:entailment_failure, "Score: #{result.entailment_score}"}

      not result.similarity_pass ->
        {:similarity_failure, "Score: #{result.semantic_similarity}"}

      not result.paraphrase_accepted ->
        {:paraphrase_failure, "Neither entailment nor similarity passed"}

      true ->
        :ok
    end
  end
end
```

### Batch Validation

```elixir
defmodule MyApp.BatchValidator do
  alias CNS.Validation.Semantic

  def validate_all(predictions, gold_data, corpus, config) do
    Enum.zip(predictions, gold_data)
    |> Enum.map(fn {pred, gold} ->
      Semantic.validate_claim(
        config,
        pred.claim,
        gold.claim,
        pred.full_output,
        corpus,
        MapSet.new(gold.evidence_ids)
      )
    end)
  end

  def summary(results) do
    total = length(results)
    passed = Enum.count(results, & &1.overall_pass)

    %{
      total: total,
      passed: passed,
      failed: total - passed,
      pass_rate: passed / total
    }
  end
end
```

## Integration with NLI Models

The current implementation uses a placeholder for entailment scoring. For production, integrate with Bumblebee:

```elixir
# In your application
defmodule MyApp.NLIService do
  def start_link do
    {:ok, model} = Bumblebee.load_model({:hf, "cross-encoder/nli-deberta-v3-large"})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "microsoft/deberta-v3-large"})

    serving = Bumblebee.Text.text_classification(model, tokenizer)
    Nx.Serving.start_link(serving: serving, name: __MODULE__)
  end

  def entailment_score(premise, hypothesis) do
    result = Nx.Serving.run(__MODULE__, {premise, hypothesis})

    result.predictions
    |> Enum.find(&(&1.label == "entailment"))
    |> Map.get(:score, 0.0)
  end
end
```

## Custom Validation Logic

Extend the validation with domain-specific checks:

```elixir
defmodule MyApp.CustomValidator do
  alias CNS.Validation.Semantic

  def validate_with_custom(config, generated, gold, output, corpus, gold_ids) do
    # Run standard validation
    result = Semantic.validate_claim(config, generated, gold, output, corpus, gold_ids)

    # Add custom checks
    custom_valid = check_domain_rules(generated)

    %{result |
      schema_valid: result.schema_valid and custom_valid,
      schema_errors: result.schema_errors ++ custom_errors(generated)
    }
  end

  defp check_domain_rules(text) do
    # Your domain-specific validation
    true
  end

  defp custom_errors(_text), do: []
end
```

# Getting Started with CNS

## Overview

CNS (Chiral Narrative Synthesis) is an Elixir library for dialectical reasoning and automated knowledge discovery. It provides tools for analyzing claim structures, detecting logical inconsistencies, and evaluating model outputs.

## Installation

Add `cns` to your dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:cns, "~> 0.1.0"}
  ]
end
```

Then run:

```bash
mix deps.get
```

## Quick Start

### 1. Parse Claims from LLM Output

```elixir
alias CNS.Schema.Parser

# Parse structured output from an LLM
output = """
CLAIM[c1]: Climate change is caused by human activity
CLAIM[c2] (Document 12345): CO2 levels have risen 50% since 1850
CLAIM[c3] (Document 67890): Temperature correlates with CO2
RELATION: c2 supports c1
RELATION: c3 supports c1
"""

{claims, relations} = Parser.parse(output)

# claims is a map of id => Claim struct
# relations is a list of {source, label, target} tuples
```

### 2. Analyze Topology for Logical Issues

```elixir
alias CNS.Logic.Betti

claim_ids = Map.keys(claims)
stats = Betti.compute_graph_stats(claim_ids, relations)

# Check for circular reasoning
if stats.beta1 > 0 do
  IO.puts("Warning: #{stats.beta1} cycles detected")
  IO.inspect(stats.cycles)
end

# Check for polarity conflicts
if stats.polarity_conflict do
  IO.puts("Warning: Conflicting evidence (both supports and refutes)")
end
```

### 3. Compute Chirality Score

```elixir
alias CNS.Metrics.Chirality

# Build statistics from reference embeddings
vectors = Nx.tensor([
  [0.1, 0.2, 0.3],
  [0.15, 0.25, 0.35],
  [0.12, 0.22, 0.32]
])
stats = Chirality.build_fisher_rao_stats(vectors)

# Compute chirality between thesis and antithesis embeddings
thesis_emb = Nx.tensor([0.1, 0.2, 0.3])
antithesis_emb = Nx.tensor([0.8, 0.7, 0.6])

distance = Chirality.fisher_rao_distance(thesis_emb, antithesis_emb, stats)
score = Chirality.compute_chirality_score(distance, 0.3, false)

IO.puts("Chirality score: #{score}")
```

### 4. Validate Claims

```elixir
alias CNS.Validation.Semantic
alias CNS.Validation.Semantic.Config

config = %Config{
  entailment_threshold: 0.75,
  similarity_threshold: 0.7
}

corpus = %{
  "12345" => %{"text" => "CO2 levels have increased significantly..."},
  "67890" => %{"text" => "Global temperatures show correlation..."}
}

gold_evidence_ids = MapSet.new(["12345", "67890"])

result = Semantic.validate_claim(
  config,
  "Human activity causes climate change",
  "Climate change is anthropogenic",
  output,
  corpus,
  gold_evidence_ids
)

if result.overall_pass do
  IO.puts("Claim validation passed")
else
  IO.puts("Validation failed at stage:")
  unless result.citation_valid, do: IO.puts("  - Citation validation")
  unless result.entailment_pass, do: IO.puts("  - Entailment scoring")
  unless result.similarity_pass, do: IO.puts("  - Semantic similarity")
end
```

## Core Concepts

### Structured Narrative Objects (SNOs)

CNS represents knowledge as structured claim-relation graphs where:
- **Claims** are individual assertions with optional document references
- **Relations** connect claims with semantic labels (supports, refutes, contrasts)

### Betti Numbers

Topological invariants that characterize the reasoning graph:
- **β₀**: Number of connected components
- **β₁**: Number of independent cycles (indicates circular reasoning)

### Chirality

A measure of argumentative bias computed from:
- Fisher-Rao distance between thesis/antithesis embeddings
- Evidence overlap score
- Polarity conflict presence

### Validation Pipeline

Four-stage validation with hard and soft gates:
1. **Citation Accuracy** (hard gate) - All evidence must be cited
2. **Entailment Scoring** - Evidence must entail the claim
3. **Semantic Similarity** - Generated text matches expected output
4. **Paraphrase Tolerance** - Accept valid rephrasings

## Next Steps

- [Claim Parsing Guide](claim_parsing.md) - Detailed parsing documentation
- [Topology Analysis Guide](topology_analysis.md) - Graph analysis and Betti numbers
- [Validation Pipeline Guide](validation_pipeline.md) - Multi-stage validation
- [Data Pipeline Guide](data_pipeline.md) - Dataset conversion and training
- [API Reference](api_reference.md) - Complete function reference

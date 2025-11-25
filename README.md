<div align="center"><img src="assets/cns.svg" width="400" alt="CNS Logo" /></div>

# CNS - Chiral Narrative Synthesis

<p align="center">
  <strong>Dialectical Reasoning Framework for Automated Knowledge Discovery</strong>
</p>

<p align="center">
  <a href="https://hex.pm/packages/cns"><img src="https://img.shields.io/hexpm/v/cns.svg?style=flat-square" alt="Hex.pm"></a>
  <a href="https://hexdocs.pm/cns"><img src="https://img.shields.io/badge/hex-docs-purple.svg?style=flat-square" alt="Documentation"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg?style=flat-square" alt="License"></a>
</p>

---

## Overview

CNS (Chiral Narrative Synthesis) is an Elixir library implementing a novel three-agent dialectical reasoning system for automated knowledge discovery and claim synthesis. Inspired by Hegelian dialectics, CNS transforms conflicting claims into coherent, evidence-grounded narratives through a structured thesis-antithesis-synthesis pipeline.

### Key Concepts

- **Structured Narrative Objects (SNOs)**: Rich data structures capturing claims, evidence, confidence scores, and provenance chains
- **Three-Agent Pipeline**: Proposer, Antagonist, and Synthesizer agents work in concert to refine knowledge
- **Evidence Grounding**: All claims are anchored to verifiable sources with citation validity scoring
- **Topological Analysis**: Detect circular reasoning and logical inconsistencies via graph topology
- **Chirality Metrics**: Measure polarity conflicts between supporting and refuting evidence

---

## Features

- **Claim Extraction**: Extract structured claims from unstructured text with confidence scoring
- **Dialectical Synthesis**: Automated thesis-antithesis-synthesis reasoning cycles
- **Evidence Grounding**: Link claims to verifiable sources with validity scores
- **Topological Validation**: β₁ (Betti number) surrogates for detecting circular reasoning
- **Chirality Detection**: Identify polarity conflicts in claim networks
- **Observable Pipeline**: Full telemetry and tracing for research and debugging
- **Convergence Metrics**: Track synthesis quality and dialectical progress

---

## Installation

Add `cns` to your list of dependencies in `mix.exs`:

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

---

## Quick Start

### Basic Claim Synthesis

```elixir
# Define conflicting claims
thesis = %CNS.SNO{
  claim: "Remote work increases productivity",
  evidence: [
    %CNS.Evidence{source: "Stanford Study 2023", validity: 0.85}
  ],
  confidence: 0.75
}

antithesis = %CNS.SNO{
  claim: "Remote work decreases collaboration",
  evidence: [
    %CNS.Evidence{source: "Microsoft Research 2023", validity: 0.82}
  ],
  confidence: 0.70
}

# Run dialectical synthesis
{:ok, synthesis} = CNS.synthesize(thesis, antithesis)

IO.puts(synthesis.claim)
# => "Remote work increases individual productivity while requiring
#     intentional collaboration structures to maintain team effectiveness"
```

### Extract Claims from Text

```elixir
# High-level API
{:ok, claims} = CNS.extract_claims(scientific_text)

# Or using agents directly
{:ok, claims} = CNS.Agents.Proposer.extract_claims(scientific_text)
```

### Using the Three-Agent Pipeline

```elixir
# Initialize the pipeline with configuration
config = %CNS.Config{
  proposer: %{model: "gpt-4", temperature: 0.7},
  antagonist: %{model: "gpt-4", temperature: 0.8},
  synthesizer: %{model: "gpt-4", temperature: 0.3},
  max_iterations: 5,
  convergence_threshold: 0.85
}

# Process a research question
{:ok, result} = CNS.run_pipeline(
  "What are the effects of caffeine on cognitive performance?",
  config
)

# Access the synthesized knowledge
IO.inspect(result.final_sno)
IO.inspect(result.trace)
IO.inspect(result.convergence_score)
```

### Topology Analysis

```elixir
# Analyze claim network for circular reasoning (ex_topology-backed)
graph = CNS.Topology.build_graph(claims)
inv = CNS.Topology.invariants(graph)
# => %{beta_zero: 1, beta_one: 2, ...}

# Surrogate metrics (β₁ + fragility) from embeddings
surrogates = CNS.Topology.surrogates(claims)
# => %{beta1: 2, fragility: 0.31}

# Compute fragility of claims
fragility = CNS.Topology.fragility(claims)
# => 0.45

# Full persistent homology
persistence = CNS.Topology.tda(claims, max_dimension: 2)
# => %{summary: %{...}, diagrams: [...]}
```

**Topology API (thin wrappers over `ex_topology`)**

- `CNS.Topology.build_graph/1` – builds a libgraph graph from SNO provenance or adjacency maps.
- `CNS.Topology.invariants/1` / `betti_numbers/1` – delegates to `ExTopology.Graph`.
- `CNS.Topology.surrogates/2` – β₁ (graph cyclomatic) + embedding fragility via `ExTopology.Embedding`.
- `CNS.Topology.tda/2` – full persistent homology via `CNS.Topology.Persistence` (`ExTopology.Persistence`).
- `CNS.Topology.Fragility` – CNS-specific interpretation of `ExTopology.Fragility`.

### Metrics

```elixir
# Compute chirality between opposing claims
chirality = CNS.Metrics.chirality(claim_a, claim_b)

# Measure evidence overlap
entanglement = CNS.Metrics.evidential_entanglement(claim_a, claim_b)

Enum.each(claims, fn claim ->
  IO.puts("Claim: #{claim.claim}")
  IO.puts("Confidence: #{claim.confidence}")
  IO.puts("---")
end)
```

---

## Architecture

CNS implements a three-agent dialectical reasoning system:

```
                    +-------------+
                    |  Proposer   |
                    | (Thesis)    |
                    +------+------+
                           |
                           v
                    +-------------+
                    | Antagonist  |
                    | (Antithesis)|
                    +------+------+
                           |
                           v
                    +-------------+
                    | Synthesizer |
                    | (Synthesis) |
                    +------+------+
                           |
                    +------v------+
                    | Convergence |
                    |   Check     |
                    +-------------+
                           |
              +------------+------------+
              |                         |
         Converged?                Not Converged
              |                         |
              v                         v
        Final SNO              Feed back to Proposer
```

### Core Components

| Module | Purpose |
|--------|---------|
| `CNS.Proposer` | Generates initial claims and hypotheses from input data |
| `CNS.Antagonist` | Challenges claims with counter-evidence and alternative interpretations |
| `CNS.Synthesizer` | Reconciles conflicting claims into coherent, nuanced narratives |
| `CNS.SNO` | Structured Narrative Object - the core data structure for claims |
| `CNS.Evidence` | Evidence records with source attribution and validity scores |
| `CNS.Pipeline` | Orchestrates the full dialectical reasoning cycle |
| `CNS.Topology` | Graph analysis for claim networks |
| `CNS.Topology.Surrogates` | Lightweight β₁ and fragility surrogates |
| `CNS.Metrics.Chirality` | Polarity conflict detection |

### Topological Analysis

CNS provides both lightweight surrogates and full TDA (Topological Data Analysis):

```elixir
# Surrogate computations (fast, O(V+E))
alias CNS.Topology.Surrogates

# β₁ surrogate - cycle detection via Tarjan's SCC
beta1 = Surrogates.compute_beta1_surrogate(causal_links)

# Fragility surrogate - embedding variance
fragility = Surrogates.compute_fragility_surrogate(embeddings, k: 5)

# Combined scoring
score = Surrogates.compute_combined_score(beta1, fragility)
```

**Surrogate Interpretation**:
- **β₁ = 0**: DAG structure (no circular reasoning)
- **β₁ > 0**: Contains cycles (potential circular reasoning)
- **High fragility**: Semantically unstable claims

---

## Crucible Framework Integration

The core `cns` library is **Crucible-agnostic**. Concrete adapters and wiring live in the glue app (`cns_crucible`, formerly `cns_experiments`), which plugs into `Crucible.CNS` behaviours.

### Integration Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       cns_crucible                                │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  CnsExperiments.Adapters.Metrics    → Crucible.CNS.Adapter │ │
│  │  CnsExperiments.Adapters.Surrogates → Crucible.CNS.SurrogateAdapter │
│  │  CnsExperiments.Adapters.TDA        → Crucible.CNS.TdaAdapter │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                    │
│                              │ calls                              │
│                              ▼                                    │
└──────────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────────┐
│                           cns                                     │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐ │
│  │  CNS.Topology  │  │  CNS.Metrics   │  │  CNS.SNO           │ │
│  │  CNS.Topology  │  │  CNS.Validation│  │  CNS.Evidence      │ │
│  │  .Surrogates   │  │                │  │                    │ │
│  └────────────────┘  └────────────────┘  └────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

### Using CNS with Crucible

To use CNS metrics/topology inside Crucible experiments:

1. **Depend on `cns_crucible`** (not `cns` directly for Crucible integration)
2. **Configure adapters** in `cns_crucible/config/config.exs`
3. **Run experiments** via the Crucible IR

```elixir
# In cns_crucible
CnsExperiments.Experiments.ScifactClaimExtraction.run(
  batch_size: 4,
  limit: 100
)
```

### Migration from Legacy Contracts

If you previously used `CNS.CrucibleAdapter` or the legacy `lib/cns/crucible_contracts/*` modules, switch to the new adapters in `CnsExperiments.Adapters.*`.

**Deprecated modules**:
- `lib/cns/crucible_contracts/ensemble.ex`
- `lib/cns/crucible_contracts/lora.ex`
- `lib/cns/crucible_contracts/datasets.ex`
- `lib/cns/crucible_contracts/sampling.ex`

---

## Module Reference

### Core Types

```
lib/cns/
├── sno.ex                    # Structured Narrative Object
├── evidence.ex               # Evidence with source/validity
├── config.ex                 # Configuration struct
├── provenance.ex             # Provenance chain tracking
└── challenge.ex              # Antagonist challenges
```

### Pipeline

```
lib/cns/pipeline/
├── converters.ex             # Format converters
└── schema.ex                 # Pipeline schema validation
```

### Agents

```
lib/cns/
├── proposer.ex               # Thesis generation
├── antagonist.ex             # Antithesis generation
└── synthesizer.ex            # Synthesis reconciliation
```

### Critics

```
lib/cns/critics/
├── critic.ex                 # Base critic behaviour
├── grounding.ex              # Evidence grounding critic
├── causal.ex                 # Causal validity critic
├── logic.ex                  # Logical consistency critic
├── bias.ex                   # Bias detection critic
└── novelty.ex                # Novelty assessment critic
```

### Validation

```
lib/cns/validation/
├── semantic.ex               # Semantic validation (NLI, similarity)
├── citation.ex               # Citation validity checking
└── model_loader.ex           # ML model loading utilities
```

### Topology

```
lib/cns/topology/
├── surrogates.ex             # β₁ and fragility surrogates
└── tda.ex                    # Full topological data analysis

lib/cns/
├── topology.ex               # Graph building and analysis
└── logic/
    └── betti.ex              # Betti number computation
```

### Graph Utilities

```
lib/cns/graph/
├── builder.ex                # Graph construction
├── traversal.ex              # Graph traversal algorithms
├── topology.ex               # Topological operations
└── visualization.ex          # Graph visualization
```

### Metrics

```
lib/cns/
├── metrics.ex                # Core metrics computation
└── metrics/
    └── chirality.ex          # Chirality/polarity metrics
```

### Training

```
lib/cns/
├── training.ex               # Training utilities
└── training/
    └── evaluation.ex         # Training evaluation
```

---

## Configuration

```elixir
# config/config.exs
config :cns,
  default_model: "gpt-4",
  max_iterations: 5,
  convergence_threshold: 0.85,
  evidence_validation: true,
  telemetry_enabled: true

# Quality targets (used by adapters)
config :cns,
  schema_compliance_threshold: 0.95,
  citation_accuracy_threshold: 0.95,
  mean_entailment_threshold: 0.50

# Model-specific settings
config :cns, CNS.Proposer,
  temperature: 0.7,
  max_tokens: 2000

config :cns, CNS.Antagonist,
  temperature: 0.8,
  max_tokens: 2000,
  critique_depth: :thorough

config :cns, CNS.Synthesizer,
  temperature: 0.3,
  max_tokens: 3000,
  citation_validity_weight: 0.4
```

---

## Development

### Prerequisites

- Elixir 1.14+
- OTP 25+
- Mix build tool

### Setup

```bash
# Clone the repository
git clone https://github.com/North-Shore-AI/cns.git
cd cns

# Install dependencies
mix deps.get

# Run tests
mix test

# Run tests with coverage
mix test --cover

# Generate documentation
mix docs

# Run static analysis
mix dialyzer
```

### Testing

```bash
# Run all tests
mix test

# Run specific test file
mix test test/cns/synthesizer_test.exs

# Run with verbose output
mix test --trace

# Run property-based tests
mix test --only property
```

---

## Research Foundation

CNS is based on research in:

- Dialectical reasoning and Hegelian synthesis
- Multi-agent debate systems for AI alignment
- Evidence-grounded natural language inference
- Claim verification and fact-checking
- Topological data analysis for semantic structure

Key theoretical contributions:
- **Convergence Theorem**: Proof that the dialectical process terminates
- **Synthesis Quality Bounds**: Theoretical guarantees on output coherence
- **Evidence Chain Validity**: Formal model for citation trustworthiness
- **Topology-Logic Correlation**: β₁ surrogates predict logical validity

---

## Related Repositories

| Repository | Purpose |
|------------|---------|
| [crucible_framework](https://github.com/North-Shore-AI/crucible_framework) | Experiment engine with IR and pipeline |
| [cns_crucible](https://github.com/North-Shore-AI/cns_crucible) | CNS + Crucible integration harness |
| [tinkex](https://github.com/North-Shore-AI/tinkex) | Tinker SDK for LoRA training |

---

## License

Copyright 2025 North-Shore-AI

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

---

## Acknowledgments

- North-Shore-AI organization for infrastructure and support
- Crucible Framework for reliable LLM orchestration
- Tinkex for LoRA training capabilities
- The Elixir community for excellent tooling

---

<p align="center">
  Made with dialectical reasoning by <a href="https://github.com/North-Shore-AI">North-Shore-AI</a>
</p>

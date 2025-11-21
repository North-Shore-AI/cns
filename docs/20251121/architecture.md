# CNS Architecture

## Overview

CNS (Chiral Narrative Synthesis) implements a dialectical reasoning framework for automated knowledge discovery. This document describes the system architecture, core data structures, theoretical foundations, and integration points.

## System Philosophy

CNS is built on the principle that knowledge emerges through the resolution of conflicting claims. Rather than treating contradictions as errors, CNS views them as opportunities for synthesis - generating richer, more nuanced understanding through structured dialectical processes.

### Hegelian Foundations

The three-agent architecture mirrors the classical dialectical triad:

1. **Thesis (Proposer)**: Initial claims and hypotheses
2. **Antithesis (Antagonist)**: Counter-claims and challenges
3. **Synthesis (Synthesizer)**: Reconciliation into coherent narrative

## Core Data Structures

### Structured Narrative Object (SNO)

The SNO is the fundamental unit of knowledge in CNS:

```elixir
defmodule CNS.SNO do
  @type t :: %__MODULE__{
    id: String.t(),
    claim: String.t(),
    evidence: [CNS.Evidence.t()],
    confidence: float(),
    provenance: CNS.Provenance.t(),
    metadata: map(),
    children: [t()],
    synthesis_history: [CNS.SynthesisRecord.t()]
  }

  defstruct [
    :id,
    :claim,
    :evidence,
    :confidence,
    :provenance,
    :metadata,
    :children,
    :synthesis_history
  ]
end
```

#### SNO Properties

- **Composability**: SNOs can be nested to represent hierarchical knowledge
- **Traceability**: Full provenance chain from source to synthesis
- **Quantifiable Confidence**: Numeric scores enable automated reasoning
- **Evidence-Grounded**: Every claim links to verifiable sources

### Evidence Structure

```elixir
defmodule CNS.Evidence do
  @type t :: %__MODULE__{
    id: String.t(),
    source: String.t(),
    content: String.t(),
    validity: float(),
    relevance: float(),
    retrieval_method: atom(),
    timestamp: DateTime.t()
  }

  defstruct [
    :id,
    :source,
    :content,
    :validity,
    :relevance,
    :retrieval_method,
    :timestamp
  ]
end
```

### Provenance Chain

```elixir
defmodule CNS.Provenance do
  @type t :: %__MODULE__{
    origin: :proposer | :antagonist | :synthesizer | :external,
    parent_ids: [String.t()],
    transformation: String.t(),
    model_id: String.t(),
    timestamp: DateTime.t()
  }
end
```

## Three-Agent Pipeline

### Architecture Diagram

```
+------------------+     +-------------------+     +------------------+
|    Input Data    |     |   Evidence Store  |     |   Model Backend  |
+--------+---------+     +---------+---------+     +--------+---------+
         |                         |                        |
         v                         v                        v
+--------+---------+     +---------+---------+     +--------+---------+
|                  |     |                   |     |                  |
|    PROPOSER      +---->+    ANTAGONIST     +---->+   SYNTHESIZER    |
|                  |     |                   |     |                  |
| - Claim Extract  |     | - Counter-claims  |     | - Reconciliation |
| - Initial SNOs   |     | - Evidence Check  |     | - Coherence      |
| - Confidence     |     | - Challenges      |     | - Final SNO      |
|                  |     |                   |     |                  |
+--------+---------+     +---------+---------+     +--------+---------+
         |                         |                        |
         |                         |                        |
         +-------------------------+------------------------+
                                   |
                          +--------v--------+
                          |   Convergence   |
                          |     Monitor     |
                          +--------+--------+
                                   |
                    +--------------+--------------+
                    |                             |
               Converged                    Not Converged
                    |                             |
                    v                             v
           +--------+--------+           +--------+--------+
           |   Final Output  |           | Iterate Pipeline|
           |      SNO        |           |  (max N times)  |
           +-----------------+           +-----------------+
```

### 1. Proposer Agent

**Purpose**: Generate initial claims and hypotheses from input data.

**Responsibilities**:
- Parse unstructured text into candidate claims
- Assign initial confidence scores based on linguistic markers
- Identify supporting evidence in source material
- Structure output as SNOs for downstream processing

**Configuration**:
```elixir
%CNS.ProposerConfig{
  model: "gpt-4",
  temperature: 0.7,          # Higher for creative claim generation
  max_claims: 10,            # Maximum claims per input
  min_confidence: 0.3,       # Filter low-confidence claims
  evidence_extraction: true  # Auto-extract evidence from text
}
```

**Processing Flow**:
1. Receive input text or data
2. Apply claim extraction prompts
3. Score confidence using linguistic analysis
4. Extract and link evidence
5. Output list of SNOs

### 2. Antagonist Agent

**Purpose**: Challenge claims with counter-evidence and alternative interpretations.

**Responsibilities**:
- Identify weaknesses in proposed claims
- Generate counter-arguments and alternative hypotheses
- Retrieve contradicting evidence from knowledge bases
- Score the strength of challenges

**Configuration**:
```elixir
%CNS.AntagonistConfig{
  model: "gpt-4",
  temperature: 0.8,           # Higher for diverse challenges
  critique_depth: :thorough,  # :quick | :standard | :thorough
  evidence_search: true,      # Search for counter-evidence
  max_challenges: 5           # Challenges per claim
}
```

**Challenge Strategies**:
- **Logical**: Identify fallacies and reasoning errors
- **Empirical**: Find contradicting evidence
- **Scope**: Question generalizability
- **Alternative**: Propose competing explanations
- **Methodology**: Challenge underlying assumptions

### 3. Synthesizer Agent

**Purpose**: Reconcile thesis and antithesis into coherent synthesis.

**Responsibilities**:
- Integrate conflicting claims into nuanced positions
- Preserve valid insights from both sides
- Produce higher-confidence, evidence-grounded output
- Maintain logical coherence and clarity

**Configuration**:
```elixir
%CNS.SynthesizerConfig{
  model: "gpt-4",
  temperature: 0.3,              # Lower for consistent synthesis
  citation_validity_weight: 0.4, # Weight for evidence quality
  coherence_threshold: 0.8,      # Minimum synthesis quality
  preserve_nuance: true          # Maintain complexity in output
}
```

**Synthesis Strategies**:
- **Integration**: Combine compatible elements from both sides
- **Qualification**: Add conditions and scope limitations
- **Hierarchy**: Establish when each claim applies
- **Transformation**: Reframe at higher level of abstraction

## Convergence Theory

### Convergence Criterion

The pipeline iterates until convergence is achieved:

```elixir
def converged?(synthesis, config) do
  synthesis.confidence >= config.convergence_threshold and
  synthesis.coherence_score >= config.coherence_threshold and
  synthesis.evidence_coverage >= config.evidence_threshold
end
```

### Convergence Theorem

**Theorem**: For any finite set of input claims with bounded evidence, the CNS dialectical process terminates in at most `k` iterations, where `k` depends on the convergence threshold and initial claim diversity.

**Proof Sketch**:
1. Each synthesis iteration reduces the claim space dimensionality
2. Evidence grounding prevents unbounded claim generation
3. Confidence scores form a martingale with bounded variance
4. The process converges to a fixed point in the SNO space

### Convergence Metrics

```elixir
defmodule CNS.ConvergenceMetrics do
  @type t :: %__MODULE__{
    iteration: integer(),
    confidence_delta: float(),
    claim_entropy: float(),
    evidence_coverage: float(),
    coherence_score: float(),
    synthesis_quality: float()
  }
end
```

## Integration with Crucible

CNS integrates with the Crucible framework for reliable LLM orchestration:

### Ensemble Voting

Use multiple models for robust claim extraction:

```elixir
config = %CNS.Config{
  proposer: %{
    ensemble: true,
    models: ["gpt-4", "claude-3", "gemini-pro"],
    voting_strategy: :weighted
  }
}
```

### Request Hedging

Reduce tail latency for time-sensitive synthesis:

```elixir
config = %CNS.Config{
  hedging: %{
    enabled: true,
    strategy: :percentile,
    percentile: 95
  }
}
```

### Telemetry Integration

Full observability via Crucible telemetry:

```elixir
:telemetry.attach_many(
  "cns-metrics",
  [
    [:cns, :proposer, :extract],
    [:cns, :antagonist, :challenge],
    [:cns, :synthesizer, :reconcile],
    [:cns, :pipeline, :converge]
  ],
  &CNS.Telemetry.handle_event/4,
  nil
)
```

## Integration with Tinkex

CNS supports domain-specific fine-tuning via Tinkex LoRA training:

### Training Pipeline

```elixir
# Define training configuration
training_config = %Tinkex.LoRA.Config{
  base_model: "mistral-7b",
  rank: 16,
  alpha: 32,
  target_modules: ["q_proj", "v_proj"],
  dataset: "scifact-dialectical",
  epochs: 3
}

# Train domain-specific adapter
{:ok, adapter} = Tinkex.LoRA.train(training_config)

# Use in CNS pipeline
cns_config = %CNS.Config{
  synthesizer: %{
    model: "mistral-7b",
    lora_adapter: adapter.path
  }
}
```

### Custom Metrics

Track domain-specific quality metrics:

```elixir
metrics = %{
  citation_accuracy: 0.92,
  claim_coverage: 0.88,
  synthesis_coherence: 0.85
}
```

## Data Flow Architecture

### Input Processing

```
Raw Text/Data
      |
      v
+-----+------+
|   Parser   |
+-----+------+
      |
      v
Structured Input
      |
      v
+-----+------+
|  Proposer  |
+-----+------+
      |
      v
Initial SNOs
```

### Evidence Pipeline

```
Claim (SNO)
      |
      v
+-----+------+
|  Evidence  |
|  Retriever |
+-----+------+
      |
      v
+-----+------+
|  Validity  |
|   Scorer   |
+-----+------+
      |
      v
Grounded SNO
```

### Output Generation

```
Final Synthesis
      |
      v
+-----+------+
|  Reporter  |
+-----+------+
      |
      +------+------+------+
      |      |      |      |
      v      v      v      v
   JSON  Markdown HTML  LaTeX
```

## Scalability Considerations

### Horizontal Scaling

- **Stateless Agents**: Each agent can run on separate nodes
- **Message Passing**: OTP distribution for agent communication
- **Load Balancing**: Route claims to available agent instances

### Caching Strategy

```elixir
defmodule CNS.Cache do
  use Nebulex.Cache,
    otp_app: :cns,
    adapter: Nebulex.Adapters.Local

  # Cache evidence retrievals
  def get_evidence(claim_id) do
    get(claim_id) || fetch_and_cache(claim_id)
  end
end
```

### Batch Processing

```elixir
# Process multiple claims in parallel
claims
|> Task.async_stream(&CNS.Pipeline.run/1, max_concurrency: 10)
|> Enum.to_list()
```

## Security Considerations

### Input Validation

- All inputs validated against schema
- Evidence sources verified before inclusion
- Prompt injection detection via LlmGuard

### Output Sanitization

- Claims checked for harmful content
- PII redaction before output
- Citation validity verification

## Performance Characteristics

### Latency Profile

| Operation | P50 | P95 | P99 |
|-----------|-----|-----|-----|
| Claim Extraction | 1.2s | 2.5s | 4.0s |
| Antagonist Challenge | 1.5s | 3.0s | 5.0s |
| Synthesis | 2.0s | 4.0s | 6.0s |
| Full Pipeline | 8s | 15s | 25s |

### Resource Usage

- **Memory**: ~100MB base + 10MB per concurrent pipeline
- **CPU**: Minimal (I/O bound on LLM calls)
- **Network**: Depends on model backend

## Future Architecture Extensions

### Planned Enhancements

1. **Multi-Modal SNOs**: Support for image and video evidence
2. **Hierarchical Synthesis**: Nested dialectical processes
3. **Real-Time Streaming**: Incremental synthesis output
4. **Federated Learning**: Distributed training across domains

### Research Directions

- Formal verification of synthesis correctness
- Automated quality bounds estimation
- Cross-domain transfer learning
- Human-in-the-loop synthesis refinement

## Glossary

- **SNO**: Structured Narrative Object - core data structure for claims
- **Provenance**: Record of how a claim was derived
- **Convergence**: When the dialectical process reaches stable output
- **Evidence Grounding**: Linking claims to verifiable sources
- **Synthesis Quality**: Metric for coherence and completeness of output

# CNS API Reference

Complete API documentation for the CNS (Chiral Narrative Synthesis) library.

## Module Overview

| Module | Purpose |
|--------|---------|
| `CNS` | Main entry point and high-level API |
| `CNS.SNO` | Structured Narrative Object data structure |
| `CNS.Evidence` | Evidence records and validation |
| `CNS.Proposer` | Claim extraction and hypothesis generation |
| `CNS.Antagonist` | Counter-argument and challenge generation |
| `CNS.Synthesizer` | Claim reconciliation and synthesis |
| `CNS.Pipeline` | Full dialectical reasoning orchestration |
| `CNS.Config` | Configuration management |

---

## CNS

Main module providing high-level API functions.

### Functions

#### synthesize/2

Synthesize two conflicting claims into a coherent narrative.

```elixir
@spec synthesize(SNO.t(), SNO.t(), keyword()) :: {:ok, SNO.t()} | {:error, term()}
```

**Parameters**:
- `thesis` - The initial claim (SNO)
- `antithesis` - The opposing claim (SNO)
- `opts` - Optional keyword list of options

**Options**:
- `:model` - Model to use for synthesis (default: configured model)
- `:temperature` - Sampling temperature (default: 0.3)
- `:max_tokens` - Maximum output tokens (default: 3000)

**Returns**: `{:ok, synthesized_sno}` or `{:error, reason}`

**Example**:
```elixir
{:ok, synthesis} = CNS.synthesize(thesis, antithesis, model: "gpt-4")
```

#### extract_claims/2

Extract structured claims from unstructured text.

```elixir
@spec extract_claims(String.t(), keyword()) :: {:ok, [SNO.t()]} | {:error, term()}
```

**Parameters**:
- `text` - Input text to analyze
- `opts` - Optional keyword list

**Options**:
- `:max_claims` - Maximum claims to extract (default: 10)
- `:min_confidence` - Minimum confidence threshold (default: 0.3)
- `:extract_evidence` - Whether to extract evidence (default: true)

**Example**:
```elixir
{:ok, claims} = CNS.extract_claims(article_text, max_claims: 5)
```

#### run_pipeline/2

Run the full dialectical reasoning pipeline.

```elixir
@spec run_pipeline(String.t() | SNO.t(), Config.t()) :: {:ok, PipelineResult.t()} | {:error, term()}
```

**Parameters**:
- `input` - Input text or initial SNO
- `config` - Pipeline configuration

**Returns**: `{:ok, result}` with final synthesis and metadata

---

## CNS.SNO

Structured Narrative Object - the core data structure.

### Type Definition

```elixir
@type t :: %__MODULE__{
  id: String.t(),
  claim: String.t(),
  evidence: [Evidence.t()],
  confidence: float(),
  provenance: Provenance.t() | nil,
  metadata: map(),
  children: [t()],
  synthesis_history: [SynthesisRecord.t()]
}
```

### Functions

#### new/1

Create a new SNO with the given attributes.

```elixir
@spec new(map()) :: t()
```

**Example**:
```elixir
sno = CNS.SNO.new(%{
  claim: "Machine learning improves diagnosis accuracy",
  confidence: 0.85,
  evidence: [evidence1, evidence2]
})
```

#### validate/1

Validate an SNO structure.

```elixir
@spec validate(t()) :: :ok | {:error, [String.t()]}
```

**Example**:
```elixir
:ok = CNS.SNO.validate(sno)
```

#### merge/2

Merge two SNOs, combining their evidence and metadata.

```elixir
@spec merge(t(), t()) :: t()
```

#### to_json/1

Serialize SNO to JSON format.

```elixir
@spec to_json(t()) :: String.t()
```

#### from_json/1

Deserialize SNO from JSON.

```elixir
@spec from_json(String.t()) :: {:ok, t()} | {:error, term()}
```

#### confidence_weighted_merge/2

Merge SNOs with confidence-weighted claim combination.

```elixir
@spec confidence_weighted_merge(t(), t()) :: t()
```

---

## CNS.Evidence

Evidence records with source attribution and validity scoring.

### Type Definition

```elixir
@type t :: %__MODULE__{
  id: String.t(),
  source: String.t(),
  content: String.t(),
  validity: float(),
  relevance: float(),
  retrieval_method: atom(),
  timestamp: DateTime.t(),
  metadata: map()
}
```

### Functions

#### new/1

Create a new Evidence record.

```elixir
@spec new(map()) :: t()
```

**Example**:
```elixir
evidence = CNS.Evidence.new(%{
  source: "Nature 2023",
  content: "Study findings...",
  validity: 0.92,
  relevance: 0.88
})
```

#### validate/1

Validate an Evidence record.

```elixir
@spec validate(t()) :: :ok | {:error, [String.t()]}
```

#### score/1

Calculate overall evidence score (validity * relevance).

```elixir
@spec score(t()) :: float()
```

**Example**:
```elixir
score = CNS.Evidence.score(evidence)  # => 0.8096
```

#### merge_evidence/1

Merge multiple evidence records, combining scores.

```elixir
@spec merge_evidence([t()]) :: t()
```

#### from_citation/1

Create evidence from a citation string.

```elixir
@spec from_citation(String.t()) :: {:ok, t()} | {:error, term()}
```

---

## CNS.Proposer

Claim extraction and hypothesis generation agent.

### Functions

#### extract_claims/2

Extract claims from input text.

```elixir
@spec extract_claims(String.t(), keyword()) :: {:ok, [SNO.t()]} | {:error, term()}
```

**Options**:
- `:model` - LLM model to use
- `:temperature` - Sampling temperature (default: 0.7)
- `:max_claims` - Maximum claims (default: 10)
- `:min_confidence` - Confidence threshold (default: 0.3)
- `:extract_evidence` - Auto-extract evidence (default: true)

**Example**:
```elixir
{:ok, claims} = CNS.Proposer.extract_claims(text, model: "gpt-4")
```

#### generate_hypothesis/2

Generate hypotheses for a research question.

```elixir
@spec generate_hypothesis(String.t(), keyword()) :: {:ok, [SNO.t()]} | {:error, term()}
```

**Example**:
```elixir
{:ok, hypotheses} = CNS.Proposer.generate_hypothesis(
  "What causes antibiotic resistance?",
  max_hypotheses: 3
)
```

#### refine_claim/2

Refine an existing claim with additional evidence.

```elixir
@spec refine_claim(SNO.t(), [Evidence.t()]) :: {:ok, SNO.t()} | {:error, term()}
```

#### score_confidence/1

Calculate confidence score for a claim.

```elixir
@spec score_confidence(SNO.t()) :: float()
```

### Configuration

```elixir
@type config :: %{
  model: String.t(),
  temperature: float(),
  max_claims: integer(),
  min_confidence: float(),
  evidence_extraction: boolean()
}
```

---

## CNS.Antagonist

Counter-argument and challenge generation agent.

### Functions

#### challenge/2

Generate challenges for a claim.

```elixir
@spec challenge(SNO.t(), keyword()) :: {:ok, [Challenge.t()]} | {:error, term()}
```

**Options**:
- `:model` - LLM model to use
- `:temperature` - Sampling temperature (default: 0.8)
- `:critique_depth` - `:quick | :standard | :thorough`
- `:max_challenges` - Maximum challenges (default: 5)
- `:evidence_search` - Search for counter-evidence (default: true)

**Example**:
```elixir
{:ok, challenges} = CNS.Antagonist.challenge(claim, critique_depth: :thorough)
```

#### generate_antithesis/2

Generate an antithesis SNO for a thesis.

```elixir
@spec generate_antithesis(SNO.t(), keyword()) :: {:ok, SNO.t()} | {:error, term()}
```

**Example**:
```elixir
{:ok, antithesis} = CNS.Antagonist.generate_antithesis(thesis)
```

#### find_counter_evidence/2

Search for evidence contradicting a claim.

```elixir
@spec find_counter_evidence(SNO.t(), keyword()) :: {:ok, [Evidence.t()]} | {:error, term()}
```

#### score_challenge_strength/1

Calculate the strength of a challenge.

```elixir
@spec score_challenge_strength(Challenge.t()) :: float()
```

### Challenge Type

```elixir
@type Challenge.t :: %{
  type: :logical | :empirical | :scope | :alternative | :methodology,
  content: String.t(),
  evidence: [Evidence.t()],
  strength: float()
}
```

### Configuration

```elixir
@type config :: %{
  model: String.t(),
  temperature: float(),
  critique_depth: :quick | :standard | :thorough,
  max_challenges: integer(),
  evidence_search: boolean()
}
```

---

## CNS.Synthesizer

Claim reconciliation and synthesis agent.

### Functions

#### synthesize/3

Synthesize thesis and antithesis into coherent output.

```elixir
@spec synthesize(SNO.t(), SNO.t(), keyword()) :: {:ok, SNO.t()} | {:error, term()}
```

**Options**:
- `:model` - LLM model to use
- `:temperature` - Sampling temperature (default: 0.3)
- `:citation_validity_weight` - Weight for evidence quality (default: 0.4)
- `:preserve_nuance` - Maintain complexity (default: true)

**Example**:
```elixir
{:ok, synthesis} = CNS.Synthesizer.synthesize(
  thesis,
  antithesis,
  citation_validity_weight: 0.5
)
```

#### reconcile/2

Reconcile a list of conflicting claims.

```elixir
@spec reconcile([SNO.t()], keyword()) :: {:ok, SNO.t()} | {:error, term()}
```

**Example**:
```elixir
{:ok, unified} = CNS.Synthesizer.reconcile(claims, model: "gpt-4")
```

#### integrate_evidence/2

Integrate evidence from multiple SNOs.

```elixir
@spec integrate_evidence([SNO.t()]) :: [Evidence.t()]
```

#### calculate_coherence/1

Calculate coherence score for a synthesis.

```elixir
@spec calculate_coherence(SNO.t()) :: float()
```

#### calculate_synthesis_quality/1

Calculate overall synthesis quality metric.

```elixir
@spec calculate_synthesis_quality(SNO.t()) :: float()
```

### Configuration

```elixir
@type config :: %{
  model: String.t(),
  temperature: float(),
  citation_validity_weight: float(),
  coherence_threshold: float(),
  preserve_nuance: boolean()
}
```

---

## CNS.Pipeline

Full dialectical reasoning orchestration.

### Functions

#### run/2

Run the complete dialectical pipeline.

```elixir
@spec run(String.t() | SNO.t(), Config.t()) :: {:ok, PipelineResult.t()} | {:error, term()}
```

**Example**:
```elixir
config = CNS.Config.new(max_iterations: 5)
{:ok, result} = CNS.Pipeline.run("What are the effects of caffeine?", config)
```

#### run_iteration/2

Run a single pipeline iteration.

```elixir
@spec run_iteration(PipelineState.t(), Config.t()) :: {:ok, PipelineState.t()} | {:error, term()}
```

#### check_convergence/2

Check if the pipeline has converged.

```elixir
@spec check_convergence(SNO.t(), Config.t()) :: boolean()
```

#### get_metrics/1

Get convergence metrics for current state.

```elixir
@spec get_metrics(PipelineState.t()) :: ConvergenceMetrics.t()
```

### PipelineResult Type

```elixir
@type PipelineResult.t :: %{
  final_synthesis: SNO.t(),
  iterations: integer(),
  convergence_metrics: ConvergenceMetrics.t(),
  evidence_chain: [Evidence.t()],
  synthesis_history: [SNO.t()],
  telemetry: map()
}
```

### PipelineState Type

```elixir
@type PipelineState.t :: %{
  current_thesis: SNO.t(),
  current_antithesis: SNO.t() | nil,
  current_synthesis: SNO.t() | nil,
  iteration: integer(),
  history: [SNO.t()],
  metrics: ConvergenceMetrics.t()
}
```

---

## CNS.Config

Configuration management for CNS pipelines.

### Type Definition

```elixir
@type t :: %__MODULE__{
  proposer: map(),
  antagonist: map(),
  synthesizer: map(),
  max_iterations: integer(),
  convergence_threshold: float(),
  evidence_validation: boolean(),
  telemetry_enabled: boolean()
}
```

### Functions

#### new/1

Create a new configuration.

```elixir
@spec new(keyword()) :: t()
```

**Example**:
```elixir
config = CNS.Config.new(
  max_iterations: 10,
  convergence_threshold: 0.9
)
```

#### validate/1

Validate a configuration.

```elixir
@spec validate(t()) :: :ok | {:error, [String.t()]}
```

#### merge/2

Merge two configurations.

```elixir
@spec merge(t(), t()) :: t()
```

#### from_env/0

Load configuration from environment variables.

```elixir
@spec from_env() :: t()
```

### Default Configuration

```elixir
%CNS.Config{
  proposer: %{
    model: "gpt-4",
    temperature: 0.7,
    max_claims: 10
  },
  antagonist: %{
    model: "gpt-4",
    temperature: 0.8,
    critique_depth: :standard
  },
  synthesizer: %{
    model: "gpt-4",
    temperature: 0.3,
    citation_validity_weight: 0.4
  },
  max_iterations: 5,
  convergence_threshold: 0.85,
  evidence_validation: true,
  telemetry_enabled: true
}
```

---

## Error Handling

### Error Types

```elixir
@type error ::
  {:invalid_input, String.t()} |
  {:model_error, String.t()} |
  {:convergence_failed, integer()} |
  {:validation_error, [String.t()]} |
  {:evidence_error, String.t()}
```

### Error Handling Example

```elixir
case CNS.Pipeline.run(input, config) do
  {:ok, result} ->
    IO.inspect(result.final_synthesis)

  {:error, {:convergence_failed, iterations}} ->
    Logger.warn("Failed to converge after #{iterations} iterations")

  {:error, {:model_error, message}} ->
    Logger.error("Model error: #{message}")

  {:error, reason} ->
    Logger.error("Pipeline error: #{inspect(reason)}")
end
```

---

## Telemetry Events

CNS emits the following telemetry events:

| Event | Measurements | Metadata |
|-------|-------------|----------|
| `[:cns, :proposer, :extract]` | `duration`, `claim_count` | `model`, `input_size` |
| `[:cns, :antagonist, :challenge]` | `duration`, `challenge_count` | `model`, `claim_id` |
| `[:cns, :synthesizer, :reconcile]` | `duration`, `quality_score` | `model`, `iteration` |
| `[:cns, :pipeline, :start]` | - | `config`, `input_type` |
| `[:cns, :pipeline, :stop]` | `duration`, `iterations` | `converged`, `final_confidence` |
| `[:cns, :pipeline, :exception]` | `duration` | `error`, `stacktrace` |

### Attaching Handlers

```elixir
:telemetry.attach_many(
  "cns-logger",
  [
    [:cns, :pipeline, :start],
    [:cns, :pipeline, :stop],
    [:cns, :pipeline, :exception]
  ],
  fn event, measurements, metadata, _config ->
    Logger.info("#{inspect(event)}: #{inspect(measurements)}")
  end,
  nil
)
```

---

## Types Reference

### Core Types

```elixir
@type claim :: String.t()
@type confidence :: float()  # 0.0 to 1.0
@type validity :: float()    # 0.0 to 1.0
@type relevance :: float()   # 0.0 to 1.0
```

### Provenance Type

```elixir
@type Provenance.t :: %{
  origin: :proposer | :antagonist | :synthesizer | :external,
  parent_ids: [String.t()],
  transformation: String.t(),
  model_id: String.t(),
  timestamp: DateTime.t()
}
```

### ConvergenceMetrics Type

```elixir
@type ConvergenceMetrics.t :: %{
  iteration: integer(),
  confidence_delta: float(),
  claim_entropy: float(),
  evidence_coverage: float(),
  coherence_score: float(),
  synthesis_quality: float()
}
```

### SynthesisRecord Type

```elixir
@type SynthesisRecord.t :: %{
  iteration: integer(),
  thesis_id: String.t(),
  antithesis_id: String.t(),
  synthesis_id: String.t(),
  quality_score: float(),
  timestamp: DateTime.t()
}
```

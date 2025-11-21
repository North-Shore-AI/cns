# Getting Started with CNS

Welcome to CNS (Chiral Narrative Synthesis)! This tutorial will guide you through setting up CNS and running your first dialectical reasoning pipeline.

## Installation

### Add Dependency

Add CNS to your `mix.exs`:

```elixir
def deps do
  [
    {:cns, "~> 0.1.0"}
  ]
end
```

### Install Dependencies

```bash
mix deps.get
mix compile
```

### Configuration

Add basic configuration to `config/config.exs`:

```elixir
config :cns,
  default_model: "gpt-4",
  max_iterations: 5,
  convergence_threshold: 0.85
```

Set your API key (for OpenAI or other providers):

```bash
export OPENAI_API_KEY="your-api-key"
```

Or in `config/runtime.exs`:

```elixir
config :cns, :openai_api_key, System.get_env("OPENAI_API_KEY")
```

## Your First Synthesis

### Step 1: Create Claims

Create two conflicting claims as Structured Narrative Objects (SNOs):

```elixir
alias CNS.{SNO, Evidence}

# Thesis: Initial claim with supporting evidence
thesis = %SNO{
  claim: "Coffee consumption improves cognitive performance",
  evidence: [
    %Evidence{
      source: "Nehlig 2010",
      content: "Caffeine enhances attention, psychomotor performance, and memory",
      validity: 0.85,
      relevance: 0.90
    }
  ],
  confidence: 0.75
}

# Antithesis: Opposing claim with its evidence
antithesis = %SNO{
  claim: "Coffee consumption has negative health effects",
  evidence: [
    %Evidence{
      source: "Poole 2017",
      content: "High caffeine intake associated with increased anxiety and sleep disruption",
      validity: 0.82,
      relevance: 0.85
    }
  ],
  confidence: 0.70
}
```

### Step 2: Run Synthesis

Use the `CNS.synthesize/2` function to reconcile the claims:

```elixir
{:ok, synthesis} = CNS.synthesize(thesis, antithesis)

IO.puts("Synthesized claim: #{synthesis.claim}")
IO.puts("Confidence: #{synthesis.confidence}")
IO.puts("Evidence count: #{length(synthesis.evidence)}")
```

**Expected output**:
```
Synthesized claim: Coffee consumption provides cognitive benefits including
enhanced attention and memory, but optimal intake levels vary by individual
and excessive consumption may lead to anxiety and sleep disruption.
Confidence: 0.82
Evidence count: 2
```

### Step 3: Examine the Result

Explore the synthesis details:

```elixir
# Check the evidence chain
Enum.each(synthesis.evidence, fn ev ->
  IO.puts("Source: #{ev.source}")
  IO.puts("Validity: #{ev.validity}")
  IO.puts("---")
end)

# Check provenance
IO.inspect(synthesis.provenance)
```

## Using the Full Pipeline

For more complex scenarios, use the complete three-agent pipeline.

### Configure the Pipeline

```elixir
config = %CNS.Config{
  proposer: %{
    model: "gpt-4",
    temperature: 0.7,
    max_claims: 5
  },
  antagonist: %{
    model: "gpt-4",
    temperature: 0.8,
    critique_depth: :thorough
  },
  synthesizer: %{
    model: "gpt-4",
    temperature: 0.3,
    citation_validity_weight: 0.4
  },
  max_iterations: 5,
  convergence_threshold: 0.85
}
```

### Run the Pipeline

```elixir
question = "What are the long-term effects of remote work on employee well-being?"

{:ok, result} = CNS.Pipeline.run(question, config)

# Access results
IO.puts("Final synthesis: #{result.final_synthesis.claim}")
IO.puts("Iterations: #{result.iterations}")
IO.puts("Convergence score: #{result.convergence_metrics.synthesis_quality}")
```

### Inspect the Process

```elixir
# View synthesis history
Enum.each(result.synthesis_history, fn sno ->
  IO.puts("Iteration #{sno.provenance.iteration}: #{sno.claim}")
end)

# Check convergence metrics
metrics = result.convergence_metrics
IO.puts("Confidence delta: #{metrics.confidence_delta}")
IO.puts("Evidence coverage: #{metrics.evidence_coverage}")
IO.puts("Coherence score: #{metrics.coherence_score}")
```

## Extracting Claims from Text

Use the Proposer to extract structured claims from unstructured text.

### Basic Extraction

```elixir
text = """
Recent meta-analyses have shown that mindfulness meditation can reduce
symptoms of anxiety and depression. However, the effect sizes are often
small to moderate, and the quality of studies varies significantly.
Some researchers argue that placebo effects may account for much of the
observed benefit.
"""

{:ok, claims} = CNS.Proposer.extract_claims(text)

Enum.each(claims, fn claim ->
  IO.puts("Claim: #{claim.claim}")
  IO.puts("Confidence: #{claim.confidence}")
  Enum.each(claim.evidence, fn ev ->
    IO.puts("  Evidence: #{ev.content}")
  end)
  IO.puts("---")
end)
```

### With Options

```elixir
{:ok, claims} = CNS.Proposer.extract_claims(text,
  max_claims: 3,
  min_confidence: 0.5,
  extract_evidence: true
)
```

## Generating Challenges

Use the Antagonist to generate counter-arguments.

```elixir
claim = %SNO{
  claim: "Artificial general intelligence will be achieved within 10 years",
  confidence: 0.60,
  evidence: []
}

{:ok, challenges} = CNS.Antagonist.challenge(claim, critique_depth: :thorough)

Enum.each(challenges, fn challenge ->
  IO.puts("Type: #{challenge.type}")
  IO.puts("Challenge: #{challenge.content}")
  IO.puts("Strength: #{challenge.strength}")
  IO.puts("---")
end)
```

## Working with Evidence

### Creating Evidence

```elixir
evidence = %CNS.Evidence{
  source: "Smith et al. 2023",
  content: "Randomized controlled trial with N=500 participants...",
  validity: 0.90,
  relevance: 0.85,
  retrieval_method: :manual,
  timestamp: DateTime.utc_now()
}
```

### Calculating Evidence Score

```elixir
score = CNS.Evidence.score(evidence)
IO.puts("Evidence score: #{score}")  # 0.765 (validity * relevance)
```

### Merging Evidence

```elixir
merged = CNS.Evidence.merge_evidence([evidence1, evidence2, evidence3])
```

## Error Handling

CNS functions return `{:ok, result}` or `{:error, reason}`:

```elixir
case CNS.synthesize(thesis, antithesis) do
  {:ok, synthesis} ->
    IO.puts("Success: #{synthesis.claim}")

  {:error, {:convergence_failed, iterations}} ->
    IO.puts("Failed to converge after #{iterations} iterations")

  {:error, {:validation_error, errors}} ->
    IO.puts("Validation errors: #{inspect(errors)}")

  {:error, reason} ->
    IO.puts("Error: #{inspect(reason)}")
end
```

## Observability

### Telemetry Integration

CNS emits telemetry events for monitoring:

```elixir
:telemetry.attach(
  "cns-logger",
  [:cns, :pipeline, :stop],
  fn _event, measurements, metadata, _config ->
    IO.puts("Pipeline completed in #{measurements.duration}ms")
    IO.puts("Converged: #{metadata.converged}")
  end,
  nil
)
```

### Logging

Enable debug logging for detailed output:

```elixir
Logger.configure(level: :debug)
```

## Next Steps

Now that you've completed the basics, explore these topics:

### 1. Architecture Deep Dive

Learn about the three-agent system and SNO data structures:

- [Architecture Guide](architecture.md)

### 2. API Reference

Explore all available functions and types:

- [API Reference](api_reference.md)

### 3. Custom Training

Train domain-specific adapters with Tinkex:

- [Training Guide](training_guide.md)

### 4. Integration

Connect CNS with other North-Shore-AI projects:

- **Crucible**: Use ensemble voting for reliable LLM calls
- **ExDataCheck**: Validate input data quality
- **LlmGuard**: Protect against adversarial inputs

## Example Projects

### Research Question Exploration

```elixir
defmodule MyApp.Research do
  def explore(question) do
    config = %CNS.Config{
      max_iterations: 10,
      convergence_threshold: 0.90
    }

    {:ok, result} = CNS.Pipeline.run(question, config)

    %{
      answer: result.final_synthesis.claim,
      confidence: result.final_synthesis.confidence,
      evidence: Enum.map(result.evidence_chain, & &1.source),
      iterations: result.iterations
    }
  end
end

# Usage
MyApp.Research.explore("What causes burnout in software developers?")
```

### Claim Verification

```elixir
defmodule MyApp.FactCheck do
  def verify(claim_text) do
    {:ok, [claim]} = CNS.Proposer.extract_claims(claim_text, max_claims: 1)
    {:ok, challenges} = CNS.Antagonist.challenge(claim)

    challenge_strength = Enum.reduce(challenges, 0, & &1.strength + &2) / length(challenges)

    cond do
      challenge_strength > 0.7 -> {:likely_false, challenges}
      challenge_strength < 0.3 -> {:likely_true, claim}
      true -> {:uncertain, {claim, challenges}}
    end
  end
end
```

### Literature Synthesis

```elixir
defmodule MyApp.LitReview do
  def synthesize_papers(paper_texts) do
    # Extract claims from each paper
    all_claims = Enum.flat_map(paper_texts, fn text ->
      {:ok, claims} = CNS.Proposer.extract_claims(text)
      claims
    end)

    # Synthesize all claims
    {:ok, synthesis} = CNS.Synthesizer.reconcile(all_claims)

    synthesis
  end
end
```

## Troubleshooting

### Common Issues

**"Model not responding"**
- Check API key is set correctly
- Verify network connectivity
- Check rate limits

**"Convergence failed"**
- Increase `max_iterations`
- Lower `convergence_threshold`
- Check input quality

**"Low confidence scores"**
- Improve evidence quality
- Add more supporting evidence
- Refine claim clarity

### Getting Help

- Check the [API Reference](api_reference.md)
- Review [Architecture](architecture.md) for system understanding
- Open an issue on GitHub

## Summary

You've learned how to:

1. Install and configure CNS
2. Create SNOs with claims and evidence
3. Run basic synthesis between two claims
4. Use the full three-agent pipeline
5. Extract claims from text
6. Generate challenges
7. Handle errors and observe the system

CNS provides a powerful framework for automated dialectical reasoning. Start with simple syntheses and gradually explore the full capabilities of the three-agent pipeline.

Happy synthesizing!

# CNS Implementation Prompt

**Generated:** 2025-12-25
**For:** Fresh Agent Implementation Task
**Repository:** `/home/home/p/g/North-Shore-AI/cns`

---

## Context

You are implementing features for CNS (Chiral Narrative Synthesis), an Elixir-based dialectical reasoning framework. CNS implements a three-agent system for automated knowledge discovery:

1. **Proposer** - Generates thesis claims from input
2. **Antagonist** - Creates antithesis challenges
3. **Synthesizer** - Reconciles conflicts into synthesis

The system uses topological analysis to detect circular reasoning and chirality metrics to measure polarity conflicts.

---

## Required Reading

Before implementing, read these files in order:

### Core Entry Points
1. `/home/home/p/g/North-Shore-AI/cns/mix.exs` - Dependencies and project structure
2. `/home/home/p/g/North-Shore-AI/cns/lib/cns.ex` - Main module (Lines 1-127)
3. `/home/home/p/g/North-Shore-AI/cns/README.md` - Full documentation

### Core Data Types (READ ALL)
4. `/home/home/p/g/North-Shore-AI/cns/lib/cns/sno.ex` - SNO struct (Lines 1-410)
   - Key: `new/2` at L78, `validate/1` at L108, `to_map/1` at L132
5. `/home/home/p/g/North-Shore-AI/cns/lib/cns/evidence.ex` - Evidence struct
6. `/home/home/p/g/North-Shore-AI/cns/lib/cns/challenge.ex` - Challenge struct
7. `/home/home/p/g/North-Shore-AI/cns/lib/cns/provenance.ex` - Provenance chain
8. `/home/home/p/g/North-Shore-AI/cns/lib/cns/config.ex` - Configuration

### Agent Modules (READ ALL)
9. `/home/home/p/g/North-Shore-AI/cns/lib/cns/agents/proposer.ex` (Lines 1-288)
   - Key: `extract_claims/2` at L41, `generate_hypothesis/2` at L69
10. `/home/home/p/g/North-Shore-AI/cns/lib/cns/agents/antagonist.ex` (Lines 1-408)
    - Key: `challenge/2` at L38, `find_contradictions/1` at L66
11. `/home/home/p/g/North-Shore-AI/cns/lib/cns/agents/synthesizer.ex` (Lines 1-337)
    - Key: `synthesize/3` at L37, `coherence_score/1` at L149
12. `/home/home/p/g/North-Shore-AI/cns/lib/cns/agents/pipeline.ex` (Lines 1-277)
    - Key: `run/2` at L38, `converged?/2` at L89

### Critic Modules
13. `/home/home/p/g/North-Shore-AI/cns/lib/cns/critics/critic.ex` - Base behaviour
14. `/home/home/p/g/North-Shore-AI/cns/lib/cns/critics/grounding.ex` - Evidence grounding
15. `/home/home/p/g/North-Shore-AI/cns/lib/cns/critics/logic.ex` - Logical consistency
16. `/home/home/p/g/North-Shore-AI/cns/lib/cns/critics/novelty.ex` - Novelty assessment
17. `/home/home/p/g/North-Shore-AI/cns/lib/cns/critics/bias.ex` - Bias detection
18. `/home/home/p/g/North-Shore-AI/cns/lib/cns/critics/causal.ex` - Causal validity

### Topology Modules (KEY FOR UNDERSTANDING)
19. `/home/home/p/g/North-Shore-AI/cns/lib/cns/topology.ex` (Lines 1-298)
    - Key: `build_graph/1` at L21, `invariants/1` at L47, `beta1/2` at L218
20. `/home/home/p/g/North-Shore-AI/cns/lib/cns/topology/surrogates.ex` (Lines 1-331)
    - Key: `compute_beta1_surrogate/1` at L69, `compute_fragility_surrogate/2` at L111
21. `/home/home/p/g/North-Shore-AI/cns/lib/cns/topology/persistence.ex` (Lines 1-637)
    - Key: `compute/2` at L144, `compare/3` at L251
22. `/home/home/p/g/North-Shore-AI/cns/lib/cns/topology/adapter.ex` (Lines 1-772)
    - Key: `sno_embeddings/2` at L102, `interpret_betti/1` at L337

### Metrics Modules
23. `/home/home/p/g/North-Shore-AI/cns/lib/cns/metrics.ex` (Lines 1-413)
    - Key: `quality_score/1` at L32, `chirality/1` at L133, `meets_targets?/1` at L260
24. `/home/home/p/g/North-Shore-AI/cns/lib/cns/metrics/chirality.ex` (Lines 1-194)
    - Key: `fisher_rao_distance/3` at L94, `compute_chirality_score/3` at L126

### Validation Modules
25. `/home/home/p/g/North-Shore-AI/cns/lib/cns/validation/semantic.ex` (Lines 1-418)
    - Key: `validate_claim/6` at L253, `compute_entailment/3` at L331
26. `/home/home/p/g/North-Shore-AI/cns/lib/cns/schema/parser.ex` (Lines 1-151)
    - Key: `parse_claims/1` at L46, `parse_relations/1` at L118

### Training Modules
27. `/home/home/p/g/North-Shore-AI/cns/lib/cns/training/training.ex` (Lines 1-339)
28. `/home/home/p/g/North-Shore-AI/cns/lib/cns/training/evaluation.ex` (Lines 1-211)

### Graph Utilities
29. `/home/home/p/g/North-Shore-AI/cns/lib/cns/graph/builder.ex` (Lines 1-154)

---

## Dialectical Flow

```
User Input
    |
    v
+-------------------+
| PROPOSER          |
| - extract_claims  |
| - score_confidence|
| - extract_evidence|
+-------------------+
    |
    | [SNO: thesis]
    v
+-------------------+
| ANTAGONIST        |
| - find_contradictions
| - find_evidence_gaps
| - find_scope_issues
| - find_logical_issues
| - generate_alternatives
+-------------------+
    |
    | [Challenge list]
    v
+-------------------+
| SYNTHESIZER       |
| - resolve_conflicts
| - merge_evidence  |
| - coherence_score |
| - entailment_score|
+-------------------+
    |
    | [SNO: synthesis]
    v
+-------------------+
| CONVERGENCE CHECK |
| - confidence >= threshold
| - coherence >= threshold
| - evidence_score >= threshold
+-------------------+
    |
    +--> Converged? --> Final SNO
    |
    +--> Not Converged --> Feed back to Proposer (iteration++)
```

---

## Critic Types

Each critic evaluates claims from a specific perspective:

### 1. Grounding Critic (`CNS.Critics.Grounding`)
- Validates evidence citations exist and are valid
- Checks evidence coverage (how much of claim is supported)
- Returns grounding score 0.0-1.0

### 2. Logic Critic (`CNS.Critics.Logic`)
- Detects logical fallacies (ad hominem, straw man, circular reasoning)
- Checks consistency within claim
- Returns logic score 0.0-1.0

### 3. Novelty Critic (`CNS.Critics.Novelty`)
- Compares claim to existing corpus
- Measures how much new information is added
- Returns novelty score 0.0-1.0

### 4. Bias Critic (`CNS.Critics.Bias`)
- Detects bias markers in language
- Checks for balanced perspective
- Returns neutrality score 0.0-1.0

### 5. Causal Critic (`CNS.Critics.Causal`)
- Validates causal claims have supporting evidence
- Checks causal chain validity
- Returns causal validity score 0.0-1.0

---

## SNO Graph Structure and Topology Metrics

### Building Graphs

SNOs form directed graphs through provenance chains:

```elixir
# Parent SNOs point to child (derived) SNOs
thesis.id --> synthesis.id <-- antithesis.id
```

### Betti Numbers (beta)

```
beta0 (H0) = Number of connected components
           = Number of claim clusters
           = If beta0 > 1, claims are disconnected

beta1 (H1) = Number of independent cycles
           = Circular reasoning patterns
           = If beta1 > 0, circular reasoning detected

beta2 (H2) = Number of voids
           = Higher-order logical structures
```

### Chirality Score

Measures semantic divergence between thesis and antithesis:

```elixir
# Fisher-Rao distance between embeddings
distance = fisher_rao_distance(thesis_embedding, antithesis_embedding, stats)

# Chirality components:
norm_distance = distance / (distance + 1.0)      # Normalized distance [0,1]
overlap_factor = 1.0 - evidence_overlap          # Evidence divergence
conflict_penalty = if polarity_conflict, do: 0.25, else: 0.0

# Final score (weights: distance 60%, overlap 20%, conflict 25%)
chirality_score = norm_distance * 0.6 + overlap_factor * 0.2 + conflict_penalty
```

### Fragility Score

Measures semantic instability via embedding variance:

```elixir
# High fragility = claims are semantically unstable
# Low fragility = claims are well-grounded

fragility = knn_variance(embeddings, k: 5)
```

---

## Quality Targets (CNS 3.0)

```elixir
%{
  schema_compliance: 0.95,    # SNOs must be 95%+ valid
  citation_accuracy: 0.95,    # Evidence citations 95%+ valid
  mean_entailment: 0.50       # Synthesis must entail from premises
}
```

---

## TDD Approach

### Step 1: Write Tests First

Before implementing any feature:

1. Create test file in `/home/home/p/g/North-Shore-AI/cns/test/cns/`
2. Write failing tests that describe expected behavior
3. Run `mix test` to confirm tests fail
4. Implement feature
5. Run `mix test` to confirm tests pass

### Step 2: Test Pattern

```elixir
defmodule CNS.YourModuleTest do
  use ExUnit.Case, async: true

  alias CNS.YourModule

  describe "function_name/arity" do
    test "describes expected behavior" do
      # Setup
      input = ...

      # Exercise
      result = YourModule.function_name(input)

      # Verify
      assert result == expected
    end

    test "handles edge case" do
      assert {:error, _} = YourModule.function_name(bad_input)
    end
  end
end
```

### Step 3: Run Quality Checks

After implementation:

```bash
# All tests pass
mix test

# No warnings
mix compile --warnings-as-errors

# Dialyzer clean
mix dialyzer

# Credo strict
mix credo --strict

# Coverage
mix test --cover
```

---

## Quality Requirements

### Code Standards

1. **No Warnings**
   ```bash
   mix compile --warnings-as-errors
   ```

2. **Dialyzer Clean**
   ```bash
   mix dialyzer
   ```
   - All @spec annotations must be accurate
   - No type mismatches

3. **Credo Strict**
   ```bash
   mix credo --strict
   ```
   - No TODO comments in production code
   - Consistent naming
   - Proper documentation

4. **All Tests Passing**
   ```bash
   mix test
   ```
   - Unit tests for all public functions
   - Edge case coverage
   - Doctest examples work

### Documentation Standards

1. Every module has `@moduledoc`
2. Every public function has `@doc`
3. Every public function has `@spec`
4. Complex functions have doctest examples

### Commit Standards

After implementing features:
1. Run all quality checks
2. Ensure README.md is updated if API changes
3. Add/update documentation as needed

---

## Current Gaps to Address

See `/home/home/p/g/North-Shore-AI/cns/docs/20251225/gaps.md` for full gap analysis.

### Priority 1: LLM Integration

The agents currently use heuristics. Real implementation needs:

```elixir
# In proposer.ex - replace heuristic extraction with LLM
defp extract_with_llm(text, opts) do
  prompt = build_extraction_prompt(text)
  {:ok, response} = LLM.complete(prompt, opts)
  parse_claims_from_response(response)
end
```

### Priority 2: Training Module

Currently stub implementation. Needs Crucible IR integration:

```elixir
# In training.ex - real training
def train(dataset, opts) do
  config = build_training_config(opts)
  {:ok, job} = CrucibleIR.submit_training(dataset, config)
  await_training(job)
end
```

### Priority 3: Critic Improvements

Critics use keyword matching. Need semantic analysis:

```elixir
# In logic.ex - semantic fallacy detection
def detect_fallacies(claim) do
  embedding = Encoder.encode(claim)
  fallacy_scores = FallacyClassifier.classify(embedding)
  filter_by_threshold(fallacy_scores, 0.5)
end
```

---

## Example Implementation Task

If asked to implement a new critic:

### Step 1: Create Test File

```elixir
# /home/home/p/g/North-Shore-AI/cns/test/cns/critics/new_critic_test.exs
defmodule CNS.Critics.NewCriticTest do
  use ExUnit.Case, async: true

  alias CNS.Critics.NewCritic
  alias CNS.SNO

  describe "evaluate/2" do
    test "returns score for valid claim" do
      sno = SNO.new("Valid claim", confidence: 0.8)
      {:ok, result} = NewCritic.evaluate(sno, [])

      assert result.score >= 0.0
      assert result.score <= 1.0
    end
  end
end
```

### Step 2: Create Module

```elixir
# /home/home/p/g/North-Shore-AI/cns/lib/cns/critics/new_critic.ex
defmodule CNS.Critics.NewCritic do
  @moduledoc """
  New critic for evaluating claims.
  """

  @behaviour CNS.Critics.Critic

  alias CNS.SNO

  @impl true
  @spec evaluate(SNO.t(), keyword()) :: {:ok, map()} | {:error, term()}
  def evaluate(%SNO{} = sno, opts \\ []) do
    # Implementation
  end
end
```

### Step 3: Verify

```bash
mix test test/cns/critics/new_critic_test.exs
mix compile --warnings-as-errors
mix dialyzer
mix credo --strict
```

---

## File Structure Reference

```
/home/home/p/g/North-Shore-AI/cns/
├── lib/
│   └── cns/
│       ├── agents/
│       │   ├── proposer.ex
│       │   ├── antagonist.ex
│       │   ├── synthesizer.ex
│       │   └── pipeline.ex
│       ├── critics/
│       │   ├── critic.ex (behaviour)
│       │   ├── grounding.ex
│       │   ├── logic.ex
│       │   ├── novelty.ex
│       │   ├── bias.ex
│       │   └── causal.ex
│       ├── topology/
│       │   ├── adapter.ex
│       │   ├── persistence.ex
│       │   ├── surrogates.ex
│       │   ├── tda.ex
│       │   └── fragility.ex
│       ├── metrics/
│       │   └── chirality.ex
│       ├── validation/
│       │   ├── semantic.ex
│       │   ├── citation.ex
│       │   └── model_loader.ex
│       ├── schema/
│       │   └── parser.ex
│       ├── pipeline/
│       │   ├── schema.ex
│       │   └── converters.ex
│       ├── training/
│       │   ├── training.ex
│       │   └── evaluation.ex
│       ├── graph/
│       │   ├── builder.ex
│       │   ├── traversal.ex
│       │   ├── topology.ex
│       │   └── visualization.ex
│       ├── embedding/
│       │   ├── encoder.ex
│       │   ├── gemini.ex
│       │   └── gemini_http.ex
│       ├── logic/
│       │   └── betti.ex
│       ├── sno.ex
│       ├── evidence.ex
│       ├── challenge.ex
│       ├── provenance.ex
│       ├── config.ex
│       ├── metrics.ex
│       ├── topology.ex
│       └── application.ex
├── test/
│   └── cns/
│       └── [corresponding test files]
├── mix.exs
└── README.md
```

---

## Commands Reference

```bash
# Install dependencies
mix deps.get

# Compile with warnings as errors
mix compile --warnings-as-errors

# Run all tests
mix test

# Run specific test file
mix test test/cns/agents/proposer_test.exs

# Run tests with coverage
mix test --cover

# Static analysis
mix dialyzer

# Code style
mix credo --strict

# Format code
mix format

# Generate docs
mix docs
```

---

## Contact Points

- **Main Module:** `CNS` in `/home/home/p/g/North-Shore-AI/cns/lib/cns.ex`
- **Pipeline Entry:** `CNS.Agents.Pipeline.run/2`
- **SNO Creation:** `CNS.SNO.new/2`
- **Topology Analysis:** `CNS.Topology.analyze_claim_network/2`
- **Metrics:** `CNS.Metrics.report/2`

# Meta-Critic Agent Specification

**Version**: 1.0.0
**Date**: 2025-12-27
**Status**: Proposed
**Author**: CNS Architecture Team

---

## 1. Executive Summary

The Meta-Critic is the 9th and final agent in the CNS (Critic-Network Synthesis) dialectical reasoning system. It serves as the orchestration layer that aggregates, weighs, and synthesizes evaluations from the 5 domain-specific critics (Grounding, Logic, Novelty, Bias, Causal) into unified quality assessments with meta-level reasoning.

### Current Critic Weights

| Critic | Weight | Focus |
|--------|--------|-------|
| Grounding | 0.40 | Factual accuracy, evidence quality |
| Logic | 0.30 | Logical consistency, entailment |
| Novelty | 0.15 | Originality, parsimony |
| Causal | 0.10 | Causal validity |
| Bias | 0.05 | Fairness, perspective balance |
| **Meta (NEW)** | **N/A** | **Orchestration, dynamic weighting** |

---

## 2. Responsibilities

### 2.1 Primary Functions

1. **Aggregate Critic Scores** - Compute weighted composite scores from all 5 critics
2. **Detect Critic Conflicts** - Identify disagreements between critics (e.g., high Logic but low Grounding)
3. **Dynamic Weight Adjustment** - Modify critic weights based on context (scientific claims need higher Grounding weight)
4. **Meta-Level Reasoning** - Produce higher-order assessments ("this claim is well-structured but weakly grounded")
5. **Improvement Suggestions** - Synthesize actionable feedback from critic issues

### 2.2 Non-Responsibilities

- Does NOT replace individual critics
- Does NOT override critic scores
- Does NOT generate new claims (that's Proposer/Synthesizer's job)

---

## 3. Architecture

### 3.1 Module Structure

```
lib/cns/critics/
├── critic.ex           # Behaviour (existing)
├── grounding.ex        # (existing)
├── logic.ex            # (existing)
├── novelty.ex          # (existing)
├── bias.ex             # (existing)
├── causal.ex           # (existing)
└── meta.ex             # NEW - Meta-Critic
```

### 3.2 Behaviour Implementation

The Meta-Critic implements `CNS.Critics.Critic` behaviour but has a distinct evaluation signature that accepts all critic results:

```elixir
defmodule CNS.Critics.Meta do
  @behaviour CNS.Critics.Critic

  use GenServer

  alias CNS.{SNO, Critics}
  alias CNS.Critics.{Grounding, Logic, Novelty, Bias, Causal}

  @critics [Grounding, Logic, Novelty, Bias, Causal]

  # Standard behaviour callbacks
  @impl true
  def name, do: :meta

  @impl true
  def weight, do: 1.0  # Meta uses all critic weights internally

  @impl true
  def evaluate(%SNO{} = sno) do
    # Orchestrate all critics and aggregate
    with {:ok, results} <- evaluate_all_critics(sno),
         {:ok, aggregate} <- aggregate_results(results),
         {:ok, conflicts} <- detect_conflicts(results),
         {:ok, suggestions} <- generate_suggestions(results, conflicts) do
      {:ok, %{
        score: aggregate.weighted_score,
        issues: aggregate.all_issues,
        details: %{
          critic_scores: aggregate.critic_scores,
          conflicts: conflicts,
          suggestions: suggestions,
          confidence_band: compute_confidence_band(results),
          dominant_critic: find_dominant_critic(results),
          weakest_dimension: find_weakest_dimension(results)
        }
      }}
    end
  end
end
```

---

## 4. Core Algorithms

### 4.1 Weighted Score Aggregation

```elixir
@spec aggregate_results(map()) :: {:ok, map()}
defp aggregate_results(results) do
  total_weight = @critics |> Enum.map(& &1.weight()) |> Enum.sum()

  weighted_score =
    results
    |> Enum.map(fn {critic, %{score: score}} ->
      critic.weight() * score
    end)
    |> Enum.sum()
    |> Kernel./(total_weight)
    |> Float.round(4)

  all_issues =
    results
    |> Enum.flat_map(fn {_critic, %{issues: issues}} -> issues end)

  critic_scores =
    results
    |> Enum.map(fn {critic, %{score: score}} ->
      {critic.name(), score}
    end)
    |> Enum.into(%{})

  {:ok, %{
    weighted_score: weighted_score,
    all_issues: all_issues,
    critic_scores: critic_scores
  }}
end
```

### 4.2 Conflict Detection

Conflicts occur when critics produce divergent scores (threshold: 0.3 delta):

```elixir
@conflict_threshold 0.3

@spec detect_conflicts(map()) :: {:ok, [conflict()]}
defp detect_conflicts(results) do
  scores = Enum.map(results, fn {critic, %{score: s}} -> {critic.name(), s} end)

  conflicts =
    for {name1, score1} <- scores,
        {name2, score2} <- scores,
        name1 < name2,  # Avoid duplicates
        abs(score1 - score2) > @conflict_threshold do
      %{
        critics: {name1, name2},
        delta: Float.round(abs(score1 - score2), 3),
        interpretation: interpret_conflict(name1, score1, name2, score2)
      }
    end

  {:ok, conflicts}
end

defp interpret_conflict(:logic, logic_score, :grounding, grounding_score)
     when logic_score > grounding_score do
  "Claim is logically coherent but lacks factual grounding - needs evidence"
end

defp interpret_conflict(:grounding, grounding_score, :novelty, novelty_score)
     when grounding_score > novelty_score do
  "Well-grounded but derivative - may be restating known facts"
end

defp interpret_conflict(:logic, logic_score, :causal, causal_score)
     when logic_score > causal_score do
  "Logical structure sound but causal claims unsubstantiated"
end

# ... other conflict interpretations
```

### 4.3 Dynamic Weight Adjustment

Context-sensitive weight modification:

```elixir
@type context :: :scientific | :philosophical | :empirical | :analytical | :default

@spec adjust_weights(context()) :: keyword()
def adjust_weights(:scientific) do
  # Scientific claims need strong grounding
  [grounding: 0.50, logic: 0.25, causal: 0.15, novelty: 0.05, bias: 0.05]
end

def adjust_weights(:philosophical) do
  # Philosophical claims prioritize logic
  [grounding: 0.20, logic: 0.45, causal: 0.10, novelty: 0.15, bias: 0.10]
end

def adjust_weights(:empirical) do
  # Empirical claims need grounding + causal
  [grounding: 0.40, logic: 0.20, causal: 0.25, novelty: 0.10, bias: 0.05]
end

def adjust_weights(:default) do
  # Use standard weights from each critic
  @critics |> Enum.map(fn c -> {c.name(), c.weight()} end)
end
```

### 4.4 Suggestion Generation

```elixir
@spec generate_suggestions(map(), [conflict()]) :: {:ok, [String.t()]}
defp generate_suggestions(results, conflicts) do
  issue_based =
    results
    |> Enum.flat_map(fn {critic, %{issues: issues, score: score}} ->
      if score < 0.6 do
        suggest_for_critic(critic.name(), issues)
      else
        []
      end
    end)

  conflict_based =
    conflicts
    |> Enum.map(fn %{interpretation: interp} ->
      "Address: #{interp}"
    end)

  {:ok, Enum.uniq(issue_based ++ conflict_based)}
end

defp suggest_for_critic(:grounding, issues) do
  Enum.map(issues, fn
    "no_evidence" <> _ -> "Add supporting evidence with citations"
    "low_validity" <> _ -> "Strengthen evidence quality"
    "low_relevance" <> _ -> "Ensure evidence directly supports claim"
    _ -> nil
  end)
  |> Enum.reject(&is_nil/1)
end

defp suggest_for_critic(:logic, issues) do
  Enum.map(issues, fn
    "circular_reasoning" <> _ -> "Break circular dependencies in argument"
    "contradiction" <> _ -> "Resolve contradictory statements"
    "weak_entailment" <> _ -> "Strengthen logical connection to evidence"
    _ -> nil
  end)
  |> Enum.reject(&is_nil/1)
end

# ... other critic suggestions
```

---

## 5. API Specification

### 5.1 Public Functions

```elixir
# Start Meta-Critic GenServer
@spec start_link(keyword()) :: GenServer.on_start()
def start_link(opts \\ [])

# Evaluate SNO through all critics
@spec evaluate(SNO.t()) :: {:ok, evaluation_result()} | {:error, term()}
def evaluate(%SNO{} = sno)

# Evaluate with context-adjusted weights
@spec evaluate(SNO.t(), context()) :: {:ok, evaluation_result()} | {:error, term()}
def evaluate(%SNO{} = sno, context)

# Get all critic scores without aggregation
@spec critic_breakdown(SNO.t()) :: {:ok, map()}
def critic_breakdown(%SNO{} = sno)

# Get improvement roadmap
@spec improvement_plan(SNO.t()) :: {:ok, [step()]}
def improvement_plan(%SNO{} = sno)

# Check if SNO meets quality gate
@spec passes_quality_gate?(SNO.t(), keyword()) :: boolean()
def passes_quality_gate?(%SNO{} = sno, opts \\ [])
```

### 5.2 Types

```elixir
@type evaluation_result :: %{
  score: float(),
  issues: [String.t()],
  details: %{
    critic_scores: %{atom() => float()},
    conflicts: [conflict()],
    suggestions: [String.t()],
    confidence_band: {float(), float()},
    dominant_critic: atom(),
    weakest_dimension: atom()
  }
}

@type conflict :: %{
  critics: {atom(), atom()},
  delta: float(),
  interpretation: String.t()
}

@type step :: %{
  priority: :high | :medium | :low,
  critic: atom(),
  action: String.t(),
  expected_impact: float()
}
```

---

## 6. Integration Points

### 6.1 Pipeline Integration

The Meta-Critic integrates into `CNS.Agents.Pipeline`:

```elixir
# In pipeline.ex
defp step_evaluate(%{synthesis: synthesis} = state) do
  with {:ok, meta_result} <- Meta.evaluate(synthesis) do
    {:ok, %{state |
      metrics: Map.put(state.metrics, :meta_evaluation, meta_result),
      quality_score: meta_result.score
    }}
  end
end
```

### 6.2 Quality Gate

```elixir
# Before convergence check
defp check_convergence(%{synthesis: synthesis, config: config} = state) do
  with {:ok, meta_result} <- Meta.evaluate(synthesis),
       true <- meta_result.score >= config.quality_threshold do
    converged = converged?(synthesis, config)
    {:ok, %{state | converged: converged, quality_validated: true}}
  else
    false -> {:ok, %{state | converged: false, quality_validated: false}}
    error -> error
  end
end
```

### 6.3 UI Integration

For `cns_ui` LiveView:

```elixir
# In dialectical_live.ex
def handle_info({:meta_evaluation, result}, socket) do
  {:noreply, assign(socket,
    meta_score: result.score,
    critic_breakdown: result.details.critic_scores,
    conflicts: result.details.conflicts,
    suggestions: result.details.suggestions
  )}
end
```

---

## 7. Configuration

### 7.1 Config Options

```elixir
# In CNS.Config
defstruct [
  # ... existing fields ...
  meta_config: %{
    conflict_threshold: 0.3,
    quality_gate_threshold: 0.7,
    context: :default,
    parallel_evaluation: true,
    cache_results: false
  }
]
```

### 7.2 Runtime Configuration

```elixir
# Application config
config :cns, CNS.Critics.Meta,
  conflict_threshold: 0.3,
  quality_gate: 0.7,
  timeout: 30_000,
  cache_ttl: :timer.minutes(5)
```

---

## 8. Testing Strategy

### 8.1 Unit Tests

```elixir
defmodule CNS.Critics.MetaTest do
  use ExUnit.Case, async: true

  alias CNS.{SNO, Evidence}
  alias CNS.Critics.Meta

  describe "evaluate/1" do
    test "aggregates all critic scores" do
      sno = SNO.new("Well-grounded claim",
        evidence: [Evidence.new("Source", "Data", validity: 0.9)],
        confidence: 0.85
      )

      {:ok, result} = Meta.evaluate(sno)

      assert result.score >= 0.0 and result.score <= 1.0
      assert map_size(result.details.critic_scores) == 5
      assert Enum.all?([:grounding, :logic, :novelty, :bias, :causal],
        &Map.has_key?(result.details.critic_scores, &1))
    end

    test "detects conflicts between critics" do
      # Claim that's logically sound but poorly grounded
      sno = SNO.new("If A then B, and A, therefore B",
        evidence: [],
        confidence: 0.7
      )

      {:ok, result} = Meta.evaluate(sno)

      assert length(result.details.conflicts) > 0
      assert Enum.any?(result.details.conflicts, fn c ->
        c.critics == {:grounding, :logic} or c.critics == {:logic, :grounding}
      end)
    end

    test "generates actionable suggestions" do
      sno = SNO.new("Claim with issues",
        evidence: [],
        confidence: 0.5
      )

      {:ok, result} = Meta.evaluate(sno)

      assert length(result.details.suggestions) > 0
      assert Enum.any?(result.details.suggestions, &String.contains?(&1, "evidence"))
    end
  end

  describe "evaluate/2 with context" do
    test "adjusts weights for scientific context" do
      sno = SNO.new("Scientific claim",
        evidence: [Evidence.new("Study", "p < 0.05", validity: 0.95)],
        confidence: 0.8
      )

      {:ok, default_result} = Meta.evaluate(sno, :default)
      {:ok, scientific_result} = Meta.evaluate(sno, :scientific)

      # Scientific context should weight grounding higher
      assert scientific_result.details.critic_scores[:grounding] * 0.5 >
             default_result.details.critic_scores[:grounding] * 0.4
    end
  end

  describe "passes_quality_gate?/2" do
    test "returns true for high-quality SNO" do
      sno = SNO.new("High quality claim",
        evidence: [
          Evidence.new("Source1", "Data1", validity: 0.9),
          Evidence.new("Source2", "Data2", validity: 0.85)
        ],
        confidence: 0.9
      )

      assert Meta.passes_quality_gate?(sno)
    end

    test "returns false for low-quality SNO" do
      sno = SNO.new("Poor claim", evidence: [], confidence: 0.3)

      refute Meta.passes_quality_gate?(sno)
    end
  end

  describe "improvement_plan/1" do
    test "returns prioritized improvement steps" do
      sno = SNO.new("Claim needing improvement",
        evidence: [Evidence.new("Weak", "data", validity: 0.4)],
        confidence: 0.6
      )

      {:ok, plan} = Meta.improvement_plan(sno)

      assert is_list(plan)
      assert Enum.all?(plan, &match?(%{priority: _, critic: _, action: _}, &1))
      # High priority items should come first
      priorities = Enum.map(plan, & &1.priority)
      assert priorities == Enum.sort(priorities, &priority_order/2)
    end
  end
end
```

### 8.2 Integration Tests

```elixir
defmodule CNS.Critics.MetaIntegrationTest do
  use ExUnit.Case

  alias CNS.Agents.Pipeline
  alias CNS.Critics.Meta
  alias CNS.Config

  @tag :integration
  test "meta-critic integrates with pipeline" do
    config = Config.new(max_iterations: 3)

    {:ok, result} = Pipeline.run("Does exercise improve cognitive function?", config)

    # Meta evaluation should be in metrics
    assert Map.has_key?(result.metrics, :meta_evaluation) or
           result.metrics[:quality_score] != nil
  end

  @tag :integration
  test "quality gate blocks low-quality synthesis" do
    config = Config.new(
      max_iterations: 1,
      meta_config: %{quality_gate_threshold: 0.95}  # Very high bar
    )

    {:ok, result} = Pipeline.run("Vague unsupported claim", config)

    # Should not converge due to quality gate
    refute result.converged
  end
end
```

---

## 9. Performance Considerations

### 9.1 Parallel Evaluation

```elixir
defp evaluate_all_critics(sno) do
  tasks = Enum.map(@critics, fn critic ->
    Task.async(fn -> {critic, critic.evaluate(sno)} end)
  end)

  results =
    tasks
    |> Task.await_many(5_000)
    |> Enum.map(fn {critic, {:ok, result}} -> {critic, result} end)
    |> Enum.into(%{})

  {:ok, results}
end
```

### 9.2 Caching

```elixir
# For repeated evaluations of same SNO
defp cached_evaluate(sno, cache) do
  cache_key = :crypto.hash(:sha256, :erlang.term_to_binary(sno))

  case :ets.lookup(cache, cache_key) do
    [{^cache_key, result, timestamp}] when timestamp > now - @ttl ->
      {:ok, result}
    _ ->
      {:ok, result} = do_evaluate(sno)
      :ets.insert(cache, {cache_key, result, now()})
      {:ok, result}
  end
end
```

---

## 10. Implementation Checklist

- [ ] Create `lib/cns/critics/meta.ex` module
- [ ] Implement `@behaviour CNS.Critics.Critic`
- [ ] Implement `evaluate_all_critics/1` with Task.async
- [ ] Implement `aggregate_results/1` with weighted scoring
- [ ] Implement `detect_conflicts/1` with threshold logic
- [ ] Implement `interpret_conflict/4` for all critic pairs
- [ ] Implement `generate_suggestions/2`
- [ ] Implement `adjust_weights/1` for all contexts
- [ ] Implement `improvement_plan/1`
- [ ] Implement `passes_quality_gate?/2`
- [ ] Add GenServer state tracking for statistics
- [ ] Create `test/cns/critics/meta_test.exs`
- [ ] Add integration with `CNS.Agents.Pipeline`
- [ ] Add configuration options to `CNS.Config`
- [ ] Update `cns_ui` for meta-critic visualization
- [ ] Write documentation and typespecs
- [ ] Run `mix test`, `mix credo --strict`, `mix dialyzer`

---

## 11. Related Documents

- [CNS Architecture Overview](../architecture.md)
- [Critic Behaviour Specification](../critics/README.md)
- [Pipeline Orchestration](../agents/pipeline.md)
- [Topology Metrics](../topology.md)

---

## Appendix A: Critic Conflict Matrix

| Critic 1 | Critic 2 | High-Low Interpretation |
|----------|----------|-------------------------|
| Logic | Grounding | Coherent but ungrounded - needs evidence |
| Grounding | Logic | Well-evidenced but reasoning flawed |
| Grounding | Novelty | Derivative - restating known facts |
| Novelty | Grounding | Original but speculative |
| Logic | Causal | Logical but causal claims unsubstantiated |
| Causal | Logic | Causal validity but logical gaps |
| Bias | Grounding | Biased framing despite evidence |
| Grounding | Bias | Neutral framing but weak evidence |
| Logic | Bias | Sound logic but biased presentation |
| Novelty | Bias | Novel perspective but potentially biased |

---

## Appendix B: Quality Gate Thresholds

| Quality Level | Score Range | Action |
|---------------|-------------|--------|
| Excellent | 0.85 - 1.00 | Accept, minimal review |
| Good | 0.70 - 0.84 | Accept, note suggestions |
| Acceptable | 0.50 - 0.69 | Accept with caveats |
| Poor | 0.30 - 0.49 | Reject, require improvements |
| Unacceptable | 0.00 - 0.29 | Reject outright |

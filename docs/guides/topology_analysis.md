# Topology Analysis Guide

## Overview

The `CNS.Logic.Betti` module computes topological invariants on claim-relation graphs to detect structural issues like circular reasoning and conflicting evidence.

## Core Concepts

### Betti Numbers

Betti numbers are topological invariants that characterize the shape of a graph:

- **β₀ (beta0)**: Number of connected components
- **β₁ (beta1)**: Number of independent cycles

For reasoning graphs:
- β₁ > 0 indicates circular reasoning patterns
- β₀ > 1 indicates disconnected argument clusters

### Formula

```
β₁ = edges - vertices + components
```

## Basic Usage

### Computing Graph Statistics

```elixir
alias CNS.Logic.Betti

claim_ids = ["c1", "c2", "c3"]
relations = [
  {"c2", "supports", "c1"},
  {"c3", "supports", "c1"}
]

stats = Betti.compute_graph_stats(claim_ids, relations)

# Result: %Betti.GraphStats{
#   nodes: 3,
#   edges: 2,
#   components: 1,
#   beta1: 0,
#   cycles: [],
#   polarity_conflict: false
# }
```

### Detecting Circular Reasoning

```elixir
# Graph with a cycle: c1 -> c2 -> c3 -> c1
claim_ids = ["c1", "c2", "c3"]
relations = [
  {"c1", "supports", "c2"},
  {"c2", "supports", "c3"},
  {"c3", "supports", "c1"}
]

stats = Betti.compute_graph_stats(claim_ids, relations)

stats.beta1
# 1 (one independent cycle)

stats.cycles
# [["c1", "c2", "c3"]] (the cycle path)
```

### Detecting Polarity Conflicts

A polarity conflict occurs when a claim receives both `supports` and `refutes` edges:

```elixir
claim_ids = ["c1", "c2", "c3"]
relations = [
  {"c2", "supports", "c1"},
  {"c3", "refutes", "c1"}
]

stats = Betti.compute_graph_stats(claim_ids, relations)

stats.polarity_conflict
# true
```

## The GraphStats Struct

```elixir
%CNS.Logic.Betti.GraphStats{
  nodes: non_neg_integer(),          # Number of claims
  edges: non_neg_integer(),          # Number of relations
  components: non_neg_integer(),     # Connected components (β₀)
  beta1: non_neg_integer(),          # Independent cycles
  cycles: [[String.t()]],            # Detected cycle paths
  polarity_conflict: boolean()       # Both supports and refutes to same target
}
```

## Individual Functions

### Polarity Conflict Detection

```elixir
relations = [
  {"c2", "supports", "c1"},
  {"c3", "refutes", "c1"}
]

# Check specific target
Betti.polarity_conflict?(relations, "c1")
# true

# Default target is "c1"
Betti.polarity_conflict?(relations)
# true
```

### Cycle Detection

```elixir
# Build graph manually
graph = Graph.new(type: :directed)
  |> Graph.add_vertices(["c1", "c2", "c3"])
  |> Graph.add_edge("c1", "c2")
  |> Graph.add_edge("c2", "c3")
  |> Graph.add_edge("c3", "c1")

cycles = Betti.find_cycles(graph)
# [["c1", "c2", "c3"]]
```

## Interpretation Guidelines

### Healthy Reasoning Graph

```elixir
%GraphStats{
  beta1: 0,              # No cycles - linear reasoning
  components: 1,         # Single connected argument
  polarity_conflict: false  # No contradictory evidence
}
```

### Warning Signs

| Metric | Value | Indicates |
|--------|-------|-----------|
| beta1 | > 0 | Circular reasoning, claims depend on themselves |
| components | > 1 | Disconnected arguments, missing links |
| polarity_conflict | true | Contradictory evidence needs resolution |

### Severity Assessment

```elixir
def assess_severity(stats) do
  cond do
    stats.beta1 > 2 -> :high    # Multiple circular patterns
    stats.polarity_conflict and stats.beta1 > 0 -> :high
    stats.beta1 > 0 -> :medium  # Some circular reasoning
    stats.polarity_conflict -> :medium
    stats.components > 2 -> :low
    true -> :none
  end
end
```

## Case-Insensitive Matching

IDs are normalized to lowercase:

```elixir
claim_ids = ["C1", "c2"]
relations = [{"C2", "supports", "c1"}]

stats = Betti.compute_graph_stats(claim_ids, relations)
stats.edges
# 1 (match found despite case difference)
```

## Complete Analysis Example

```elixir
defmodule MyApp.TopologyChecker do
  alias CNS.Schema.Parser
  alias CNS.Logic.Betti

  def check(llm_output) do
    {claims, relations} = Parser.parse(llm_output)
    claim_ids = Map.keys(claims)
    stats = Betti.compute_graph_stats(claim_ids, relations)

    issues = []

    issues = if stats.beta1 > 0 do
      cycles = stats.cycles
        |> Enum.map(&Enum.join(&1, " -> "))
        |> Enum.join("; ")
      [{:circular_reasoning, "Cycles detected: #{cycles}"} | issues]
    else
      issues
    end

    issues = if stats.polarity_conflict do
      [{:polarity_conflict, "Both supports and refutes for same claim"} | issues]
    else
      issues
    end

    issues = if stats.components > 1 do
      [{:disconnected, "#{stats.components} disconnected components"} | issues]
    else
      issues
    end

    %{
      valid: issues == [],
      issues: issues,
      stats: stats
    }
  end
end
```

## Performance Considerations

- Cycle detection is limited to 100 cycles to prevent combinatorial explosion
- For large graphs (1000+ nodes), consider sampling or pruning
- Connected component detection uses BFS, O(V + E) complexity

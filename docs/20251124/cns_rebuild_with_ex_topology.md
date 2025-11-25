# CNS Rebuild Plan: ex_topology Integration

**Date**: 2024-11-24
**Status**: Proposal
**Dependency**: ex_topology v0.1.1 (hex.pm)

---

## Executive Summary

This document proposes rebuilding CNS (Chiral Narrative Synthesis) with **ex_topology** as the primary topological analysis dependency. The integration replaces CNS's current surrogate-based topology approximations with a full persistent homology pipeline while preserving the dialectical reasoning architecture.

**Key Benefits**:
- Replace stub TDA implementation with production-grade persistent homology
- Unify graph topology, embedding analysis, and statistical validation
- Maintain compatibility (shared libgraph, Nx dependencies)
- Enable research-grade topological analysis of claim networks

---

## Current State Analysis

### CNS Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    CNS Pipeline                              │
├─────────────────────────────────────────────────────────────┤
│  Proposer ──► Antagonist ──► Synthesizer ──► Validation     │
│     │              │              │              │           │
│     └──────────────┴──────────────┴──────────────┘           │
│                         │                                    │
│              ┌──────────┴──────────┐                        │
│              │  Topology Analysis  │                        │
│              │  (Surrogates only)  │ ◄── Replace with       │
│              └─────────────────────┘     ex_topology        │
└─────────────────────────────────────────────────────────────┘
```

### Current Topology Modules (~2000 LOC)

| Module | LOC | Function | Status |
|--------|-----|----------|--------|
| `CNS.Topology` | 443 | Graph building, Betti numbers, DAG validation | Working |
| `CNS.Topology.Surrogates` | 493 | β₁ surrogate, fragility surrogate | Working |
| `CNS.Topology.TDA` | 170 | Persistent homology stub | **Stub only** |
| `CNS.Logic.Betti` | ~100 | Cycle detection, polarity conflict | Working |
| `CNS.Graph.*` | ~400 | Builder, traversal, visualization | Working |

### Key Limitations

1. **No real persistent homology** - TDA module returns cyclomatic approximation
2. **No persistence diagrams** - Stubs return empty structures
3. **No higher Betti numbers** - Only β₀, β₁ supported
4. **No filtration** - No Vietoris-Rips or graph filtrations
5. **Basic fragility** - Simple k-NN variance only

---

## ex_topology Capabilities (v0.1.1)

| Module | Purpose | Replaces CNS Module |
|--------|---------|---------------------|
| `ExTopology.Graph` | β₀, β₁, Euler characteristic | `CNS.Topology`, `CNS.Logic.Betti` |
| `ExTopology.Distance` | Pairwise distance matrices | Manual calculations |
| `ExTopology.Neighborhood` | k-NN, ε-ball, Gabriel graphs | `CNS.Graph.Builder` (partial) |
| `ExTopology.Embedding` | k-NN variance, density, isolation | `CNS.Topology.Surrogates` |
| `ExTopology.Statistics` | Correlation, effect size | `CNS.Metrics` (partial) |
| `ExTopology.Simplex` | Simplicial complexes, boundary | **New capability** |
| `ExTopology.Filtration` | Vietoris-Rips, graph filtrations | **New capability** |
| `ExTopology.Persistence` | Persistent homology computation | `CNS.Topology.TDA` |
| `ExTopology.Diagram` | Diagram analysis, distances | **New capability** |
| `ExTopology.Fragility` | Topological stability analysis | `CNS.Topology.Surrogates` |

---

## Proposed Architecture

### New Module Structure

```
lib/cns/
├── core/                          # Preserve existing
│   ├── sno.ex                     # Structured Narrative Object
│   ├── evidence.ex                # Evidence records
│   ├── challenge.ex               # Antagonist challenges
│   └── provenance.ex              # Provenance tracking
│
├── agents/                        # Preserve existing
│   ├── proposer.ex
│   ├── antagonist.ex
│   ├── synthesizer.ex
│   └── pipeline.ex
│
├── topology/                      # REFACTOR: ex_topology wrappers
│   ├── adapter.ex                 # ex_topology integration layer
│   ├── graph_analysis.ex          # Graph topology (via ex_topology)
│   ├── persistence.ex             # Persistent homology (via ex_topology)
│   ├── fragility.ex               # Stability analysis (via ex_topology)
│   └── claim_network.ex           # CNS-specific claim network analysis
│
├── embedding/                     # NEW: Unified embedding pipeline
│   ├── encoder.ex                 # MiniLM embedding generation
│   ├── analysis.ex                # Embedding quality (via ex_topology)
│   └── cache.ex                   # Embedding caching
│
├── metrics/                       # ENHANCE: Use ex_topology.Statistics
│   ├── quality.ex                 # Quality scoring
│   ├── chirality.ex               # Fisher-Rao distance
│   └── statistical.ex             # Statistical validation (via ex_topology)
│
├── validation/                    # Preserve + enhance
│   ├── semantic.ex
│   ├── citation.ex
│   └── topological.ex             # NEW: Topological validation
│
└── critics/                       # Preserve existing
    ├── grounding.ex
    ├── causal.ex
    ├── logic.ex
    ├── bias.ex
    └── novelty.ex
```

### Integration Layer Design

```elixir
# lib/cns/topology/adapter.ex
defmodule CNS.Topology.Adapter do
  @moduledoc """
  Bridge between CNS claim networks and ex_topology.

  Converts CNS data structures (SNOs, Evidence) to formats
  suitable for topological analysis, then interprets results
  in the dialectical reasoning context.
  """

  alias ExTopology.{Distance, Neighborhood, Embedding, Statistics}
  alias ExTopology.{Simplex, Filtration, Persistence, Diagram, Fragility}
  alias ExTopology.Graph, as: Topo

  @doc "Convert SNO list to point cloud (embeddings)"
  def sno_embeddings(snos, opts \\ [])

  @doc "Build neighborhood graph from claim embeddings"
  def claim_graph(embeddings, strategy \\ :knn, opts \\ [])

  @doc "Compute full persistent homology on claim network"
  def persistent_homology(embeddings, opts \\ [])

  @doc "Analyze topological fragility of claim network"
  def fragility_analysis(embeddings, opts \\ [])

  @doc "Compute diagram distances between two claim networks"
  def diagram_distance(snos1, snos2, opts \\ [])
end
```

---

## Implementation Phases

### Phase 1: Dependency & Foundation (Week 1)

**Goal**: Add ex_topology, create adapter layer, validate compatibility.

#### Tasks

1. **Update mix.exs**
   ```elixir
   defp deps do
     [
       {:ex_topology, "~> 0.1.1"},
       # ... existing deps
     ]
   end
   ```

2. **Create CNS.Topology.Adapter**
   - SNO → Nx.Tensor conversion
   - Embedding extraction pipeline
   - Result interpretation helpers

3. **Add equivalence tests**
   - Verify `ExTopology.Graph.beta_one/1` matches `CNS.Topology.betti_numbers/1`
   - Compare cycle detection results
   - Validate component counting

4. **Deliverables**
   - [ ] ex_topology added to deps
   - [ ] Adapter module with SNO conversion
   - [ ] Test suite comparing old vs. new implementations
   - [ ] No regressions in existing tests

---

### Phase 2: Graph Topology Migration (Week 2)

**Goal**: Replace manual Betti number calculations with ex_topology.

#### Module Changes

| Old Module | New Implementation |
|------------|-------------------|
| `CNS.Topology.betti_numbers/1` | `Topo.invariants(graph)` |
| `CNS.Topology.detect_cycles/1` | `Topo.beta_one(graph) > 0` + DFS |
| `CNS.Topology.is_dag?/1` | `Topo.beta_one(directed) == 0` |
| `CNS.Logic.Betti.compute_graph_stats/2` | `Topo.invariants(graph)` |

#### New Capabilities

```elixir
# lib/cns/topology/graph_analysis.ex
defmodule CNS.Topology.GraphAnalysis do
  alias ExTopology.Graph, as: Topo

  @doc "Full topological invariants for claim graph"
  def invariants(claim_graph) do
    inv = Topo.invariants(claim_graph)

    %{
      components: inv.beta_zero,
      cycles: inv.beta_one,
      euler_characteristic: inv.euler_characteristic,
      is_tree: Topo.tree?(claim_graph),
      is_forest: Topo.forest?(claim_graph),
      # CNS-specific interpretation
      has_circular_reasoning: inv.beta_one > 0,
      claim_clusters: inv.beta_zero
    }
  end

  @doc "Detect polarity conflicts via graph analysis"
  def polarity_conflicts(claim_graph, evidence_polarities) do
    # Use ex_topology for structure, CNS logic for interpretation
    ...
  end
end
```

#### Deliverables
- [ ] `CNS.Topology.GraphAnalysis` module
- [ ] Deprecation warnings on old functions
- [ ] Updated tests using ex_topology
- [ ] Performance benchmarks (old vs. new)

---

### Phase 3: Embedding Analysis Enhancement (Week 3)

**Goal**: Replace fragility surrogates with ex_topology.Embedding.

#### Current Surrogates → ex_topology

| CNS Surrogate | ex_topology Replacement |
|---------------|------------------------|
| `compute_fragility_surrogate/2` | `Embedding.knn_variance/2` |
| Manual k-NN variance | `Embedding.statistics/2` |
| — | `Embedding.isolation_scores/2` (NEW) |
| — | `Embedding.local_density/2` (NEW) |
| — | `Embedding.sparse_points/2` (NEW) |

#### New Module

```elixir
# lib/cns/embedding/analysis.ex
defmodule CNS.Embedding.Analysis do
  alias ExTopology.Embedding

  @doc "Comprehensive embedding quality analysis for claim network"
  def analyze(embeddings, opts \\ []) do
    k = Keyword.get(opts, :k, 5)

    %{
      knn_variance: Embedding.knn_variance(embeddings, k: k),
      local_density: Embedding.local_density(embeddings, k: k),
      isolation_scores: Embedding.isolation_scores(embeddings, k: k),
      sparse_points: Embedding.sparse_points(embeddings, k: k, percentile: 10),
      statistics: Embedding.statistics(embeddings, k: k)
    }
  end

  @doc "Identify semantically isolated claims (potential outliers)"
  def isolated_claims(snos, opts \\ []) do
    embeddings = CNS.Topology.Adapter.sno_embeddings(snos)
    scores = Embedding.isolation_scores(embeddings, k: Keyword.get(opts, :k, 3))

    snos
    |> Enum.zip(Nx.to_flat_list(scores))
    |> Enum.filter(fn {_, score} -> score > Keyword.get(opts, :threshold, 2.0) end)
    |> Enum.map(fn {sno, score} -> {sno, score} end)
  end
end
```

#### Deliverables
- [ ] `CNS.Embedding.Analysis` module
- [ ] Integration with MiniLM encoder
- [ ] Isolated claim detection
- [ ] Updated fragility metrics

---

### Phase 4: Full Persistent Homology (Week 4)

**Goal**: Replace TDA stub with real persistent homology pipeline.

#### TDA Pipeline Implementation

```elixir
# lib/cns/topology/persistence.ex
defmodule CNS.Topology.Persistence do
  alias ExTopology.{Filtration, Persistence, Diagram}

  @doc """
  Compute persistent homology for claim network.

  Returns persistence diagrams for H₀ (claim clusters),
  H₁ (circular reasoning), and H₂ (higher-order structures).
  """
  def compute(snos, opts \\ []) do
    embeddings = CNS.Topology.Adapter.sno_embeddings(snos, opts)
    max_dim = Keyword.get(opts, :max_dimension, 2)

    # Build Vietoris-Rips filtration
    filtration = Filtration.vietoris_rips(embeddings,
      max_dimension: max_dim,
      max_epsilon: Keyword.get(opts, :max_epsilon, 2.0)
    )

    # Compute persistence
    diagrams = Persistence.compute(filtration, max_dimension: max_dim)

    # Interpret in CNS context
    interpret_diagrams(diagrams, snos)
  end

  defp interpret_diagrams(diagrams, snos) do
    h0 = Enum.find(diagrams, & &1.dimension == 0)
    h1 = Enum.find(diagrams, & &1.dimension == 1)
    h2 = Enum.find(diagrams, & &1.dimension == 2)

    %{
      # H₀: Claim clusters
      cluster_analysis: %{
        total_clusters: length(h0.pairs),
        persistent_clusters: Diagram.filter_by_persistence(h0, min: 0.5) |> Map.get(:pairs) |> length(),
        cluster_stability: Diagram.total_persistence(h0)
      },

      # H₁: Circular reasoning detection
      circular_reasoning: %{
        detected_cycles: length(h1.pairs),
        persistent_cycles: Diagram.filter_by_persistence(h1, min: 0.3) |> Map.get(:pairs) |> length(),
        cycle_severity: Diagram.max_persistence(h1)
      },

      # H₂: Higher-order structures
      higher_order: %{
        voids: length(h2.pairs),
        complexity: Diagram.entropy(h2)
      },

      # Raw diagrams for further analysis
      diagrams: diagrams
    }
  end

  @doc "Compare topological structure of two claim networks"
  def compare(snos1, snos2, opts \\ []) do
    result1 = compute(snos1, opts)
    result2 = compute(snos2, opts)

    # Compute bottleneck distances per dimension
    distances =
      Enum.zip(result1.diagrams, result2.diagrams)
      |> Enum.map(fn {d1, d2} ->
        %{
          dimension: d1.dimension,
          bottleneck: Diagram.bottleneck_distance(d1, d2),
          wasserstein: Diagram.wasserstein_distance(d1, d2, p: 2)
        }
      end)

    %{
      distances: distances,
      total_distance: Enum.sum(Enum.map(distances, & &1.bottleneck)),
      topologically_similar?: Enum.all?(distances, & &1.bottleneck < 0.5)
    }
  end
end
```

#### Deliverables
- [ ] `CNS.Topology.Persistence` module
- [ ] Real persistence diagram computation
- [ ] Circular reasoning detection via H₁
- [ ] Claim network comparison via bottleneck distance
- [ ] Integration tests with synthetic claim networks

---

### Phase 5: Topological Fragility (Week 5)

**Goal**: Implement advanced stability analysis using ex_topology.Fragility.

#### Fragility Analysis Module

```elixir
# lib/cns/topology/fragility.ex
defmodule CNS.Topology.Fragility do
  alias ExTopology.Fragility

  @doc """
  Analyze topological fragility of claim network.

  Identifies critical claims whose removal would significantly
  change the network's topological structure.
  """
  def analyze(snos, opts \\ []) do
    embeddings = CNS.Topology.Adapter.sno_embeddings(snos, opts)

    # Point removal sensitivity
    removal_scores = Fragility.point_removal_sensitivity(embeddings,
      max_dimension: Keyword.get(opts, :max_dimension, 1)
    )

    # Identify critical points
    critical_indices = Fragility.identify_critical_points(removal_scores,
      top_k: Keyword.get(opts, :top_k, 5)
    )

    # Overall robustness
    robustness = Fragility.robustness_score(embeddings)

    %{
      robustness_score: robustness,
      interpretation: interpret_robustness(robustness),
      critical_claims: Enum.map(critical_indices, & Enum.at(snos, &1)),
      removal_sensitivity: Enum.zip(snos, Map.values(removal_scores)),
      bottleneck_stability: Fragility.bottleneck_stability(embeddings)
    }
  end

  defp interpret_robustness(score) when score > 0.7, do: :highly_robust
  defp interpret_robustness(score) when score > 0.4, do: :moderately_robust
  defp interpret_robustness(_), do: :fragile

  @doc "Analyze local fragility around a specific claim"
  def local_analysis(snos, claim_index, opts \\ []) do
    embeddings = CNS.Topology.Adapter.sno_embeddings(snos, opts)

    local = Fragility.local_fragility(embeddings, claim_index,
      k: Keyword.get(opts, :k, 3)
    )

    %{
      claim: Enum.at(snos, claim_index),
      removal_impact: local.removal_impact,
      neighborhood_fragility: local.neighborhood_mean_fragility,
      relative_fragility: local.relative_fragility,
      neighbors: Enum.map(local.neighbor_indices, & Enum.at(snos, &1))
    }
  end
end
```

#### Deliverables
- [ ] `CNS.Topology.Fragility` module
- [ ] Critical claim identification
- [ ] Robustness scoring for claim networks
- [ ] Local fragility analysis per claim

---

### Phase 6: Statistical Validation Enhancement (Week 6)

**Goal**: Enhance metrics with ex_topology.Statistics.

#### Statistical Enhancements

```elixir
# lib/cns/metrics/statistical.ex
defmodule CNS.Metrics.Statistical do
  alias ExTopology.Statistics

  @doc "Compute correlation between thesis and antithesis embeddings"
  def embedding_correlation(thesis_emb, antithesis_emb, method \\ :pearson) do
    # Flatten embeddings to 1D for correlation
    thesis_flat = Nx.to_flat_list(thesis_emb)
    antithesis_flat = Nx.to_flat_list(antithesis_emb)

    Statistics.correlation(
      Nx.tensor(thesis_flat),
      Nx.tensor(antithesis_flat),
      method: method
    )
  end

  @doc "Effect size between thesis and antithesis claim groups"
  def dialectical_effect_size(thesis_snos, antithesis_snos) do
    thesis_embs = CNS.Topology.Adapter.sno_embeddings(thesis_snos)
    antithesis_embs = CNS.Topology.Adapter.sno_embeddings(antithesis_snos)

    # Compute effect size on embedding norms
    thesis_norms = Nx.LinAlg.norm(thesis_embs, axes: [1])
    antithesis_norms = Nx.LinAlg.norm(antithesis_embs, axes: [1])

    cohens_d = Statistics.cohens_d(thesis_norms, antithesis_norms)

    %{
      cohens_d: Nx.to_number(cohens_d),
      interpretation: interpret_effect_size(Nx.to_number(cohens_d))
    }
  end

  defp interpret_effect_size(d) when abs(d) < 0.2, do: :negligible
  defp interpret_effect_size(d) when abs(d) < 0.5, do: :small
  defp interpret_effect_size(d) when abs(d) < 0.8, do: :medium
  defp interpret_effect_size(_), do: :large

  @doc "Z-scores for claim embeddings (outlier detection)"
  def embedding_z_scores(snos) do
    embeddings = CNS.Topology.Adapter.sno_embeddings(snos)
    norms = Nx.LinAlg.norm(embeddings, axes: [1])

    z_scores = Statistics.z_scores(norms)

    snos
    |> Enum.zip(Nx.to_flat_list(z_scores))
    |> Enum.map(fn {sno, z} -> %{sno: sno, z_score: z, is_outlier: abs(z) > 2.0} end)
  end
end
```

#### Deliverables
- [ ] `CNS.Metrics.Statistical` module
- [ ] Correlation analysis for dialectical pairs
- [ ] Effect size computation
- [ ] Z-score outlier detection
- [ ] Integration with chirality scoring

---

### Phase 7: Integration & Testing (Week 7-8)

**Goal**: Full integration, comprehensive testing, documentation.

#### Integration Tasks

1. **Update Pipeline**
   - Add topological validation step
   - Integrate fragility checks
   - Add persistence-based convergence criteria

2. **Update Critics**
   - `CNS.Critics.Logic` - Use ex_topology cycle detection
   - `CNS.Critics.Grounding` - Use embedding isolation scores

3. **Documentation**
   - API documentation for new modules
   - Guide: "Topological Analysis in Dialectical Reasoning"
   - Migration guide from surrogates

4. **Testing**
   - Property-based tests for adapter
   - Integration tests with synthetic SNOs
   - Performance benchmarks
   - Regression tests for existing functionality

#### Deliverables
- [ ] Updated pipeline with topological validation
- [ ] Updated critics using ex_topology
- [ ] Comprehensive test suite
- [ ] Performance benchmarks
- [ ] Documentation and guides

---

## Dependency Compatibility

### Current CNS Dependencies
```elixir
{:libgraph, "~> 0.16"},
{:nx, "~> 0.7"},
{:bumblebee, "~> 0.5", optional: true},
{:telemetry, "~> 1.2"},
{:uuid, "~> 1.1"},
{:nimble_parsec, "~> 1.4"}
```

### ex_topology Dependencies
```elixir
{:libgraph, "~> 0.16"},  # ✅ Exact match
{:nx, "~> 0.7"},         # ✅ Exact match
{:scholar, "~> 0.3"}     # New (no conflict)
```

**Compatibility**: ✅ **EXCELLENT** - No conflicts, shared dependencies.

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Performance regression | Medium | Medium | Benchmark old vs. new, optimize hot paths |
| API incompatibility | Low | Low | Adapter pattern isolates changes |
| Test failures | Medium | Low | Comprehensive equivalence tests |
| Embedding pipeline issues | Medium | Medium | Caching, lazy evaluation |
| Large network scalability | Medium | High | Consider EXLA backend, batching |

---

## Success Criteria

1. **Functional**: All existing tests pass with new implementation
2. **Performance**: No more than 10% regression on typical workloads
3. **Capability**: Full persistent homology (not stubs)
4. **Quality**: 90%+ test coverage on new modules
5. **Documentation**: Complete API docs and migration guide

---

## Timeline Summary

| Phase | Duration | Key Deliverable |
|-------|----------|-----------------|
| 1. Foundation | Week 1 | Adapter module, equivalence tests |
| 2. Graph Topology | Week 2 | `CNS.Topology.GraphAnalysis` |
| 3. Embedding Analysis | Week 3 | `CNS.Embedding.Analysis` |
| 4. Persistent Homology | Week 4 | `CNS.Topology.Persistence` |
| 5. Fragility | Week 5 | `CNS.Topology.Fragility` |
| 6. Statistics | Week 6 | `CNS.Metrics.Statistical` |
| 7-8. Integration | Weeks 7-8 | Full integration, testing, docs |

**Total**: 8 weeks to production-ready integration

---

## Appendix: Code Examples

### Example 1: Full TDA Pipeline

```elixir
# Analyze a synthesis result topologically
def analyze_synthesis(synthesis_sno) do
  # Get all claims from synthesis
  snos = extract_all_claims(synthesis_sno)

  # Compute persistent homology
  persistence = CNS.Topology.Persistence.compute(snos)

  # Check for circular reasoning
  if persistence.circular_reasoning.persistent_cycles > 0 do
    Logger.warning("Detected #{persistence.circular_reasoning.persistent_cycles} circular reasoning patterns")
  end

  # Analyze fragility
  fragility = CNS.Topology.Fragility.analyze(snos)

  %{
    persistence: persistence,
    fragility: fragility,
    quality_score: compute_quality(persistence, fragility)
  }
end
```

### Example 2: Comparing Thesis vs Antithesis

```elixir
# Compare topological structure of opposing viewpoints
def compare_dialectical_pair(thesis_snos, antithesis_snos) do
  # Persistence comparison
  distance = CNS.Topology.Persistence.compare(thesis_snos, antithesis_snos)

  # Statistical comparison
  effect_size = CNS.Metrics.Statistical.dialectical_effect_size(
    thesis_snos,
    antithesis_snos
  )

  %{
    topological_distance: distance.total_distance,
    topologically_similar?: distance.topologically_similar?,
    effect_size: effect_size
  }
end
```

---

## References

- [ex_topology v0.1.1](https://hex.pm/packages/ex_topology) - Hex package
- [ex_topology GitHub](https://github.com/North-Shore-AI/ex_topology) - Source
- [CNS Architecture](../guides/architecture.md) - Current architecture docs
- [Surrogate Validation](../surrogate_validation_implementation.md) - Existing surrogate work

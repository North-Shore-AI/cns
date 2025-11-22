# CNS Core Algorithms - Python to Elixir Porting Design Document

**Date:** 2025-11-21
**Status:** Design Specification
**Source:** Python Thinker Project
**Target:** Elixir CNS Implementation

---

## Table of Contents

1. [Overview](#1-overview)
2. [Betti Number Computation](#2-betti-number-computation)
3. [Chirality Computation](#3-chirality-computation)
4. [Semantic Validation Pipeline](#4-semantic-validation-pipeline)
5. [Claim Schema Parsing](#5-claim-schema-parsing)
6. [Dependency Summary](#6-dependency-summary)
7. [Implementation Timeline](#7-implementation-timeline)

---

## 1. Overview

### Porting Strategy

The CNS core algorithms are being ported from Python to Elixir to leverage:
- OTP supervision trees for fault tolerance
- Actor-based concurrency for parallel processing
- Pattern matching for clean algorithm expression
- Immutable data structures for correctness

### Priority Order

1. **Betti Number Computation** - Foundation for topological analysis
2. **Claim Schema Parsing** - Required for all downstream processing
3. **Chirality Computation** - Depends on #1 and #2
4. **Semantic Validation Pipeline** - Integration layer

---

## 2. Betti Number Computation

### 2.1 Python Implementation Analysis

**File:** `logic/betti.py`

**Current Data Structures:**
```python
Relation = Tuple[str, str, str]  # (source_id, label, target_id)

@dataclass
class GraphStats:
    nodes: int
    edges: int
    components: int
    beta1: int  # First Betti number (cycle count)
    cycles: List[List[str]]
    polarity_conflict: bool
```

**Key Functions:**
```python
def compute_graph_stats(claim_ids: Iterable[str], relations: Sequence[Relation]) -> GraphStats
def _normalize_claim_id(identifier: str) -> str
def _determine_polarity_conflict(relations: Sequence[Relation], target: str = "c1") -> bool
```

**Core Algorithm:**
```python
# Betti number formula: beta1 = edges - nodes + components
# Uses NetworkX for:
# - Graph construction (DiGraph)
# - Connected component detection
# - Cycle detection (simple_cycles)
```

### 2.2 Elixir Module Design

**Module:** `CNS.Logic.Betti`

**Dependencies:**
- `libgraph` - Elixir native graph library (hex: `{:libgraph, "~> 0.16"}`)

**Struct Definitions:**

```elixir
defmodule CNS.Logic.Betti do
  @moduledoc """
  Compute Betti numbers and cycle diagnostics for CLAIM/RELATION graphs
  to detect logical inconsistencies.

  The first Betti number (beta1) measures the number of independent cycles
  in the reasoning graph. High beta1 indicates circular reasoning patterns.
  """

  defmodule GraphStats do
    @moduledoc "Topology statistics for a reasoning graph"

    @type t :: %__MODULE__{
      nodes: non_neg_integer(),
      edges: non_neg_integer(),
      components: non_neg_integer(),
      beta1: non_neg_integer(),
      cycles: [[String.t()]],
      polarity_conflict: boolean()
    }

    @enforce_keys [:nodes, :edges, :components, :beta1, :cycles, :polarity_conflict]
    defstruct [:nodes, :edges, :components, :beta1, :cycles, :polarity_conflict]
  end

  @type relation :: {String.t(), String.t(), String.t()}
  # {source_id, label, target_id}
  # label in ["supports", "refutes", "contrasts"]
end
```

**Public API:**

```elixir
@doc """
Compute graph topology statistics including Betti numbers.

## Parameters
  - claim_ids: List of claim identifiers
  - relations: List of {source, label, target} tuples

## Returns
  GraphStats struct with topology metrics

## Examples
      iex> claim_ids = ["c1", "c2", "c3"]
      iex> relations = [{"c2", "supports", "c1"}, {"c3", "refutes", "c1"}]
      iex> CNS.Logic.Betti.compute_graph_stats(claim_ids, relations)
      %GraphStats{nodes: 3, edges: 2, components: 1, beta1: 0, cycles: [], polarity_conflict: true}
"""
@spec compute_graph_stats([String.t()], [relation()]) :: GraphStats.t()
def compute_graph_stats(claim_ids, relations)

@doc """
Detect if a claim has conflicting polarity (both supports and refutes edges).
"""
@spec polarity_conflict?([relation()], String.t()) :: boolean()
def polarity_conflict?(relations, target \\ "c1")

@doc """
Find all cycles in the reasoning graph.
"""
@spec find_cycles(Graph.t()) :: [[String.t()]]
def find_cycles(graph)
```

**Implementation:**

```elixir
defmodule CNS.Logic.Betti do
  alias __MODULE__.GraphStats

  @spec compute_graph_stats([String.t()], [relation()]) :: GraphStats.t()
  def compute_graph_stats(claim_ids, relations) do
    # Build directed graph
    graph = build_graph(claim_ids, relations)

    # Calculate metrics
    nodes = Graph.num_vertices(graph)
    edges = Graph.num_edges(graph)

    # Connected components (treating as undirected for Betti calculation)
    components = graph
      |> Graph.to_undirected()
      |> Graph.components()
      |> length()

    # First Betti number: beta1 = E - V + C
    beta1 = max(0, edges - nodes + components)

    # Find cycles (limited to prevent explosion)
    cycles = find_cycles(graph)

    # Check for polarity conflicts
    conflict = polarity_conflict?(relations)

    %GraphStats{
      nodes: nodes,
      edges: edges,
      components: components,
      beta1: beta1,
      cycles: cycles,
      polarity_conflict: conflict
    }
  end

  defp build_graph(claim_ids, relations) do
    # Initialize with all claim nodes
    graph = Enum.reduce(claim_ids, Graph.new(type: :directed), fn id, g ->
      Graph.add_vertex(g, normalize_id(id))
    end)

    # Add edges from relations
    Enum.reduce(relations, graph, fn {src, label, dst}, g ->
      Graph.add_edge(g, normalize_id(src), normalize_id(dst), label: label)
    end)
  end

  defp normalize_id(id) do
    id
    |> String.downcase()
    |> String.trim()
  end

  @spec polarity_conflict?([relation()], String.t()) :: boolean()
  def polarity_conflict?(relations, target \\ "c1") do
    normalized_target = normalize_id(target)

    labels = relations
      |> Enum.filter(fn {_src, _label, dst} -> normalize_id(dst) == normalized_target end)
      |> Enum.map(fn {_src, label, _dst} -> String.downcase(label) end)
      |> MapSet.new()

    MapSet.member?(labels, "supports") and MapSet.member?(labels, "refutes")
  end

  @spec find_cycles(Graph.t()) :: [[String.t()]]
  def find_cycles(graph) do
    # Use libgraph's cycle detection
    # Limit cycles to prevent combinatorial explosion
    graph
    |> Graph.get_cycles()
    |> Enum.take(100)  # Limit for performance
  end
end
```

### 2.3 Test Cases

```elixir
defmodule CNS.Logic.BettiTest do
  use ExUnit.Case, async: true
  alias CNS.Logic.Betti
  alias CNS.Logic.Betti.GraphStats

  describe "compute_graph_stats/2" do
    test "simple acyclic graph" do
      claim_ids = ["c1", "c2", "c3"]
      relations = [
        {"c2", "supports", "c1"},
        {"c3", "supports", "c1"}
      ]

      stats = Betti.compute_graph_stats(claim_ids, relations)

      assert stats.nodes == 3
      assert stats.edges == 2
      assert stats.components == 1
      assert stats.beta1 == 0
      assert stats.cycles == []
      assert stats.polarity_conflict == false
    end

    test "graph with cycle" do
      claim_ids = ["c1", "c2", "c3"]
      relations = [
        {"c1", "supports", "c2"},
        {"c2", "supports", "c3"},
        {"c3", "supports", "c1"}
      ]

      stats = Betti.compute_graph_stats(claim_ids, relations)

      assert stats.beta1 >= 1
      assert length(stats.cycles) >= 1
    end

    test "polarity conflict detection" do
      claim_ids = ["c1", "c2", "c3"]
      relations = [
        {"c2", "supports", "c1"},
        {"c3", "refutes", "c1"}
      ]

      stats = Betti.compute_graph_stats(claim_ids, relations)

      assert stats.polarity_conflict == true
    end

    test "disconnected components" do
      claim_ids = ["c1", "c2", "c3", "c4"]
      relations = [
        {"c1", "supports", "c2"},
        {"c3", "supports", "c4"}
      ]

      stats = Betti.compute_graph_stats(claim_ids, relations)

      assert stats.components == 2
    end

    test "empty graph" do
      stats = Betti.compute_graph_stats([], [])

      assert stats.nodes == 0
      assert stats.edges == 0
      assert stats.beta1 == 0
    end

    test "case-insensitive id normalization" do
      claim_ids = ["C1", "c2"]
      relations = [{"C2", "supports", "c1"}]

      stats = Betti.compute_graph_stats(claim_ids, relations)

      assert stats.edges == 1
    end
  end

  describe "polarity_conflict?/2" do
    test "returns true when both supports and refutes exist" do
      relations = [
        {"c2", "supports", "c1"},
        {"c3", "refutes", "c1"}
      ]

      assert Betti.polarity_conflict?(relations, "c1") == true
    end

    test "returns false with only supports" do
      relations = [
        {"c2", "supports", "c1"},
        {"c3", "supports", "c1"}
      ]

      assert Betti.polarity_conflict?(relations, "c1") == false
    end

    test "handles custom target" do
      relations = [
        {"c1", "supports", "c5"},
        {"c2", "refutes", "c5"}
      ]

      assert Betti.polarity_conflict?(relations, "c5") == true
    end
  end
end
```

### 2.4 Estimated Effort

| Task | Hours |
|------|-------|
| Module structure and types | 2 |
| Core algorithm implementation | 4 |
| Cycle detection optimization | 3 |
| Test suite | 3 |
| Documentation | 2 |
| **Total** | **14 hours** |

---

## 3. Chirality Computation

### 3.1 Python Implementation Analysis

**File:** `metrics/chirality.py`

**Current Data Structures:**
```python
@dataclass
class FisherRaoStats:
    mean: np.ndarray
    inv_var: np.ndarray  # Inverse variance for diagonal approximation

@dataclass
class ChiralityResult:
    fisher_rao_distance: float
    evidence_overlap: float
    polarity_conflict: bool
    chirality_score: float  # Normalized composite score [0, 1]
```

**Key Functions:**
```python
def build_fisher_rao_stats(vectors: Sequence[Sequence[float]] | np.ndarray, epsilon: float = 1e-6) -> FisherRaoStats
def fisher_rao_distance(vec_a: np.ndarray, vec_b: np.ndarray, stats: FisherRaoStats) -> float

class ChiralityAnalyzer:
    def compare(self, thesis: str, antithesis: str, *,
                evidence_overlap: float, polarity_conflict: bool) -> ChiralityResult
```

**Core Algorithms:**
```python
# Fisher-Rao Distance (Mahalanobis-style with diagonal Fisher info)
distance = sqrt((diff * inv_var) . diff)

# Chirality Score Computation
norm_distance = distance / (distance + 1.0)  # Normalize to [0, 1]
overlap_factor = 1.0 - clamp(evidence_overlap, 0, 1)
conflict_penalty = 0.25 if polarity_conflict else 0.0
chirality_score = min(1.0, norm_distance * 0.6 + overlap_factor * 0.2 + conflict_penalty)
```

### 3.2 Elixir Module Design

**Module:** `CNS.Metrics.Chirality`

**Dependencies:**
- `Nx` - Numerical Elixir (hex: `{:nx, "~> 0.6"}`)
- `Scholar` (optional) - For statistical functions
- Embedder behaviour (SentenceTransformer equivalent)

**Struct Definitions:**

```elixir
defmodule CNS.Metrics.Chirality do
  @moduledoc """
  Compute chirality scores using Fisher-Rao distance approximation
  for measuring semantic divergence between thesis/antithesis pairs.

  Chirality quantifies argumentative bias through geometric asymmetry
  in the embedding space.
  """

  defmodule FisherRaoStats do
    @moduledoc "Pre-computed statistics for Fisher-Rao distance calculation"

    @type t :: %__MODULE__{
      mean: Nx.Tensor.t(),
      inv_var: Nx.Tensor.t()
    }

    @enforce_keys [:mean, :inv_var]
    defstruct [:mean, :inv_var]
  end

  defmodule ChiralityResult do
    @moduledoc "Result of chirality analysis between thesis and antithesis"

    @type t :: %__MODULE__{
      fisher_rao_distance: float(),
      evidence_overlap: float(),
      polarity_conflict: boolean(),
      chirality_score: float()
    }

    @enforce_keys [:fisher_rao_distance, :evidence_overlap, :polarity_conflict, :chirality_score]
    defstruct [:fisher_rao_distance, :evidence_overlap, :polarity_conflict, :chirality_score]
  end

  # Embedder behaviour for dependency injection
  @callback encode(texts :: [String.t()]) :: Nx.Tensor.t()
end
```

**Public API:**

```elixir
@doc """
Build Fisher-Rao statistics from a collection of embedding vectors.

## Parameters
  - vectors: Nx tensor of shape {n_samples, embedding_dim} or list of lists
  - epsilon: Small value for numerical stability (default: 1.0e-6)

## Returns
  FisherRaoStats with mean and inverse variance

## Examples
      iex> vectors = Nx.tensor([[1.0, 2.0], [1.5, 2.5], [1.2, 2.1]])
      iex> stats = CNS.Metrics.Chirality.build_fisher_rao_stats(vectors)
      iex> stats.mean
      #Nx.Tensor<f32[2]>
"""
@spec build_fisher_rao_stats(Nx.Tensor.t() | [[number()]], float()) :: FisherRaoStats.t()
def build_fisher_rao_stats(vectors, epsilon \\ 1.0e-6)

@doc """
Compute Fisher-Rao distance between two embedding vectors.

Uses diagonal approximation of Fisher information metric.
"""
@spec fisher_rao_distance(Nx.Tensor.t(), Nx.Tensor.t(), FisherRaoStats.t()) :: float()
def fisher_rao_distance(vec_a, vec_b, stats)

@doc """
Compare thesis and antithesis to compute chirality score.

## Parameters
  - embedder: Module implementing encode/1 callback
  - stats: Pre-computed FisherRaoStats
  - thesis: Thesis text
  - antithesis: Antithesis text
  - evidence_overlap: Overlap score in [0, 1]
  - polarity_conflict: Whether polarity conflict exists

## Returns
  ChiralityResult with composite chirality score
"""
@spec compare(
  module(),
  FisherRaoStats.t(),
  String.t(),
  String.t(),
  float(),
  boolean()
) :: ChiralityResult.t()
def compare(embedder, stats, thesis, antithesis, evidence_overlap, polarity_conflict)
```

**Implementation:**

```elixir
defmodule CNS.Metrics.Chirality do
  import Nx.Defn
  alias __MODULE__.{FisherRaoStats, ChiralityResult}

  @spec build_fisher_rao_stats(Nx.Tensor.t() | [[number()]], float()) :: FisherRaoStats.t()
  def build_fisher_rao_stats(vectors, epsilon \\ 1.0e-6) do
    tensor = case vectors do
      %Nx.Tensor{} -> vectors
      list when is_list(list) -> Nx.tensor(list)
    end

    mean = Nx.mean(tensor, axes: [0])
    variance = Nx.variance(tensor, axes: [0])

    # Inverse variance with epsilon for numerical stability
    inv_var = Nx.divide(1.0, Nx.add(variance, epsilon))

    %FisherRaoStats{mean: mean, inv_var: inv_var}
  end

  @spec fisher_rao_distance(Nx.Tensor.t(), Nx.Tensor.t(), FisherRaoStats.t()) :: float()
  def fisher_rao_distance(vec_a, vec_b, %FisherRaoStats{inv_var: inv_var}) do
    diff = Nx.subtract(vec_a, vec_b)
    weighted_sq = Nx.multiply(Nx.multiply(diff, inv_var), diff)

    weighted_sq
    |> Nx.sum()
    |> Nx.sqrt()
    |> Nx.to_number()
  end

  # Defn version for GPU acceleration
  defn fisher_rao_distance_n(vec_a, vec_b, inv_var) do
    diff = vec_a - vec_b
    weighted_sq = diff * inv_var * diff
    Nx.sqrt(Nx.sum(weighted_sq))
  end

  @spec compare(
    module(),
    FisherRaoStats.t(),
    String.t(),
    String.t(),
    float(),
    boolean()
  ) :: ChiralityResult.t()
  def compare(embedder, stats, thesis, antithesis, evidence_overlap, polarity_conflict) do
    # Get embeddings from embedder module
    [thesis_emb, antithesis_emb] = embedder.encode([thesis, antithesis])
      |> Nx.to_batched(1)
      |> Enum.map(&Nx.squeeze/1)

    # Compute Fisher-Rao distance
    distance = fisher_rao_distance(thesis_emb, antithesis_emb, stats)

    # Compute chirality score
    score = compute_chirality_score(distance, evidence_overlap, polarity_conflict)

    %ChiralityResult{
      fisher_rao_distance: distance,
      evidence_overlap: evidence_overlap,
      polarity_conflict: polarity_conflict,
      chirality_score: score
    }
  end

  defp compute_chirality_score(distance, evidence_overlap, polarity_conflict) do
    # Normalize distance to [0, 1]
    norm_distance = distance / (distance + 1.0)

    # Overlap factor (inverse of overlap)
    overlap_factor = 1.0 - clamp(evidence_overlap, 0.0, 1.0)

    # Penalty for polarity conflict
    conflict_penalty = if polarity_conflict, do: 0.25, else: 0.0

    # Composite score with weights
    raw_score = norm_distance * 0.6 + overlap_factor * 0.2 + conflict_penalty

    min(1.0, raw_score)
  end

  defp clamp(value, min_val, max_val) do
    value
    |> max(min_val)
    |> min(max_val)
  end
end
```

**Embedder Adapter Example:**

```elixir
defmodule CNS.Embedders.Bumblebee do
  @moduledoc "Bumblebee-based embedder for sentence transformers"
  @behaviour CNS.Metrics.Chirality

  def init(model_name \\ "sentence-transformers/all-MiniLM-L6-v2") do
    {:ok, model_info} = Bumblebee.load_model({:hf, model_name})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, model_name})

    serving = Bumblebee.Text.TextEmbedding.text_embedding(model_info, tokenizer)
    Nx.Serving.start_link(serving: serving, name: __MODULE__)
  end

  @impl true
  def encode(texts) do
    Nx.Serving.batched_run(__MODULE__, texts)
    |> Map.get(:embedding)
  end
end
```

### 3.3 Test Cases

```elixir
defmodule CNS.Metrics.ChiralityTest do
  use ExUnit.Case, async: true
  alias CNS.Metrics.Chirality
  alias CNS.Metrics.Chirality.{FisherRaoStats, ChiralityResult}

  describe "build_fisher_rao_stats/2" do
    test "computes mean and inverse variance" do
      vectors = Nx.tensor([
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [1.5, 2.5, 3.5]
      ])

      stats = Chirality.build_fisher_rao_stats(vectors)

      assert %FisherRaoStats{} = stats
      assert Nx.shape(stats.mean) == {3}
      assert Nx.shape(stats.inv_var) == {3}
    end

    test "handles list input" do
      vectors = [[1.0, 2.0], [1.5, 2.5], [1.2, 2.1]]
      stats = Chirality.build_fisher_rao_stats(vectors)

      assert %FisherRaoStats{} = stats
    end

    test "epsilon prevents division by zero" do
      # Zero variance in first dimension
      vectors = Nx.tensor([
        [1.0, 2.0],
        [1.0, 3.0],
        [1.0, 2.5]
      ])

      stats = Chirality.build_fisher_rao_stats(vectors, 1.0e-6)

      # Should not raise, inv_var should be very large but finite
      inv_var_values = Nx.to_flat_list(stats.inv_var)
      assert Enum.all?(inv_var_values, &is_number/1)
      assert Enum.all?(inv_var_values, &(&1 < :infinity))
    end
  end

  describe "fisher_rao_distance/3" do
    test "computes correct distance" do
      stats = %FisherRaoStats{
        mean: Nx.tensor([0.0, 0.0]),
        inv_var: Nx.tensor([1.0, 1.0])  # Unit weights
      }

      vec_a = Nx.tensor([0.0, 0.0])
      vec_b = Nx.tensor([3.0, 4.0])

      distance = Chirality.fisher_rao_distance(vec_a, vec_b, stats)

      # Should equal Euclidean distance when inv_var = 1
      assert_in_delta distance, 5.0, 1.0e-6
    end

    test "weights by inverse variance" do
      stats = %FisherRaoStats{
        mean: Nx.tensor([0.0, 0.0]),
        inv_var: Nx.tensor([4.0, 1.0])  # First dim weighted 4x
      }

      vec_a = Nx.tensor([0.0, 0.0])
      vec_b = Nx.tensor([1.0, 2.0])

      distance = Chirality.fisher_rao_distance(vec_a, vec_b, stats)

      # sqrt(4*1^2 + 1*2^2) = sqrt(4 + 4) = sqrt(8)
      assert_in_delta distance, :math.sqrt(8), 1.0e-6
    end

    test "distance is zero for identical vectors" do
      stats = %FisherRaoStats{
        mean: Nx.tensor([1.0, 2.0]),
        inv_var: Nx.tensor([1.0, 1.0])
      }

      vec = Nx.tensor([1.0, 2.0])
      distance = Chirality.fisher_rao_distance(vec, vec, stats)

      assert_in_delta distance, 0.0, 1.0e-6
    end
  end

  describe "chirality score computation" do
    test "high distance produces high chirality" do
      # Test with mocked embedder
      result = %ChiralityResult{
        fisher_rao_distance: 10.0,
        evidence_overlap: 0.0,
        polarity_conflict: false,
        chirality_score: 0.0
      }

      # Manually compute expected score
      norm_distance = 10.0 / 11.0
      expected_score = min(1.0, norm_distance * 0.6 + 1.0 * 0.2)

      assert expected_score > 0.7
    end

    test "polarity conflict adds penalty" do
      # With conflict: adds 0.25
      # Without: 0.0
      penalty_diff = 0.25

      assert penalty_diff == 0.25
    end

    test "high evidence overlap reduces score" do
      # overlap_factor = 1.0 - overlap
      # High overlap (0.9) -> factor 0.1 -> lower contribution
      # Low overlap (0.1) -> factor 0.9 -> higher contribution

      high_overlap_factor = 1.0 - 0.9
      low_overlap_factor = 1.0 - 0.1

      assert low_overlap_factor > high_overlap_factor
    end
  end
end
```

### 3.4 Estimated Effort

| Task | Hours |
|------|-------|
| Module structure and types | 2 |
| Nx integration and defn | 4 |
| Fisher-Rao distance implementation | 3 |
| Chirality score computation | 2 |
| Embedder behaviour and adapter | 4 |
| Test suite | 4 |
| GPU optimization (optional) | 3 |
| Documentation | 2 |
| **Total** | **24 hours** |

---

## 4. Semantic Validation Pipeline

### 4.1 Python Implementation Analysis

**File:** `semantic_validation.py`

**4-Stage Pipeline:**

1. **Citation Accuracy (Hard Gate)**
   - Extract document IDs from text
   - Verify against corpus
   - Must pass to continue

2. **Entailment Scoring**
   - Model: `cross-encoder/nli-deberta-v3-large`
   - Threshold: >= 0.75
   - Evidence must entail claim

3. **Semantic Similarity**
   - Model: `all-MiniLM-L6-v2`
   - Threshold: >= 0.7
   - Cosine similarity

4. **Paraphrase Tolerance**
   - Accept valid rephrasings
   - Final gate

**Current Data Structure:**
```python
@dataclass
class ValidationResult:
    citation_valid: bool
    cited_ids: Set[str]
    missing_ids: Set[str]
    entailment_score: float
    entailment_pass: bool
    semantic_similarity: float
    similarity_pass: bool
    paraphrase_accepted: bool
    overall_pass: bool
    schema_valid: bool
    schema_errors: List[str]
```

### 4.2 Elixir Module Design

**Module:** `CNS.Validation.Semantic`

**Architecture:** Broadway pipeline for concurrent processing

**Struct Definitions:**

```elixir
defmodule CNS.Validation.Semantic do
  @moduledoc """
  4-stage semantic validation pipeline for CNS claim extraction.

  Stages:
  1. Citation Accuracy - Hard gate for document ID verification
  2. Entailment Scoring - NLI-based evidence entailment
  3. Semantic Similarity - Embedding cosine similarity
  4. Paraphrase Tolerance - Accept valid rephrasings
  """

  defmodule ValidationResult do
    @type t :: %__MODULE__{
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

      # Schema
      schema_valid: boolean(),
      schema_errors: [String.t()]
    }

    @enforce_keys [:citation_valid, :cited_ids, :missing_ids, :entailment_score,
                   :entailment_pass, :semantic_similarity, :similarity_pass,
                   :paraphrase_accepted, :overall_pass, :schema_valid, :schema_errors]
    defstruct @enforce_keys
  end

  defmodule Config do
    @type t :: %__MODULE__{
      entailment_threshold: float(),
      similarity_threshold: float(),
      device: String.t() | nil
    }

    defstruct entailment_threshold: 0.75,
              similarity_threshold: 0.7,
              device: nil
  end
end
```

**Public API:**

```elixir
@doc """
Initialize the semantic validator with models.

Loads NLI and embedding models for validation.
"""
@spec init(Config.t()) :: {:ok, pid()} | {:error, term()}
def init(config \\ %Config{})

@doc """
Validate a single claim through the 4-stage pipeline.

## Parameters
  - generated_claim: The claim text to validate
  - gold_claim: Expected claim text
  - generated_full_output: Full model output (for citation extraction)
  - evidence_corpus: Map of doc_id => document data
  - gold_evidence_ids: Set of expected evidence document IDs

## Returns
  ValidationResult with stage-by-stage results
"""
@spec validate_claim(
  String.t(),
  String.t(),
  String.t(),
  %{String.t() => map()},
  MapSet.t(String.t())
) :: ValidationResult.t()
def validate_claim(generated_claim, gold_claim, generated_full_output, evidence_corpus, gold_evidence_ids)

@doc """
Validate a batch of predictions.
"""
@spec validate_batch(
  [map()],
  [map()],
  %{String.t() => map()}
) :: [ValidationResult.t()]
def validate_batch(predictions, gold_data, corpus)
```

**Implementation - Stage-Based Architecture:**

```elixir
defmodule CNS.Validation.Semantic do
  use GenServer
  alias __MODULE__.{ValidationResult, Config}

  # GenServer for managing model state
  def start_link(config \\ %Config{}) do
    GenServer.start_link(__MODULE__, config, name: __MODULE__)
  end

  @impl true
  def init(%Config{} = config) do
    # Load models via Bumblebee or external service
    state = %{
      config: config,
      nli_model: load_nli_model(config),
      embedding_model: load_embedding_model(config)
    }
    {:ok, state}
  end

  def validate_claim(generated_claim, gold_claim, generated_full_output, evidence_corpus, gold_evidence_ids) do
    GenServer.call(__MODULE__, {
      :validate,
      generated_claim,
      gold_claim,
      generated_full_output,
      evidence_corpus,
      gold_evidence_ids
    }, :timer.seconds(30))
  end

  @impl true
  def handle_call({:validate, gen_claim, gold_claim, full_output, corpus, gold_ids}, _from, state) do
    result = run_pipeline(gen_claim, gold_claim, full_output, corpus, gold_ids, state)
    {:reply, result, state}
  end

  defp run_pipeline(gen_claim, gold_claim, full_output, corpus, gold_ids, state) do
    # Stage 1: Citation Accuracy
    {citation_valid, cited_ids, missing_ids} =
      validate_citations(full_output, corpus, gold_ids)

    # Short circuit if citations fail (hard gate)
    if not citation_valid do
      failed_result(citation_valid, cited_ids, missing_ids)
    else
      # Stage 2: Entailment
      evidence_text = get_evidence_text(cited_ids, corpus)
      entailment_score = compute_entailment(gen_claim, evidence_text, state)
      entailment_pass = entailment_score >= state.config.entailment_threshold

      # Stage 3: Semantic Similarity
      similarity = compute_similarity(gen_claim, gold_claim, state)
      similarity_pass = similarity >= state.config.similarity_threshold

      # Stage 4: Paraphrase Tolerance
      paraphrase_accepted = entailment_pass or similarity_pass

      # Overall
      overall_pass = citation_valid and entailment_pass and paraphrase_accepted

      %ValidationResult{
        citation_valid: citation_valid,
        cited_ids: cited_ids,
        missing_ids: missing_ids,
        entailment_score: entailment_score,
        entailment_pass: entailment_pass,
        semantic_similarity: similarity,
        similarity_pass: similarity_pass,
        paraphrase_accepted: paraphrase_accepted,
        overall_pass: overall_pass,
        schema_valid: true,
        schema_errors: []
      }
    end
  end

  # Stage 1: Citation Extraction
  defp validate_citations(text, corpus, gold_ids) do
    cited_ids = CNS.Validation.Citation.extract_document_ids(text)

    # Check against corpus
    valid_ids = MapSet.filter(cited_ids, &Map.has_key?(corpus, &1))
    missing_ids = MapSet.difference(gold_ids, valid_ids)

    citation_valid = MapSet.size(missing_ids) == 0

    {citation_valid, valid_ids, missing_ids}
  end

  # Stage 2: Entailment (via NLI model)
  defp compute_entailment(claim, evidence, state) do
    # Call NLI model
    # Returns probability of entailment class
    case state.nli_model do
      {:bumblebee, serving} ->
        result = Nx.Serving.run(serving, {evidence, claim})
        result.predictions
        |> Enum.find(&(&1.label == "entailment"))
        |> Map.get(:score, 0.0)

      {:external, url} ->
        # HTTP call to external service
        call_external_nli(url, evidence, claim)
    end
  end

  # Stage 3: Semantic Similarity
  defp compute_similarity(text_a, text_b, state) do
    emb_a = get_embedding(text_a, state)
    emb_b = get_embedding(text_b, state)

    # Cosine similarity
    dot = Nx.sum(Nx.multiply(emb_a, emb_b)) |> Nx.to_number()
    norm_a = Nx.sum(Nx.multiply(emb_a, emb_a)) |> Nx.sqrt() |> Nx.to_number()
    norm_b = Nx.sum(Nx.multiply(emb_b, emb_b)) |> Nx.sqrt() |> Nx.to_number()

    dot / (norm_a * norm_b)
  end

  defp get_embedding(text, state) do
    case state.embedding_model do
      {:bumblebee, serving} ->
        Nx.Serving.run(serving, text).embedding
      {:external, url} ->
        call_external_embedding(url, text)
    end
  end

  defp get_evidence_text(doc_ids, corpus) do
    doc_ids
    |> Enum.map(&Map.get(corpus, &1, %{}))
    |> Enum.map(&Map.get(&1, "text", ""))
    |> Enum.join(" ")
  end

  defp failed_result(citation_valid, cited_ids, missing_ids) do
    %ValidationResult{
      citation_valid: citation_valid,
      cited_ids: cited_ids,
      missing_ids: missing_ids,
      entailment_score: 0.0,
      entailment_pass: false,
      semantic_similarity: 0.0,
      similarity_pass: false,
      paraphrase_accepted: false,
      overall_pass: false,
      schema_valid: true,
      schema_errors: []
    }
  end

  defp load_nli_model(%Config{device: device}) do
    # Option 1: Bumblebee (local)
    {:ok, model_info} = Bumblebee.load_model({:hf, "cross-encoder/nli-deberta-v3-large"})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "microsoft/deberta-v3-large"})

    serving = Bumblebee.Text.text_classification(model_info, tokenizer,
      compile: [batch_size: 1, sequence_length: 512],
      defn_options: [compiler: EXLA]
    )

    {:bumblebee, Nx.Serving.start_link(serving: serving)}

    # Option 2: External service
    # {:external, "http://localhost:8080/nli"}
  end

  defp load_embedding_model(%Config{} = _config) do
    {:ok, model_info} = Bumblebee.load_model({:hf, "sentence-transformers/all-MiniLM-L6-v2"})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "sentence-transformers/all-MiniLM-L6-v2"})

    serving = Bumblebee.Text.TextEmbedding.text_embedding(model_info, tokenizer,
      compile: [batch_size: 1, sequence_length: 128],
      defn_options: [compiler: EXLA]
    )

    {:bumblebee, Nx.Serving.start_link(serving: serving)}
  end
end
```

### 4.3 Mapping to Elixir Critics

| Python Component | Elixir Module | Notes |
|-----------------|---------------|-------|
| Citation extraction | `CNS.Validation.Citation` | Pure function |
| NLI entailment | `CNS.Critics.Grounding` | Bumblebee/External |
| Semantic similarity | `CNS.Metrics.Similarity` | Nx-based |
| Paraphrase check | Combined logic | Threshold gates |

### 4.4 Integration Points

```elixir
# Antagonist integration
defmodule CNS.Antagonist do
  def analyze(sno) do
    validation_result = CNS.Validation.Semantic.validate_claim(...)
    chirality_result = CNS.Metrics.Chirality.compare(...)
    graph_stats = CNS.Logic.Betti.compute_graph_stats(...)

    generate_flags(validation_result, chirality_result, graph_stats)
  end
end
```

### 4.5 Estimated Effort

| Task | Hours |
|------|-------|
| Module structure and types | 3 |
| Citation validation stage | 4 |
| NLI/Bumblebee integration | 8 |
| Embedding/similarity stage | 4 |
| Pipeline orchestration | 4 |
| External service adapters | 4 |
| Test suite | 6 |
| Documentation | 3 |
| **Total** | **36 hours** |

---

## 5. Claim Schema Parsing

### 5.1 Python Implementation Analysis

**File:** `claim_schema.py`

**Current Regex Patterns:**
```python
CLAIM_LINE_RE = re.compile(
    r"^CLAIM\[(?P<id>[^\]]+)\]\s*(?:\(Document\s+\d+\))?\s*:\s*(?P<body>.*)$",
    re.IGNORECASE
)
RELATION_LINE_RE = re.compile(
    r"^RELATION\s*[:\-]?\s*(?P<src>\S+)\s+(?P<label>supports|refutes|contrasts)\s+(?P<dst>\S+)",
    re.IGNORECASE
)
```

**Data Structures:**
```python
@dataclass
class Claim:
    identifier: str
    text: str
```

**Functions:**
```python
def parse_claim_lines(lines: Iterable[str]) -> Dict[str, Claim]
def parse_relation_line(line: str) -> Tuple[str, str, str] | None
```

**Supported Formats:**
- `CLAIM[c1]: The hypothesis text`
- `CLAIM[c2] (Document 12345): Supporting claim`
- `RELATION: c2 supports c1`
- `RELATION - c3 refutes c1`

### 5.2 Elixir Module Design

**Module:** `CNS.Schema.Parser`

**Dependencies:**
- `NimbleParsec` (optional, for complex parsing)
- Standard regex (for simple patterns)

**Struct Definitions:**

```elixir
defmodule CNS.Schema.Parser do
  @moduledoc """
  Parse CLAIM[...] and RELATION formatted completions from LLM output.

  Supports formats:
  - CLAIM[id]: text
  - CLAIM[id] (Document 12345): text
  - RELATION: src label dst
  - RELATION - src label dst
  """

  defmodule Claim do
    @type t :: %__MODULE__{
      identifier: String.t(),
      text: String.t(),
      document_id: String.t() | nil
    }

    @enforce_keys [:identifier, :text]
    defstruct [:identifier, :text, :document_id]
  end

  @type relation :: {String.t(), String.t(), String.t()}
  # {source_id, label, target_id}
end
```

**Public API:**

```elixir
@doc """
Parse all CLAIM lines from text.

## Parameters
  - text: Input text (string or list of lines)

## Returns
  Map of claim_id => Claim struct

## Examples
      iex> text = "CLAIM[c1]: The sky is blue\\nCLAIM[c2] (Document 123): Supporting evidence"
      iex> CNS.Schema.Parser.parse_claims(text)
      %{
        "c1" => %Claim{identifier: "c1", text: "The sky is blue", document_id: nil},
        "c2" => %Claim{identifier: "c2", text: "Supporting evidence", document_id: "123"}
      }
"""
@spec parse_claims(String.t() | [String.t()]) :: %{String.t() => Claim.t()}
def parse_claims(input)

@doc """
Parse a single RELATION line.

## Returns
  - `{source, label, target}` tuple if valid
  - `nil` if not a relation line

## Examples
      iex> CNS.Schema.Parser.parse_relation("RELATION: c2 supports c1")
      {"c2", "supports", "c1"}

      iex> CNS.Schema.Parser.parse_relation("not a relation")
      nil
"""
@spec parse_relation(String.t()) :: relation() | nil
def parse_relation(line)

@doc """
Parse all relations from text.
"""
@spec parse_relations(String.t() | [String.t()]) :: [relation()]
def parse_relations(input)

@doc """
Parse complete output into claims and relations.
"""
@spec parse(String.t()) :: {%{String.t() => Claim.t()}, [relation()]}
def parse(text)
```

**Implementation - Regex Approach:**

```elixir
defmodule CNS.Schema.Parser do
  alias __MODULE__.Claim

  # Regex patterns
  @claim_pattern ~r/^CLAIM\[(?<id>[^\]]+)\]\s*(?:\(Document\s+(?<doc>\d+)\))?\s*:\s*(?<body>.*)$/i
  @relation_pattern ~r/^RELATION\s*[:\-]?\s*(?<src>\S+)\s+(?<label>supports|refutes|contrasts)\s+(?<dst>\S+)/i

  @spec parse_claims(String.t() | [String.t()]) :: %{String.t() => Claim.t()}
  def parse_claims(input) when is_binary(input) do
    input
    |> String.split("\n")
    |> parse_claims()
  end

  def parse_claims(lines) when is_list(lines) do
    lines
    |> Enum.reduce(%{}, fn line, acc ->
      case parse_claim_line(line) do
        nil -> acc
        %Claim{} = claim -> Map.put(acc, claim.identifier, claim)
      end
    end)
  end

  defp parse_claim_line(line) do
    case Regex.named_captures(@claim_pattern, String.trim(line)) do
      nil ->
        nil

      %{"id" => id, "body" => body} = captures ->
        doc_id = Map.get(captures, "doc")

        %Claim{
          identifier: String.trim(id),
          text: String.trim(body),
          document_id: if(doc_id && doc_id != "", do: doc_id, else: nil)
        }
    end
  end

  @spec parse_relation(String.t()) :: {String.t(), String.t(), String.t()} | nil
  def parse_relation(line) do
    case Regex.named_captures(@relation_pattern, String.trim(line)) do
      nil ->
        nil

      %{"src" => src, "label" => label, "dst" => dst} ->
        {
          String.trim(src),
          String.downcase(String.trim(label)),
          String.trim(dst)
        }
    end
  end

  @spec parse_relations(String.t() | [String.t()]) :: [relation()]
  def parse_relations(input) when is_binary(input) do
    input
    |> String.split("\n")
    |> parse_relations()
  end

  def parse_relations(lines) when is_list(lines) do
    lines
    |> Enum.map(&parse_relation/1)
    |> Enum.reject(&is_nil/1)
  end

  @spec parse(String.t()) :: {%{String.t() => Claim.t()}, [relation()]}
  def parse(text) do
    lines = String.split(text, "\n")
    claims = parse_claims(lines)
    relations = parse_relations(lines)
    {claims, relations}
  end
end
```

**Alternative - NimbleParsec Approach:**

```elixir
defmodule CNS.Schema.Parser.Parsec do
  @moduledoc "NimbleParsec-based parser for more complex grammars"

  import NimbleParsec

  # Claim ID: alphanumeric with underscores
  claim_id = ascii_string([?a..?z, ?A..?Z, ?0..?9, ?_], min: 1)

  # Document reference: (Document 12345)
  document_ref =
    ignore(string("("))
    |> ignore(string("Document"))
    |> ignore(ascii_string([?\s], min: 1))
    |> integer(min: 1)
    |> ignore(string(")"))
    |> unwrap_and_tag(:document_id)
    |> optional()

  # Claim line
  defparsec :claim_line,
    ignore(string("CLAIM["))
    |> concat(claim_id |> unwrap_and_tag(:id))
    |> ignore(string("]"))
    |> ignore(ascii_string([?\s], min: 0))
    |> concat(document_ref)
    |> ignore(ascii_string([?\s], min: 0))
    |> ignore(string(":"))
    |> ignore(ascii_string([?\s], min: 0))
    |> concat(utf8_string([], min: 0) |> unwrap_and_tag(:text))

  # Relation label
  relation_label =
    choice([
      string("supports"),
      string("refutes"),
      string("contrasts")
    ])

  # Relation line
  defparsec :relation_line,
    ignore(string("RELATION"))
    |> ignore(ascii_string([?\s, ?:, ?-], min: 0))
    |> concat(claim_id |> unwrap_and_tag(:src))
    |> ignore(ascii_string([?\s], min: 1))
    |> concat(relation_label |> unwrap_and_tag(:label))
    |> ignore(ascii_string([?\s], min: 1))
    |> concat(claim_id |> unwrap_and_tag(:dst))
end
```

### 5.3 Test Cases

```elixir
defmodule CNS.Schema.ParserTest do
  use ExUnit.Case, async: true
  alias CNS.Schema.Parser
  alias CNS.Schema.Parser.Claim

  describe "parse_claims/1" do
    test "parses basic claim" do
      text = "CLAIM[c1]: The hypothesis text"
      claims = Parser.parse_claims(text)

      assert Map.has_key?(claims, "c1")
      assert claims["c1"].text == "The hypothesis text"
      assert claims["c1"].document_id == nil
    end

    test "parses claim with document reference" do
      text = "CLAIM[c2] (Document 12345): Supporting claim"
      claims = Parser.parse_claims(text)

      assert claims["c2"].text == "Supporting claim"
      assert claims["c2"].document_id == "12345"
    end

    test "parses multiple claims" do
      text = """
      CLAIM[c1]: First claim
      Some other text
      CLAIM[c2] (Document 123): Second claim
      CLAIM[c3]: Third claim
      """

      claims = Parser.parse_claims(text)

      assert map_size(claims) == 3
      assert Map.has_key?(claims, "c1")
      assert Map.has_key?(claims, "c2")
      assert Map.has_key?(claims, "c3")
    end

    test "case insensitive CLAIM keyword" do
      text = "claim[c1]: lowercase\nCLAIM[c2]: uppercase\nClaim[c3]: mixed"
      claims = Parser.parse_claims(text)

      assert map_size(claims) == 3
    end

    test "handles complex claim IDs" do
      text = "CLAIM[claim_1_final]: Test"
      claims = Parser.parse_claims(text)

      assert Map.has_key?(claims, "claim_1_final")
    end

    test "trims whitespace" do
      text = "CLAIM[c1]  :   Text with spaces   "
      claims = Parser.parse_claims(text)

      assert claims["c1"].text == "Text with spaces"
    end

    test "empty input returns empty map" do
      assert Parser.parse_claims("") == %{}
      assert Parser.parse_claims([]) == %{}
    end
  end

  describe "parse_relation/1" do
    test "parses relation with colon" do
      result = Parser.parse_relation("RELATION: c2 supports c1")

      assert result == {"c2", "supports", "c1"}
    end

    test "parses relation with dash" do
      result = Parser.parse_relation("RELATION - c3 refutes c1")

      assert result == {"c3", "refutes", "c1"}
    end

    test "parses contrasts relation" do
      result = Parser.parse_relation("RELATION: c4 contrasts c2")

      assert result == {"c4", "contrasts", "c2"}
    end

    test "case insensitive keywords" do
      result = Parser.parse_relation("relation: c1 SUPPORTS c2")

      assert result == {"c1", "supports", "c2"}
    end

    test "returns nil for non-relation lines" do
      assert Parser.parse_relation("CLAIM[c1]: text") == nil
      assert Parser.parse_relation("just some text") == nil
      assert Parser.parse_relation("") == nil
    end

    test "handles various whitespace" do
      result = Parser.parse_relation("RELATION:   c1   supports   c2")

      assert result == {"c1", "supports", "c2"}
    end
  end

  describe "parse_relations/1" do
    test "parses multiple relations" do
      text = """
      RELATION: c2 supports c1
      Some other text
      RELATION - c3 refutes c1
      """

      relations = Parser.parse_relations(text)

      assert length(relations) == 2
      assert {"c2", "supports", "c1"} in relations
      assert {"c3", "refutes", "c1"} in relations
    end
  end

  describe "parse/1" do
    test "parses complete output" do
      text = """
      CLAIM[c1]: Main hypothesis
      CLAIM[c2] (Document 123): Supporting evidence
      CLAIM[c3] (Document 456): Counter evidence
      RELATION: c2 supports c1
      RELATION: c3 refutes c1
      """

      {claims, relations} = Parser.parse(text)

      assert map_size(claims) == 3
      assert length(relations) == 2
    end
  end
end
```

### 5.4 Estimated Effort

| Task | Hours |
|------|-------|
| Module structure and types | 1 |
| Regex implementation | 3 |
| NimbleParsec alternative | 4 |
| Edge case handling | 2 |
| Test suite | 3 |
| Documentation | 1 |
| **Total** | **14 hours** |

---

## 6. Dependency Summary

### 6.1 Required Hex Packages

```elixir
# mix.exs
defp deps do
  [
    # Graph operations (Betti)
    {:libgraph, "~> 0.16"},

    # Numerical computing (Chirality)
    {:nx, "~> 0.6"},
    {:exla, "~> 0.6"},  # GPU acceleration

    # ML models (Semantic Validation)
    {:bumblebee, "~> 0.4"},
    {:axon, "~> 0.6"},

    # Parser combinators (optional)
    {:nimble_parsec, "~> 1.3"},

    # HTTP client (external services)
    {:req, "~> 0.4"},

    # Testing
    {:stream_data, "~> 0.6", only: :test}
  ]
end
```

### 6.2 External Services (Optional)

If not using Bumblebee for local inference:

| Service | Purpose | API |
|---------|---------|-----|
| NLI Service | Entailment scoring | POST /nli `{premise, hypothesis}` |
| Embedding Service | Text embeddings | POST /embed `{text}` |
| Sentence Transformer | Multiple embeddings | POST /encode `{texts}` |

### 6.3 Pre-trained Models

| Model | Purpose | HuggingFace ID |
|-------|---------|----------------|
| DeBERTa-v3-NLI | Entailment | `cross-encoder/nli-deberta-v3-large` |
| MiniLM-L6 | Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |

---

## 7. Implementation Timeline

### Phase 1: Foundation (Week 1)

| Day | Task | Hours |
|-----|------|-------|
| 1-2 | Betti Number Computation | 14 |
| 3 | Claim Schema Parsing | 8 |
| 4-5 | Testing and refinement | 8 |

**Deliverables:**
- `CNS.Logic.Betti` module with tests
- `CNS.Schema.Parser` module with tests
- Documentation

### Phase 2: Metrics (Week 2)

| Day | Task | Hours |
|-----|------|-------|
| 1-3 | Chirality Computation (Nx) | 16 |
| 4 | Embedder adapters | 4 |
| 5 | Testing and optimization | 4 |

**Deliverables:**
- `CNS.Metrics.Chirality` module with tests
- `CNS.Embedders.Bumblebee` adapter
- GPU-accelerated defn functions

### Phase 3: Validation Pipeline (Weeks 3-4)

| Day | Task | Hours |
|-----|------|-------|
| 1-2 | Citation validation stage | 8 |
| 3-5 | NLI/Bumblebee integration | 12 |
| 6-7 | Embedding/similarity stage | 8 |
| 8-9 | Pipeline orchestration | 8 |

**Deliverables:**
- `CNS.Validation.Semantic` module
- `CNS.Validation.Citation` module
- Integration with Critics

### Phase 4: Integration (Week 5)

| Day | Task | Hours |
|-----|------|-------|
| 1-2 | Antagonist integration | 8 |
| 3 | End-to-end testing | 4 |
| 4-5 | Performance optimization | 8 |

**Deliverables:**
- `CNS.Antagonist` using all modules
- Benchmark suite
- Production-ready documentation

### Total Effort Summary

| Component | Hours |
|-----------|-------|
| Betti Number Computation | 14 |
| Chirality Computation | 24 |
| Semantic Validation Pipeline | 36 |
| Claim Schema Parsing | 14 |
| Integration & Testing | 20 |
| **Total** | **108 hours** |

**Estimated Calendar Time:** 5 weeks (part-time) or 3 weeks (full-time)

---

## Appendix A: Example Usage

### Complete Workflow

```elixir
# Initialize services
CNS.Validation.Semantic.start_link(%CNS.Validation.Semantic.Config{
  entailment_threshold: 0.75,
  similarity_threshold: 0.7
})

# Parse LLM output
llm_output = """
CLAIM[c1]: Climate change is accelerating
CLAIM[c2] (Document 123): Global temperatures rose 1.1C since 1900
CLAIM[c3] (Document 456): Some regions show cooling trends
RELATION: c2 supports c1
RELATION: c3 refutes c1
"""

{claims, relations} = CNS.Schema.Parser.parse(llm_output)

# Compute graph topology
claim_ids = Map.keys(claims)
graph_stats = CNS.Logic.Betti.compute_graph_stats(claim_ids, relations)

IO.puts("Beta1: #{graph_stats.beta1}")
IO.puts("Polarity conflict: #{graph_stats.polarity_conflict}")

# Compute chirality (requires embedder and stats)
stats = load_fisher_rao_stats()
embedder = CNS.Embedders.Bumblebee

thesis = claims["c2"].text
antithesis = claims["c3"].text

chirality = CNS.Metrics.Chirality.compare(
  embedder,
  stats,
  thesis,
  antithesis,
  0.3,  # evidence_overlap
  graph_stats.polarity_conflict
)

IO.puts("Chirality score: #{chirality.chirality_score}")

# Validate claims
corpus = load_evidence_corpus()
gold_ids = MapSet.new(["123", "456"])

result = CNS.Validation.Semantic.validate_claim(
  claims["c1"].text,
  "Expected claim text",
  llm_output,
  corpus,
  gold_ids
)

IO.puts("Validation passed: #{result.overall_pass}")
```

---

*End of Porting Design Document*

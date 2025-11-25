defmodule CNS.Topology.Persistence do
  @moduledoc """
  Persistent homology computation for CNS claim networks.

  This module provides production-grade topological data analysis using
  ex_topology's persistent homology pipeline. It interprets topological
  features in the context of dialectical reasoning:

  - **H₀ (Dimension 0)**: Claim clusters and their stability
  - **H₁ (Dimension 1)**: Circular reasoning patterns
  - **H₂ (Dimension 2)**: Higher-order logical structures

  ## Persistent Homology Overview

  Persistent homology tracks topological features across multiple scales,
  recording when each feature appears (birth) and disappears (death).
  Features with high persistence (death - birth) are considered structurally
  significant rather than noise.

  ## Integration

  This module depends on `CNS.Topology.Adapter` for converting SNOs to
  embeddings and building filtrations suitable for topological analysis.

  ## Examples

      # Compute persistence for a claim network
      snos = [sno1, sno2, sno3]
      result = CNS.Topology.Persistence.compute(snos)

      # Check for circular reasoning
      if result.circular_reasoning.detected_cycles > 0 do
        IO.puts("Warning: Circular reasoning detected")
      end

      # Compare two claim networks
      comp = CNS.Topology.Persistence.compare(thesis_snos, antithesis_snos)
      IO.puts("Topological distance: \#{comp.total_distance}")
  """

  alias CNS.SNO
  alias ExTopology.Diagram

  require Logger

  @type persistence_result :: %{
          cluster_analysis: cluster_analysis(),
          circular_reasoning: circular_reasoning(),
          higher_order: higher_order(),
          diagrams: [Diagram.diagram()],
          summary: summary()
        }

  @type cluster_analysis :: %{
          total_clusters: non_neg_integer(),
          persistent_clusters: non_neg_integer(),
          cluster_stability: float(),
          cluster_entropy: float()
        }

  @type circular_reasoning :: %{
          detected_cycles: non_neg_integer(),
          persistent_cycles: non_neg_integer(),
          cycle_severity: float(),
          max_cycle_persistence: float(),
          interpretation: :none | :weak | :moderate | :severe
        }

  @type higher_order :: %{
          voids: non_neg_integer(),
          complexity: float(),
          max_void_persistence: float()
        }

  @type summary :: %{
          total_features: non_neg_integer(),
          significant_features: non_neg_integer(),
          overall_complexity: float(),
          topological_robustness: float()
        }

  @type comparison_result :: %{
          distances: [dimension_distance()],
          total_distance: float(),
          wasserstein_distances: [dimension_distance()],
          topologically_similar?: boolean(),
          interpretation: String.t()
        }

  @type dimension_distance :: %{
          dimension: non_neg_integer(),
          bottleneck: float(),
          wasserstein: float()
        }

  @default_max_dimension 2
  @default_max_epsilon 2.0
  @default_persistence_threshold 0.3
  @default_similarity_threshold 0.5

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Compute persistent homology for a claim network.

  Builds a Vietoris-Rips filtration from claim embeddings and computes
  persistence diagrams for dimensions 0 through `max_dimension`. Results
  are interpreted in the CNS context (clusters, cycles, voids).

  ## Parameters

  - `snos` - List of SNO structs representing the claim network
  - `opts` - Keyword list:
    - `:max_dimension` - Maximum homology dimension (default: 2)
    - `:max_epsilon` - Maximum filtration scale (default: 2.0)
    - `:persistence_threshold` - Minimum persistence to count as significant (default: 0.3)
    - `:adapter_opts` - Options passed to Adapter.sno_embeddings/2

  ## Returns

  - `persistence_result()` map with cluster analysis, circular reasoning
    detection, higher-order features, raw diagrams, and summary statistics

  ## Examples

      iex> snos = [
      ...>   CNS.SNO.new("Claim A", id: "1"),
      ...>   CNS.SNO.new("Claim B", id: "2"),
      ...>   CNS.SNO.new("Claim C", id: "3")
      ...> ]
      iex> result = CNS.Topology.Persistence.compute(snos)
      iex> result.cluster_analysis.total_clusters >= 0
      true

  ## Interpretation

  - **High cluster_stability**: Well-separated claim groups
  - **persistent_cycles > 0**: Circular reasoning detected
  - **High cycle_severity**: Strong circular dependencies
  - **voids > 0**: Complex higher-dimensional structures
  """
  @spec compute([SNO.t()], keyword()) :: persistence_result()
  def compute(snos, opts \\ []) when is_list(snos) do
    max_dim = Keyword.get(opts, :max_dimension, @default_max_dimension)
    max_epsilon = Keyword.get(opts, :max_epsilon, @default_max_epsilon)
    pers_threshold = Keyword.get(opts, :persistence_threshold, @default_persistence_threshold)
    adapter_opts = Keyword.get(opts, :adapter_opts, [])

    Logger.debug("Computing persistent homology for #{length(snos)} claims")

    # Convert SNOs to embeddings via Adapter
    embeddings = CNS.Topology.Adapter.sno_embeddings(snos, adapter_opts)

    # Build Vietoris-Rips filtration
    filtration =
      ExTopology.Filtration.vietoris_rips(embeddings,
        max_dimension: max_dim,
        max_epsilon: max_epsilon
      )

    Logger.debug("Built filtration with #{length(filtration)} simplices")

    # Compute persistence
    diagrams = ExTopology.Persistence.compute(filtration, max_dimension: max_dim)

    # Interpret results in CNS context
    interpret_diagrams(diagrams, snos, pers_threshold)
  end

  @doc """
  Compute persistent homology from pre-computed embeddings.

  Useful when embeddings are already available (e.g., cached) or when
  working directly with embedding tensors without SNO conversion.

  ## Parameters

  - `embeddings` - Nx tensor of shape `{n, d}` (n claims, d dimensions)
  - `opts` - Keyword list (same as `compute/2`)

  ## Returns

  - Map with persistence diagrams and basic statistics (no SNO-specific interpretation)

  ## Examples

      iex> embeddings = Nx.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
      iex> result = CNS.Topology.Persistence.compute_from_embeddings(embeddings)
      iex> is_list(result.diagrams)
      true
  """
  @spec compute_from_embeddings(Nx.Tensor.t(), keyword()) :: %{
          diagrams: [Diagram.diagram()],
          statistics: map()
        }
  def compute_from_embeddings(embeddings, opts \\ []) do
    max_dim = Keyword.get(opts, :max_dimension, @default_max_dimension)
    max_epsilon = Keyword.get(opts, :max_epsilon, @default_max_epsilon)

    filtration =
      ExTopology.Filtration.vietoris_rips(embeddings,
        max_dimension: max_dim,
        max_epsilon: max_epsilon
      )

    diagrams = ExTopology.Persistence.compute(filtration, max_dimension: max_dim)

    %{
      diagrams: diagrams,
      statistics: compute_diagram_statistics(diagrams)
    }
  end

  @doc """
  Compare topological structure of two claim networks.

  Computes bottleneck and Wasserstein distances between persistence diagrams
  for each dimension. Lower distances indicate more similar topological
  structures.

  ## Parameters

  - `snos1` - First claim network (list of SNOs)
  - `snos2` - Second claim network (list of SNOs)
  - `opts` - Keyword list:
    - `:max_dimension` - Maximum dimension (default: 2)
    - `:similarity_threshold` - Bottleneck threshold for similarity (default: 0.5)
    - `:wasserstein_p` - Wasserstein distance power (default: 2)

  ## Returns

  - `comparison_result()` with per-dimension distances and similarity assessment

  ## Examples

      iex> thesis = [CNS.SNO.new("Thesis 1"), CNS.SNO.new("Thesis 2")]
      iex> antithesis = [CNS.SNO.new("Anti 1"), CNS.SNO.new("Anti 2")]
      iex> comparison = CNS.Topology.Persistence.compare(thesis, antithesis)
      iex> comparison.total_distance >= 0.0
      true

  ## Interpretation

  - **total_distance < 0.5**: Networks are topologically very similar
  - **total_distance 0.5-1.5**: Moderate topological differences
  - **total_distance > 1.5**: Networks have fundamentally different structures
  - **topologically_similar? = true**: All dimensions below similarity threshold
  """
  @spec compare([SNO.t()], [SNO.t()], keyword()) :: comparison_result()
  def compare(snos1, snos2, opts \\ []) do
    max_dim = Keyword.get(opts, :max_dimension, @default_max_dimension)
    similarity_threshold = Keyword.get(opts, :similarity_threshold, @default_similarity_threshold)
    wasserstein_p = Keyword.get(opts, :wasserstein_p, 2)

    Logger.debug("Comparing #{length(snos1)} vs #{length(snos2)} claims")

    # Compute persistence for both networks
    result1 = compute(snos1, Keyword.put(opts, :max_dimension, max_dim))
    result2 = compute(snos2, Keyword.put(opts, :max_dimension, max_dim))

    # Compute distances per dimension
    distances =
      Enum.zip(result1.diagrams, result2.diagrams)
      |> Enum.map(fn {d1, d2} ->
        bottleneck = ExTopology.Diagram.bottleneck_distance(d1, d2)
        wasserstein = ExTopology.Diagram.wasserstein_distance(d1, d2, p: wasserstein_p)

        %{
          dimension: d1.dimension,
          bottleneck: bottleneck,
          wasserstein: wasserstein
        }
      end)

    total_distance = Enum.sum(Enum.map(distances, & &1.bottleneck))
    similar? = Enum.all?(distances, &(&1.bottleneck < similarity_threshold))

    %{
      distances: distances,
      total_distance: total_distance,
      wasserstein_distances: distances,
      topologically_similar?: similar?,
      interpretation: interpret_comparison(total_distance, similar?)
    }
  end

  @doc """
  Compare a claim network against a baseline (e.g., ideal structure).

  Useful for evaluating synthesis quality against known good structures.

  ## Parameters

  - `snos` - Claim network to evaluate
  - `baseline_embeddings` - Pre-computed embeddings for baseline structure
  - `opts` - Keyword list (same as `compare/3`)

  ## Returns

  - `comparison_result()` with distance from baseline

  ## Examples

      iex> snos = [CNS.SNO.new("Test")]
      iex> baseline = Nx.tensor([[0.0, 0.0], [1.0, 1.0]])
      iex> result = CNS.Topology.Persistence.compare_to_baseline(snos, baseline)
      iex> is_float(result.total_distance)
      true
  """
  @spec compare_to_baseline([SNO.t()], Nx.Tensor.t(), keyword()) :: comparison_result()
  def compare_to_baseline(snos, baseline_embeddings, opts \\ []) do
    max_dim = Keyword.get(opts, :max_dimension, @default_max_dimension)

    result1 = compute(snos, opts)

    result2 =
      compute_from_embeddings(baseline_embeddings, Keyword.put(opts, :max_dimension, max_dim))

    distances =
      Enum.zip(result1.diagrams, result2.diagrams)
      |> Enum.map(fn {d1, d2} ->
        %{
          dimension: d1.dimension,
          bottleneck: ExTopology.Diagram.bottleneck_distance(d1, d2),
          wasserstein:
            ExTopology.Diagram.wasserstein_distance(d1, d2, p: Keyword.get(opts, :wasserstein_p, 2))
        }
      end)

    total_distance = Enum.sum(Enum.map(distances, & &1.bottleneck))

    similar? =
      Enum.all?(
        distances,
        &(&1.bottleneck < Keyword.get(opts, :similarity_threshold, @default_similarity_threshold))
      )

    %{
      distances: distances,
      total_distance: total_distance,
      wasserstein_distances: distances,
      topologically_similar?: similar?,
      interpretation: "Distance from baseline: #{Float.round(total_distance, 3)}"
    }
  end

  @doc """
  Extract persistence barcodes for visualization.

  Returns birth-death pairs formatted for plotting.

  ## Parameters

  - `result` - Output from `compute/2`
  - `dimension` - Homology dimension to extract (default: 1)

  ## Returns

  - List of `{birth, death}` tuples

  ## Examples

      iex> snos = [CNS.SNO.new("A"), CNS.SNO.new("B")]
      iex> result = CNS.Topology.Persistence.compute(snos)
      iex> barcodes = CNS.Topology.Persistence.barcodes(result, 1)
      iex> is_list(barcodes)
      true
  """
  @spec barcodes(persistence_result(), non_neg_integer()) :: [
          {float(), float() | :infinity}
        ]
  def barcodes(%{diagrams: diagrams}, dimension \\ 1) do
    diagram = Enum.find(diagrams, &(&1.dimension == dimension))

    if diagram do
      diagram.pairs
    else
      []
    end
  end

  @doc """
  Get summary statistics for persistence result.

  ## Parameters

  - `result` - Output from `compute/2`

  ## Returns

  - Map with summary statistics

  ## Examples

      iex> snos = [CNS.SNO.new("Test")]
      iex> result = CNS.Topology.Persistence.compute(snos)
      iex> summary = CNS.Topology.Persistence.summary(result)
      iex> Map.has_key?(summary, :total_features)
      true
  """
  @spec summary(persistence_result()) :: summary()
  def summary(%{summary: summary}), do: summary

  @doc """
  Check if claim network exhibits circular reasoning.

  Convenience function that checks H₁ persistence for significant cycles.

  ## Parameters

  - `snos` - List of SNO structs
  - `opts` - Keyword list (passed to `compute/2`)

  ## Returns

  - Boolean indicating presence of circular reasoning

  ## Examples

      iex> snos = [CNS.SNO.new("A"), CNS.SNO.new("B")]
      iex> CNS.Topology.Persistence.has_circular_reasoning?(snos)
      false
  """
  @spec has_circular_reasoning?([SNO.t()], keyword()) :: boolean()
  def has_circular_reasoning?(snos, opts \\ []) do
    result = compute(snos, opts)
    result.circular_reasoning.persistent_cycles > 0
  end

  @doc """
  Compute topological complexity score (0.0 - 1.0).

  Higher scores indicate more complex topological structure.

  ## Parameters

  - `snos` - List of SNO structs
  - `opts` - Keyword list (passed to `compute/2`)

  ## Returns

  - Float between 0.0 and 1.0

  ## Examples

      iex> snos = [CNS.SNO.new("A"), CNS.SNO.new("B"), CNS.SNO.new("C")]
      iex> complexity = CNS.Topology.Persistence.complexity_score(snos)
      iex> complexity >= 0.0 and complexity <= 1.0
      true
  """
  @spec complexity_score([SNO.t()], keyword()) :: float()
  def complexity_score(snos, opts \\ []) do
    result = compute(snos, opts)
    result.summary.overall_complexity
  end

  # ============================================================================
  # Private Helper Functions
  # ============================================================================

  # Interpret persistence diagrams in CNS context
  @spec interpret_diagrams([Diagram.diagram()], [SNO.t()], float()) ::
          persistence_result()
  defp interpret_diagrams(diagrams, snos, pers_threshold) do
    h0 = Enum.find(diagrams, &(&1.dimension == 0)) || %{dimension: 0, pairs: []}
    h1 = Enum.find(diagrams, &(&1.dimension == 1)) || %{dimension: 1, pairs: []}
    h2 = Enum.find(diagrams, &(&1.dimension == 2)) || %{dimension: 2, pairs: []}

    cluster_analysis = analyze_clusters(h0, snos, pers_threshold)
    circular_reasoning = analyze_cycles(h1, snos, pers_threshold)
    higher_order = analyze_higher_order(h2, pers_threshold)

    summary =
      build_summary(cluster_analysis, circular_reasoning, higher_order, diagrams, pers_threshold)

    %{
      cluster_analysis: cluster_analysis,
      circular_reasoning: circular_reasoning,
      higher_order: higher_order,
      diagrams: diagrams,
      summary: summary
    }
  end

  # Analyze H₀ (connected components / claim clusters)
  @spec analyze_clusters(Diagram.diagram(), [SNO.t()], float()) :: cluster_analysis()
  defp analyze_clusters(h0, _snos, pers_threshold) do
    total_clusters = length(h0.pairs)
    persistent_pairs = ExTopology.Diagram.filter_by_persistence(h0, min: pers_threshold)
    persistent_clusters = length(persistent_pairs.pairs)

    cluster_stability = ExTopology.Diagram.total_persistence(h0)
    cluster_entropy = ExTopology.Diagram.entropy(h0)

    %{
      total_clusters: total_clusters,
      persistent_clusters: persistent_clusters,
      cluster_stability: Float.round(cluster_stability, 4),
      cluster_entropy: Float.round(cluster_entropy, 4)
    }
  end

  # Analyze H₁ (cycles / circular reasoning)
  @spec analyze_cycles(Diagram.diagram(), [SNO.t()], float()) :: circular_reasoning()
  defp analyze_cycles(h1, _snos, pers_threshold) do
    detected_cycles = length(h1.pairs)
    persistent_pairs = ExTopology.Diagram.filter_by_persistence(h1, min: pers_threshold)
    persistent_cycles = length(persistent_pairs.pairs)

    stats = ExTopology.Diagram.summary_statistics(h1)
    max_persistence = stats.max_persistence
    cycle_severity = stats.total_persistence

    interpretation = interpret_cycle_severity(persistent_cycles, max_persistence)

    %{
      detected_cycles: detected_cycles,
      persistent_cycles: persistent_cycles,
      cycle_severity: Float.round(cycle_severity, 4),
      max_cycle_persistence: Float.round(max_persistence, 4),
      interpretation: interpretation
    }
  end

  # Analyze H₂ (voids / higher-order structures)
  @spec analyze_higher_order(Diagram.diagram(), float()) :: higher_order()
  defp analyze_higher_order(h2, _pers_threshold) do
    voids = length(h2.pairs)
    complexity = ExTopology.Diagram.entropy(h2)
    stats = ExTopology.Diagram.summary_statistics(h2)

    %{
      voids: voids,
      complexity: Float.round(complexity, 4),
      max_void_persistence: Float.round(stats.max_persistence, 4)
    }
  end

  # Build summary statistics
  @spec build_summary(
          cluster_analysis(),
          circular_reasoning(),
          higher_order(),
          [Diagram.diagram()],
          float()
        ) ::
          summary()
  defp build_summary(clusters, cycles, higher, diagrams, pers_threshold) do
    # Count all features across dimensions
    total_features =
      Enum.sum(Enum.map(diagrams, fn d -> length(d.pairs) end))

    # Count significant features (persistence > threshold)
    significant_features =
      diagrams
      |> Enum.map(fn d ->
        ExTopology.Diagram.filter_by_persistence(d, min: pers_threshold)
        |> Map.get(:pairs)
        |> length()
      end)
      |> Enum.sum()

    # Overall complexity: weighted average of entropies
    overall_complexity =
      (clusters.cluster_entropy * 0.3 +
         cycles.cycle_severity * 0.5 +
         higher.complexity * 0.2)
      |> min(1.0)
      |> Float.round(4)

    # Robustness: ratio of significant to total features
    topological_robustness =
      if total_features > 0 do
        Float.round(significant_features / total_features, 4)
      else
        0.0
      end

    %{
      total_features: total_features,
      significant_features: significant_features,
      overall_complexity: overall_complexity,
      topological_robustness: topological_robustness
    }
  end

  # Interpret cycle severity
  @spec interpret_cycle_severity(non_neg_integer(), float()) ::
          :none | :weak | :moderate | :severe
  defp interpret_cycle_severity(0, _), do: :none
  defp interpret_cycle_severity(_, max_pers) when max_pers < 0.3, do: :weak
  defp interpret_cycle_severity(_, max_pers) when max_pers < 0.6, do: :moderate
  defp interpret_cycle_severity(_, _), do: :severe

  # Interpret comparison results
  @spec interpret_comparison(float(), boolean()) :: String.t()
  defp interpret_comparison(distance, similar?) do
    cond do
      similar? and distance < 0.3 ->
        "Networks are topologically nearly identical"

      similar? ->
        "Networks are topologically similar with minor structural differences"

      distance < 1.0 ->
        "Networks have moderate topological differences"

      distance < 2.0 ->
        "Networks have substantial topological differences"

      true ->
        "Networks have fundamentally different topological structures"
    end
  end

  # Compute basic statistics for diagrams
  @spec compute_diagram_statistics([Diagram.diagram()]) :: map()
  defp compute_diagram_statistics(diagrams) do
    diagrams
    |> Enum.map(fn d ->
      stats = ExTopology.Diagram.summary_statistics(d)
      {d.dimension, stats}
    end)
    |> Map.new()
  end
end

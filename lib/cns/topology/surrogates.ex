defmodule CNS.Topology.Surrogates do
  @moduledoc """
  Lightweight surrogate computations for topological validation.

  These surrogates approximate expensive persistent homology computations
  with O(V+E) graph algorithms, enabling fast validation of the
  topology-logic correlation hypothesis before investing in full TDA.

  ## Surrogates

  - `beta1_surrogate` - Approximates β₁ (first Betti number) using cycle
    detection on the causal link graph. High β₁ indicates circular reasoning.

  - `fragility_surrogate` - Approximates semantic instability using local
    embedding variance. High fragility indicates brittle, easily-perturbed claims.

  ## Validation Protocol

  Per the roadmap (20251121_ROADMAP.md), surrogates must achieve:
  - Graph β₁ vs human-annotated circularity: r > 0.35
  - Embedding fragility vs perturbation sensitivity: 2× differential

  If surrogates fail validation, full TDA implementation is not justified.

  ## Examples

      iex> graph = %{"a" => ["b"], "b" => ["c"], "c" => ["a"]}
      iex> CNS.Topology.Surrogates.compute_beta1_surrogate(graph)
      1

      iex> embeddings = [[0.1, 0.2], [0.15, 0.25], [0.8, 0.9]]
      iex> fragility = CNS.Topology.Surrogates.compute_fragility_surrogate(embeddings)
      iex> fragility >= 0.0 and fragility <= 1.0
      true
  """

  alias CNS.Topology.Adapter
  alias ExTopology.Embedding
  alias ExTopology.Graph, as: TopoGraph
  alias Graph

  @doc """
  Compute β₁ surrogate using cycle detection on causal link graph.

  Approximates the first Betti number (number of independent cycles)
  using graph cycle detection. This is an O(V+E) approximation of the
  expensive persistent homology computation.

  ## Parameters

    - `graph` - Map representing directed graph where keys are nodes
                and values are lists of child nodes

  ## Returns

  Non-negative integer representing the number of independent cycles.

  ## Examples

      iex> CNS.Topology.Surrogates.compute_beta1_surrogate(%{})
      0

      iex> CNS.Topology.Surrogates.compute_beta1_surrogate(%{"a" => ["b"], "b" => []})
      0

      iex> CNS.Topology.Surrogates.compute_beta1_surrogate(%{"a" => ["b"], "b" => ["a"]})
      1
  """
  @spec compute_beta1_surrogate(map() | Graph.t()) :: non_neg_integer()
  def compute_beta1_surrogate(graph_like) do
    graph = CNS.Topology.build_graph(graph_like)

    if Graph.is_acyclic?(graph) do
      0
    else
      TopoGraph.beta_one(graph)
    end
  end

  @doc """
  Compute fragility surrogate using embedding variance in local neighborhoods.

  High variance in the k-nearest neighbor embeddings indicates semantic
  instability - the claim's meaning is fragile and easily perturbed.

  ## Parameters

    - `embeddings` - List of embedding vectors (list of lists of floats) or
                     Nx tensor of shape {n_samples, embedding_dim}
    - `opts` - Options:
      - `:k` - Number of neighbors to consider (default: 5)
      - `:metric` - Distance metric (:euclidean or :cosine, default: :cosine)

  ## Returns

  Float in [0, 1] representing fragility score. Higher values indicate
  more fragile/unstable embeddings.

  ## Examples

      iex> embeddings = [[0.5, 0.5], [0.51, 0.49], [0.48, 0.52]]
      iex> fragility = CNS.Topology.Surrogates.compute_fragility_surrogate(embeddings)
      iex> fragility < 0.3  # Low fragility for similar embeddings
      true

      iex> embeddings = [[0.1, 0.1], [0.9, 0.9], [0.1, 0.9]]
      iex> fragility = CNS.Topology.Surrogates.compute_fragility_surrogate(embeddings)
      iex> fragility > 0.5  # High fragility for diverse embeddings
      true
  """
  @spec compute_fragility_surrogate(list(list(float())) | Nx.Tensor.t(), keyword()) :: float()
  def compute_fragility_surrogate(embeddings, opts \\ [])
  def compute_fragility_surrogate([], _opts), do: 0.0

  def compute_fragility_surrogate(embeddings, opts) do
    k = Keyword.get(opts, :k, 5)
    metric = Keyword.get(opts, :metric, :euclidean)

    tensor = Adapter.to_tensor(embeddings, type: :f32)
    {n_samples, _dim} = Nx.shape(tensor)

    cond do
      n_samples <= 1 ->
        0.0

      true ->
        neighbors = min(k, max(n_samples - 1, 1))

        knn_dists =
          Embedding.knn_distances(tensor,
            k: neighbors,
            metric: metric
          )

        variance =
          knn_dists
          |> Nx.variance(axes: [1])
          |> Nx.mean()
          |> Nx.to_number()

        mean_distance =
          knn_dists
          |> Nx.mean()
          |> Nx.to_number()

        normalize_variance(variance + mean_distance)
    end
  end

  @doc """
  Compute both surrogates for a structured narrative object.

  Extracts the causal graph and embeddings from an SNO and computes
  both the β₁ and fragility surrogates.

  ## Parameters

    - `sno` - A map containing:
      - `:causal_links` - List of {source, target} tuples
      - `:embeddings` - List of embedding vectors
    - `opts` - Options passed to compute_fragility_surrogate

  ## Returns

  Map with `:beta1` and `:fragility` scores.

  ## Examples

      iex> sno = %{
      ...>   causal_links: [{"a", "b"}, {"b", "c"}],
      ...>   embeddings: [[0.1, 0.2], [0.15, 0.25]]
      ...> }
      iex> scores = CNS.Topology.Surrogates.compute_surrogates(sno)
      iex> Map.has_key?(scores, :beta1) and Map.has_key?(scores, :fragility)
      true
  """
  @spec compute_surrogates(map(), keyword()) :: %{beta1: non_neg_integer(), fragility: float()}
  def compute_surrogates(sno, opts \\ []) do
    beta1 = compute_beta1_from_links(Map.get(sno, :causal_links, []))

    fragility =
      sno
      |> Map.get(:embeddings, [])
      |> compute_fragility_surrogate(opts)

    %{
      beta1: beta1,
      fragility: Float.round(fragility, 4)
    }
  end

  @doc """
  Validate surrogate correlation with ground truth labels.

  Computes correlation between surrogate scores and human annotations
  to determine if surrogates are predictive of logical validity.

  ## Parameters

    - `examples` - List of maps with :surrogates and :label fields
    - `opts` - Options:
      - `:metric` - Correlation metric (:pearson or :spearman, default: :pearson)

  ## Returns

  Map with correlation results and p-values.

  ## Examples

      iex> examples = [
      ...>   %{surrogates: %{beta1: 0, fragility: 0.1}, label: 0},
      ...>   %{surrogates: %{beta1: 1, fragility: 0.8}, label: 1}
      ...> ]
      iex> results = CNS.Topology.Surrogates.validate_correlation(examples)
      iex> Map.has_key?(results, :beta1_correlation)
      true
  """
  @spec validate_correlation(list(map()), keyword()) :: map()
  def validate_correlation(examples, opts \\ []) do
    metric = Keyword.get(opts, :metric, :pearson)

    # Extract features and labels
    beta1_values = Enum.map(examples, fn ex -> ex.surrogates.beta1 end)
    fragility_values = Enum.map(examples, fn ex -> ex.surrogates.fragility end)
    labels = Enum.map(examples, fn ex -> ex.label end)

    # Compute correlations
    beta1_corr = compute_correlation(beta1_values, labels, metric)
    fragility_corr = compute_correlation(fragility_values, labels, metric)

    # Compute combined score correlation
    combined_scores =
      Enum.zip([beta1_values, fragility_values])
      |> Enum.map(fn {b1, frag} -> b1 * 0.5 + frag * 0.5 end)

    combined_corr = compute_correlation(combined_scores, labels, metric)

    %{
      beta1_correlation: beta1_corr.correlation,
      beta1_p_value: beta1_corr.p_value,
      fragility_correlation: fragility_corr.correlation,
      fragility_p_value: fragility_corr.p_value,
      combined_correlation: combined_corr.correlation,
      combined_p_value: combined_corr.p_value,
      n_samples: length(examples),
      passes_gate1: beta1_corr.correlation > 0.35
    }
  end

  # Private functions

  defp compute_beta1_from_links(links) do
    graph =
      Enum.reduce(links, Graph.new(type: :directed), fn {source, target}, g ->
        g
        |> Graph.add_vertex(source)
        |> Graph.add_vertex(target)
        |> Graph.add_edge(source, target)
      end)

    compute_beta1_surrogate(graph)
  end

  defp normalize_variance(value) when is_number(value) do
    scaled = value * 2.5
    score = scaled / (scaled + 1.0)
    score |> min(1.0) |> max(0.0) |> Float.round(4)
  end

  defp compute_correlation(x_values, y_values, :pearson) do
    n = length(x_values)

    if n < 2 do
      %{correlation: 0.0, p_value: 1.0}
    else
      x_tensor = Nx.tensor(x_values, type: :f32)
      y_tensor = Nx.tensor(y_values, type: :f32)

      x_mean = Nx.mean(x_tensor)
      y_mean = Nx.mean(y_tensor)

      x_centered = Nx.subtract(x_tensor, x_mean)
      y_centered = Nx.subtract(y_tensor, y_mean)

      numerator = Nx.sum(Nx.multiply(x_centered, y_centered))

      x_std = Nx.sqrt(Nx.sum(Nx.multiply(x_centered, x_centered)))
      y_std = Nx.sqrt(Nx.sum(Nx.multiply(y_centered, y_centered)))

      denom_value = Nx.multiply(x_std, y_std) |> Nx.to_number()

      correlation =
        cond do
          denom_value == 0.0 -> 0.0
          true -> Nx.divide(numerator, denom_value) |> Nx.to_number()
        end
        |> then(&max(min(&1, 1.0), -1.0))

      p_value =
        if n < 3 or abs(correlation) >= 0.9999 do
          0.0
        else
          t_stat = correlation * :math.sqrt((n - 2) / max(1.0e-6, 1 - correlation * correlation))
          if abs(t_stat) < 2.0, do: 0.1, else: 0.01
        end

      %{
        correlation: Float.round(correlation, 4),
        p_value: Float.round(p_value, 4)
      }
    end
  end

  defp compute_correlation(x_values, y_values, :spearman) do
    # Compute Spearman correlation (rank correlation)
    x_ranks = compute_ranks(x_values)
    y_ranks = compute_ranks(y_values)

    compute_correlation(x_ranks, y_ranks, :pearson)
  end

  defp compute_ranks(values) do
    values
    |> Enum.with_index()
    |> Enum.sort_by(&elem(&1, 0))
    |> Enum.with_index(1)
    |> Enum.sort_by(fn {{_, orig_idx}, _} -> orig_idx end)
    |> Enum.map(fn {{_, _}, rank} -> rank end)
  end
end

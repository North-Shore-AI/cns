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

  require Logger

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
  @spec compute_beta1_surrogate(map()) :: non_neg_integer()
  def compute_beta1_surrogate(graph) when is_map(graph) do
    # β₁ surrogate using cyclomatic number (edges - nodes + components).
    # This is monotonic with respect to adding edges and correctly counts
    # self-loops and multiple disjoint cycles.
    if map_size(graph) == 0 do
      0
    else
      {nodes, edge_count} =
        Enum.reduce(graph, {MapSet.new(), 0}, fn {node, children}, {node_acc, edge_acc} ->
          updated_nodes =
            children
            |> Enum.reduce(MapSet.put(node_acc, node), fn child, acc -> MapSet.put(acc, child) end)

          {updated_nodes, edge_acc + length(children)}
        end)

      component_count = count_components(graph, nodes)

      if has_cycle?(graph, nodes) do
        beta1 = edge_count - MapSet.size(nodes) + component_count
        max(beta1, 0)
      else
        0
      end
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

    metric =
      case Keyword.get(opts, :metric, :euclidean) do
        :cosine -> :cosine
        :euclidean -> :euclidean
        _ -> :euclidean
      end

    tensor = ensure_tensor(embeddings)
    {n_samples, _dim} = Nx.shape(tensor)

    cond do
      n_samples <= 1 ->
        0.0

      n_samples <= k ->
        compute_knn_variance(tensor, max(n_samples - 1, 1), metric)

      true ->
        compute_knn_variance(tensor, min(k, n_samples - 1), metric)
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
      case Map.get(sno, :embeddings) do
        nil -> 0.0
        [] -> 0.0
        embeddings -> compute_fragility_surrogate(embeddings, opts)
      end

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
    # Build graph from causal links
    graph =
      links
      |> Enum.reduce(%{}, fn {source, target}, acc ->
        acc
        |> Map.update(source, [target], &[target | &1])
        |> Map.put_new(target, [])
      end)

    compute_beta1_surrogate(graph)
  end

  @spec count_components(%{optional(any()) => [any()]}, MapSet.t(any())) :: non_neg_integer()
  defp count_components(graph, %MapSet{} = nodes) do
    adjacency =
      Enum.reduce(graph, %{}, fn {node, children}, acc ->
        acc
        |> Map.update(node, Enum.uniq(children), fn existing ->
          existing
          |> Kernel.++(children)
          |> Enum.uniq()
        end)
        |> add_reverse_edges(node, children)
      end)
      |> ensure_all_nodes(nodes)

    walk_components(%{}, Map.keys(adjacency), adjacency, 0)
  end

  defp add_reverse_edges(acc, _node, []), do: acc

  defp add_reverse_edges(acc, node, [child | rest]) do
    updated =
      Map.update(acc, child, [node], fn existing ->
        [node | existing] |> Enum.uniq()
      end)

    add_reverse_edges(updated, node, rest)
  end

  defp ensure_all_nodes(adjacency, %MapSet{} = nodes) do
    nodes
    |> MapSet.to_list()
    |> Enum.reduce(adjacency, fn node, acc ->
      Map.put_new(acc, node, [])
    end)
  end

  @spec walk_components(map(), [any()], map(), non_neg_integer()) :: non_neg_integer()
  defp walk_components(_visited, [], _adjacency, count), do: count

  defp walk_components(visited, [node | rest], adjacency, count) do
    if Map.has_key?(visited, node) do
      walk_components(visited, rest, adjacency, count)
    else
      neighbors = Map.get(adjacency, node, [])
      component_nodes = dfs(neighbors, adjacency, Map.put(visited, node, true))
      walk_components(component_nodes, rest, adjacency, count + 1)
    end
  end

  @spec dfs([any()], map(), map()) :: map()
  defp dfs([], _adjacency, visited), do: visited

  defp dfs([neighbor | rest], adjacency, visited) do
    if Map.has_key?(visited, neighbor) do
      dfs(rest, adjacency, visited)
    else
      next_neighbors = Map.get(adjacency, neighbor, [])
      dfs(next_neighbors ++ rest, adjacency, Map.put(visited, neighbor, true))
    end
  end

  defp ensure_tensor(embeddings) when is_struct(embeddings, Nx.Tensor), do: embeddings

  defp ensure_tensor(embeddings) when is_list(embeddings) do
    Nx.tensor(embeddings, type: :f32)
  end

  defp has_cycle?(graph, nodes) do
    dg = :digraph.new()

    try do
      Enum.each(nodes, &:digraph.add_vertex(dg, &1))

      Enum.each(graph, fn {from, children} ->
        Enum.each(children, fn child ->
          :digraph.add_vertex(dg, child)
          :digraph.add_edge(dg, from, child)
        end)
      end)

      not :digraph_utils.is_acyclic(dg)
    after
      :digraph.delete(dg)
    end
  end

  @spec compute_knn_variance(Nx.Tensor.t(), pos_integer(), :euclidean | :cosine) :: float()
  defp compute_knn_variance(tensor, k, metric) when metric in [:euclidean, :cosine] do
    {n_samples, _} = Nx.shape(tensor)
    distances = compute_distance_matrix(tensor, metric)

    neighbor_means =
      for i <- 0..(n_samples - 1) do
        row =
          distances[i]
          |> Nx.to_flat_list()
          |> Enum.reject(&(&1 == 0.0))
          |> Enum.sort()
          |> Enum.take(k)

        case row do
          [] -> 0.0
          _ -> Enum.sum(row) / length(row)
        end
      end

    mean_distance = Enum.sum(neighbor_means) / max(length(neighbor_means), 1)
    normalize_fragility(mean_distance, metric)
  end

  @spec compute_distance_matrix(Nx.Tensor.t(), :euclidean | :cosine) :: Nx.Tensor.t()
  defp compute_distance_matrix(tensor, :euclidean) do
    # Compute Euclidean distance matrix
    {_n, _} = Nx.shape(tensor)

    # Expand dimensions for broadcasting
    # Shape: {n, 1, d}
    a = Nx.new_axis(tensor, 1)
    # Shape: {1, n, d}
    b = Nx.new_axis(tensor, 0)

    # Compute squared differences
    diff = Nx.subtract(a, b)
    squared = Nx.multiply(diff, diff)
    summed = Nx.sum(squared, axes: [2])

    Nx.sqrt(summed)
  end

  defp compute_distance_matrix(tensor, :cosine) do
    # Compute cosine distance matrix (1 - cosine_similarity)
    {n, _} = Nx.shape(tensor)

    # Normalize vectors
    norms =
      tensor
      |> Nx.multiply(tensor)
      |> Nx.sum(axes: [1])
      |> Nx.sqrt()
      |> Nx.reshape({n, 1})

    normalized = Nx.divide(tensor, Nx.add(norms, 1.0e-8))

    # Compute cosine similarity matrix
    similarity = Nx.dot(normalized, [1], normalized, [1])

    # Convert to distance (1 - similarity)
    Nx.subtract(1.0, similarity)
  end

  @spec normalize_fragility(number(), :cosine | :euclidean) :: float()
  defp normalize_fragility(distance, metric) when metric in [:cosine, :euclidean] do
    max_distance =
      case metric do
        :cosine -> 2.0
        :euclidean -> 1.5
      end

    score = distance / max_distance
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

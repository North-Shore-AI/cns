defmodule CNS.Graph.Topology do
  @moduledoc """
  Topological analysis of reasoning graphs.

  Provides metrics for:
  - Cycle detection
  - Connected components
  - Betti numbers (simplified)
  - Graph density

  ## Examples

      iex> sno = CNS.SNO.new("Claim")
      iex> {:ok, graph} = CNS.Graph.Builder.from_sno(sno)
      iex> CNS.Graph.Topology.is_acyclic?(graph)
      true
  """

  alias ExTopology.Graph, as: TopoGraph

  @doc """
  Check if graph is acyclic (no circular reasoning).
  """
  @spec acyclic?(Graph.t()) :: boolean()
  def acyclic?(graph) do
    Graph.is_acyclic?(graph)
  end

  @doc """
  Find all cycles in the graph.
  """
  @spec find_cycles(Graph.t()) :: [[any()]]
  def find_cycles(graph) do
    # Find strongly connected components with more than one vertex
    graph
    |> Graph.strong_components()
    |> Enum.filter(fn component -> length(component) > 1 end)
  end

  @doc """
  Get number of strongly connected components.
  """
  @spec num_components(Graph.t()) :: non_neg_integer()
  def num_components(graph) do
    TopoGraph.beta_zero(graph)
  end

  @doc """
  Calculate graph density.

  Density = edges / (vertices * (vertices - 1))
  For directed graphs.
  """
  @spec density(Graph.t()) :: float()
  def density(graph) do
    v = Graph.num_vertices(graph)
    e = Graph.num_edges(graph)

    if v <= 1 do
      0.0
    else
      max_edges = v * (v - 1)
      Float.round(e / max_edges, 4)
    end
  end

  @doc """
  Calculate simplified Betti numbers.

  - b0: Number of connected components (clusters of related claims)
  - b1: Number of independent cycles (circular reasoning paths)
  """
  @spec betti_numbers(Graph.t()) :: %{b0: non_neg_integer(), b1: non_neg_integer()}
  def betti_numbers(graph) do
    inv = TopoGraph.invariants(graph)
    %{b0: inv.beta_zero, b1: inv.beta_one}
  end

  @doc """
  Find the longest path in the graph (reasoning depth).
  """
  @spec longest_path_length(Graph.t()) :: non_neg_integer()
  def longest_path_length(graph) do
    cond do
      not Graph.is_acyclic?(graph) -> -1
      Graph.num_vertices(graph) == 0 -> 0
      true -> compute_longest_path(graph)
    end
  end

  defp compute_longest_path(graph) do
    roots = Enum.filter(Graph.vertices(graph), fn v -> Graph.in_degree(graph, v) == 0 end)
    if Enum.empty?(roots), do: 0, else: find_max_path_from_roots(graph, roots)
  end

  defp find_max_path_from_roots(graph, roots) do
    roots
    |> Enum.map(fn root -> longest_path_from(graph, root) end)
    |> Enum.max(fn -> 0 end)
  end

  @doc """
  Get degree statistics for the graph.
  """
  @spec degree_stats(Graph.t()) :: %{
          avg_in: float(),
          avg_out: float(),
          max_in: non_neg_integer(),
          max_out: non_neg_integer()
        }
  def degree_stats(graph) do
    vertices = Graph.vertices(graph)

    if Enum.empty?(vertices) do
      %{avg_in: 0.0, avg_out: 0.0, max_in: 0, max_out: 0}
    else
      in_degrees = Enum.map(vertices, &Graph.in_degree(graph, &1))
      out_degrees = Enum.map(vertices, &Graph.out_degree(graph, &1))

      %{
        avg_in: Float.round(Enum.sum(in_degrees) / length(vertices), 4),
        avg_out: Float.round(Enum.sum(out_degrees) / length(vertices), 4),
        max_in: Enum.max(in_degrees),
        max_out: Enum.max(out_degrees)
      }
    end
  end

  @doc """
  Check if graph has a specific topological property.
  """
  @spec has_property?(Graph.t(), atom()) :: boolean()
  def has_property?(graph, property) do
    case property do
      :acyclic -> acyclic?(graph)
      :connected -> num_components(graph) == 1
      :tree -> tree?(graph)
      :dag -> acyclic?(graph)
      _ -> false
    end
  end

  @doc """
  Get a summary of topological metrics.
  """
  @spec summary(Graph.t()) :: map()
  def summary(graph) do
    betti = betti_numbers(graph)
    degrees = degree_stats(graph)

    %{
      vertices: Graph.num_vertices(graph),
      edges: Graph.num_edges(graph),
      density: density(graph),
      is_acyclic: acyclic?(graph),
      components: num_components(graph),
      cycles: length(find_cycles(graph)),
      betti_b0: betti.b0,
      betti_b1: betti.b1,
      max_depth: longest_path_length(graph),
      avg_in_degree: degrees.avg_in,
      avg_out_degree: degrees.avg_out
    }
  end

  # Private functions

  defp longest_path_from(graph, start) do
    do_longest_path(graph, start, %{}, 0)
  end

  @spec do_longest_path(Graph.t(), any(), map(), non_neg_integer()) :: non_neg_integer()
  defp do_longest_path(_graph, current, visited, depth) when is_map_key(visited, current), do: depth

  defp do_longest_path(graph, current, visited, depth) do
    new_visited = Map.put(visited, current, true)
    neighbors = Graph.out_neighbors(graph, current)
    find_longest_neighbor_path(graph, neighbors, new_visited, depth)
  end

  defp find_longest_neighbor_path(_graph, [], _visited, depth), do: depth

  defp find_longest_neighbor_path(graph, neighbors, visited, depth) do
    neighbors
    |> Enum.map(fn neighbor -> do_longest_path(graph, neighbor, visited, depth + 1) end)
    |> Enum.max()
  end

  defp tree?(graph) do
    TopoGraph.tree?(graph)
  end
end

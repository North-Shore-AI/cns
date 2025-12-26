defmodule CNS.Graph.Traversal do
  @moduledoc """
  Graph traversal algorithms for reasoning graphs.

  Provides:
  - Pathfinding between vertices
  - Reachability analysis
  - Evidence chain extraction
  - Claim dependency analysis
  """

  @doc """
  Find all paths between two vertices.
  """
  @spec find_paths(Graph.t(), any(), any()) :: [[any()]]
  def find_paths(graph, from, to) do
    do_find_paths(graph, from, to, [from], [])
  end

  @doc """
  Check if one vertex is reachable from another.
  """
  @spec reachable?(Graph.t(), any(), any()) :: boolean()
  def reachable?(graph, from, to) do
    Graph.reachable(graph, [from])
    |> Enum.member?(to)
  end

  @doc """
  Get all vertices reachable from a starting vertex.
  """
  @spec reachable_from(Graph.t(), any()) :: [any()]
  def reachable_from(graph, vertex) do
    Graph.reachable(graph, [vertex])
  end

  @doc """
  Get the shortest path between two vertices.
  """
  @spec shortest_path(Graph.t(), any(), any()) :: [any()] | nil
  def shortest_path(graph, from, to) do
    Graph.Pathfinding.dijkstra(graph, from, to)
  end

  @doc """
  Get ancestors (all vertices that can reach this vertex).
  """
  @spec ancestors(Graph.t(), any()) :: [any()]
  def ancestors(graph, vertex) do
    # Use reaching instead of reachable for incoming edges
    Graph.reaching(graph, [vertex])
    |> Enum.reject(&(&1 == vertex))
  end

  @doc """
  Get descendants (all vertices reachable from this vertex).
  """
  @spec descendants(Graph.t(), any()) :: [any()]
  def descendants(graph, vertex) do
    Graph.reachable(graph, [vertex])
    |> Enum.reject(&(&1 == vertex))
  end

  @doc """
  Extract evidence chains supporting a claim.

  Returns all paths from evidence to the claim.
  """
  @spec evidence_chains(Graph.t(), any()) :: [[any()]]
  def evidence_chains(graph, claim_id) do
    # Find all evidence vertices
    evidence_vertices =
      graph
      |> Graph.vertices()
      |> Enum.filter(fn v ->
        labels = Graph.vertex_labels(graph, v)
        Keyword.get(labels, :type) == :evidence
      end)

    # Find paths from each evidence to the claim
    evidence_vertices
    |> Enum.flat_map(fn ev ->
      find_paths(graph, ev, claim_id)
    end)
    |> Enum.reject(&Enum.empty?/1)
  end

  @doc """
  Get the depth of a vertex (longest path from roots).
  """
  @spec vertex_depth(Graph.t(), any()) :: non_neg_integer()
  def vertex_depth(graph, vertex) do
    roots = find_roots(graph)
    if Enum.empty?(roots) or vertex in roots, do: 0, else: find_max_depth(graph, roots, vertex)
  end

  defp find_roots(graph) do
    Graph.vertices(graph)
    |> Enum.filter(fn v -> Graph.in_degree(graph, v) == 0 end)
  end

  defp find_max_depth(graph, roots, vertex) do
    roots
    |> Enum.map(&path_depth_to_vertex(graph, &1, vertex))
    |> Enum.max(fn -> 0 end)
  end

  defp path_depth_to_vertex(graph, root, vertex) do
    case shortest_path(graph, root, vertex) do
      nil -> 0
      path -> length(path) - 1
    end
  end

  @doc """
  Perform breadth-first traversal from a vertex.
  """
  @spec bfs(Graph.t(), any()) :: [any()]
  def bfs(graph, start) do
    do_bfs(graph, [start], MapSet.new([start]), [start])
  end

  @doc """
  Perform depth-first traversal from a vertex.
  """
  @spec dfs(Graph.t(), any()) :: [any()]
  def dfs(graph, start) do
    do_dfs(graph, start, MapSet.new(), [])
    |> Enum.reverse()
  end

  @doc """
  Get topological ordering if graph is acyclic.
  """
  @spec topological_sort(Graph.t()) :: {:ok, [any()]} | {:error, :cyclic}
  def topological_sort(graph) do
    case Graph.topsort(graph) do
      false -> {:error, :cyclic}
      sorted -> {:ok, sorted}
    end
  end

  @doc """
  Find common ancestors of two vertices.
  """
  @spec common_ancestors(Graph.t(), any(), any()) :: [any()]
  def common_ancestors(graph, v1, v2) do
    a1 = MapSet.new(ancestors(graph, v1))
    a2 = MapSet.new(ancestors(graph, v2))

    MapSet.intersection(a1, a2)
    |> MapSet.to_list()
  end

  # Private functions

  defp do_find_paths(_graph, to, to, path, acc) do
    [Enum.reverse(path) | acc]
  end

  defp do_find_paths(graph, from, to, path, acc) do
    neighbors = Graph.out_neighbors(graph, from)

    Enum.reduce(neighbors, acc, fn neighbor, paths ->
      if neighbor in path do
        # Avoid cycles
        paths
      else
        do_find_paths(graph, neighbor, to, [neighbor | path], paths)
      end
    end)
  end

  defp do_bfs(_graph, [], _visited, result) do
    Enum.reverse(result)
  end

  defp do_bfs(graph, queue, visited, result) do
    [current | rest] = queue

    neighbors =
      Graph.out_neighbors(graph, current)
      |> Enum.reject(&MapSet.member?(visited, &1))

    new_visited = Enum.reduce(neighbors, visited, &MapSet.put(&2, &1))
    new_queue = rest ++ neighbors
    new_result = neighbors ++ result

    do_bfs(graph, new_queue, new_visited, new_result)
  end

  defp do_dfs(_graph, nil, _visited, result), do: result

  defp do_dfs(graph, vertex, visited, result) do
    if MapSet.member?(visited, vertex) do
      result
    else
      new_visited = MapSet.put(visited, vertex)
      new_result = [vertex | result]

      Graph.out_neighbors(graph, vertex)
      |> Enum.reduce({new_visited, new_result}, fn neighbor, {vis, res} ->
        updated_res = do_dfs(graph, neighbor, vis, res)
        {MapSet.put(vis, neighbor), updated_res}
      end)
      |> elem(1)
    end
  end
end

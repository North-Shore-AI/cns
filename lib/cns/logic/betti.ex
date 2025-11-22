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
      iex> stats = CNS.Logic.Betti.compute_graph_stats(claim_ids, relations)
      iex> stats.polarity_conflict
      true
  """
  @spec compute_graph_stats([String.t()], [relation()]) :: GraphStats.t()
  def compute_graph_stats(claim_ids, relations) do
    graph = build_graph(claim_ids, relations)

    nodes = Graph.num_vertices(graph)
    edges = Graph.num_edges(graph)

    components =
      if nodes == 0 do
        0
      else
        graph
        |> to_undirected()
        |> count_components()
      end

    beta1 = max(0, edges - nodes + components)

    cycles = find_cycles(graph)

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

  @doc """
  Detect if a claim has conflicting polarity (both supports and refutes edges).

  ## Examples

      iex> relations = [{"c2", "supports", "c1"}, {"c3", "refutes", "c1"}]
      iex> CNS.Logic.Betti.polarity_conflict?(relations, "c1")
      true

      iex> relations = [{"c2", "supports", "c1"}, {"c3", "supports", "c1"}]
      iex> CNS.Logic.Betti.polarity_conflict?(relations, "c1")
      false
  """
  @spec polarity_conflict?([relation()], String.t()) :: boolean()
  def polarity_conflict?(relations, target \\ "c1") do
    normalized_target = normalize_id(target)

    labels =
      relations
      |> Enum.filter(fn {_src, _label, dst} -> normalize_id(dst) == normalized_target end)
      |> Enum.map(fn {_src, label, _dst} -> String.downcase(label) end)
      |> MapSet.new()

    MapSet.member?(labels, "supports") and MapSet.member?(labels, "refutes")
  end

  @doc """
  Find all cycles in the reasoning graph.

  ## Examples

      iex> graph = Graph.new(type: :directed) |> Graph.add_vertices(["c1", "c2", "c3"])
      iex> graph = graph |> Graph.add_edge("c1", "c2") |> Graph.add_edge("c2", "c3") |> Graph.add_edge("c3", "c1")
      iex> cycles = CNS.Logic.Betti.find_cycles(graph)
      iex> length(cycles) >= 1
      true
  """
  @spec find_cycles(Graph.t()) :: [[String.t()]]
  def find_cycles(graph) do
    vertices = Graph.vertices(graph)

    cycles =
      Enum.flat_map(vertices, fn start_node ->
        find_cycles_from(graph, start_node, [start_node], MapSet.new([start_node]))
      end)

    cycles
    |> Enum.map(&normalize_cycle/1)
    |> Enum.uniq()
    |> Enum.take(100)
  end

  defp find_cycles_from(graph, current, path, visited) do
    neighbors = Graph.out_neighbors(graph, current)
    start_node = hd(path)

    Enum.flat_map(neighbors, fn neighbor ->
      cond do
        neighbor == start_node and length(path) > 1 ->
          [path ++ [neighbor]]

        MapSet.member?(visited, neighbor) ->
          []

        true ->
          find_cycles_from(graph, neighbor, path ++ [neighbor], MapSet.put(visited, neighbor))
      end
    end)
  end

  defp normalize_cycle(cycle) do
    min_elem = Enum.min(cycle)
    min_idx = Enum.find_index(cycle, &(&1 == min_elem))

    cycle_no_dup =
      if List.last(cycle) == hd(cycle), do: Enum.drop(cycle, -1), else: cycle

    {before, after_min} = Enum.split(cycle_no_dup, min_idx)
    after_min ++ before
  end

  # Private functions

  defp build_graph(claim_ids, relations) do
    graph =
      Enum.reduce(claim_ids, Graph.new(type: :directed), fn id, g ->
        Graph.add_vertex(g, normalize_id(id))
      end)

    Enum.reduce(relations, graph, fn {src, label, dst}, g ->
      normalized_src = normalize_id(src)
      normalized_dst = normalize_id(dst)

      if Graph.has_vertex?(g, normalized_src) and Graph.has_vertex?(g, normalized_dst) do
        Graph.add_edge(g, normalized_src, normalized_dst, label: label)
      else
        g
      end
    end)
  end

  defp normalize_id(id) do
    id
    |> String.downcase()
    |> String.trim()
  end

  defp to_undirected(graph) do
    vertices = Graph.vertices(graph)
    edges = Graph.edges(graph)

    undirected =
      Enum.reduce(vertices, Graph.new(), fn v, g ->
        Graph.add_vertex(g, v)
      end)

    Enum.reduce(edges, undirected, fn edge, g ->
      g
      |> Graph.add_edge(edge.v1, edge.v2)
      |> Graph.add_edge(edge.v2, edge.v1)
    end)
  end

  defp count_components(graph) do
    vertices = Graph.vertices(graph)

    if Enum.empty?(vertices) do
      0
    else
      {_visited, count} =
        Enum.reduce(vertices, {MapSet.new(), 0}, fn vertex, {visited, count} ->
          if MapSet.member?(visited, vertex) do
            {visited, count}
          else
            new_visited = bfs(graph, vertex, visited)
            {new_visited, count + 1}
          end
        end)

      count
    end
  end

  defp bfs(graph, start, visited) do
    queue = :queue.in(start, :queue.new())
    do_bfs(graph, queue, MapSet.put(visited, start))
  end

  defp do_bfs(graph, queue, visited) do
    case :queue.out(queue) do
      {:empty, _} ->
        visited

      {{:value, node}, rest} ->
        neighbors = Graph.neighbors(graph, node)

        {new_queue, new_visited} =
          Enum.reduce(neighbors, {rest, visited}, fn neighbor, {q, v} ->
            if MapSet.member?(v, neighbor) do
              {q, v}
            else
              {:queue.in(neighbor, q), MapSet.put(v, neighbor)}
            end
          end)

        do_bfs(graph, new_queue, new_visited)
    end
  end
end

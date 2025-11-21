defmodule CNS.Topology do
  @moduledoc """
  Graph topology analysis for CNS claim networks.

  Analyzes the structure of claim relationships including:
  - Betti number calculation
  - Cycle detection
  - DAG validation
  - Connectivity analysis

  These topological features provide insights into the complexity
  and structure of the dialectical reasoning process.
  """

  alias CNS.{SNO, Provenance}

  @doc """
  Build a graph from a list of SNOs based on provenance relationships.

  Returns a map of node IDs to their children (edges).

  ## Examples

      iex> s1 = CNS.SNO.new("A", id: "1")
      iex> prov = CNS.Provenance.new(:synthesizer, parent_ids: ["1"])
      iex> s2 = CNS.SNO.new("B", id: "2", provenance: prov)
      iex> graph = CNS.Topology.build_graph([s1, s2])
      iex> Map.has_key?(graph, "1")
      true
  """
  @spec build_graph([SNO.t()]) :: map()
  def build_graph(snos) when is_list(snos) do
    # Initialize graph with all nodes
    nodes = Enum.map(snos, & &1.id)
    graph = Map.new(nodes, fn id -> {id, []} end)

    # Add edges based on provenance
    Enum.reduce(snos, graph, fn sno, acc ->
      case sno.provenance do
        %Provenance{parent_ids: parent_ids} when is_list(parent_ids) ->
          # Add edges from parents to this node
          Enum.reduce(parent_ids, acc, fn parent_id, inner_acc ->
            if Map.has_key?(inner_acc, parent_id) do
              Map.update(inner_acc, parent_id, [sno.id], &[sno.id | &1])
            else
              inner_acc
            end
          end)

        _ ->
          acc
      end
    end)
  end

  @doc """
  Calculate Betti numbers for the claim graph.

  Betti numbers characterize the topology:
  - b0: Number of connected components
  - b1: Number of independent cycles

  ## Examples

      iex> graph = %{"a" => ["b"], "b" => ["c"], "c" => []}
      iex> betti = CNS.Topology.betti_numbers(graph)
      iex> betti.b0
      1
  """
  @spec betti_numbers(map()) :: %{b0: non_neg_integer(), b1: non_neg_integer()}
  def betti_numbers(graph) when is_map(graph) do
    nodes = Map.keys(graph)
    n = length(nodes)

    if n == 0 do
      %{b0: 0, b1: 0}
    else
      # b0 = number of connected components
      b0 = count_components(graph)

      # Count edges
      edges = count_edges(graph)

      # b1 = edges - nodes + components (for undirected graph)
      # For directed graph, this is an approximation
      b1 = max(0, edges - n + b0)

      %{b0: b0, b1: b1}
    end
  end

  @doc """
  Detect cycles in the claim graph.

  Returns list of cycles found.

  ## Examples

      iex> graph = %{"a" => ["b"], "b" => ["a"]}
      iex> cycles = CNS.Topology.detect_cycles(graph)
      iex> length(cycles) > 0
      true
  """
  @spec detect_cycles(map()) :: [[String.t()]]
  def detect_cycles(graph) when is_map(graph) do
    nodes = Map.keys(graph)

    cycles =
      Enum.flat_map(nodes, fn start_node ->
        find_cycles_from(graph, start_node, [start_node], MapSet.new([start_node]))
      end)

    # Deduplicate cycles (same cycle can be found from different starting points)
    cycles
    |> Enum.map(&normalize_cycle/1)
    |> Enum.uniq()
  end

  @doc """
  Check if the graph is a valid DAG (Directed Acyclic Graph).

  ## Examples

      iex> dag = %{"a" => ["b"], "b" => ["c"], "c" => []}
      iex> CNS.Topology.is_dag?(dag)
      true

      iex> cyclic = %{"a" => ["b"], "b" => ["a"]}
      iex> CNS.Topology.is_dag?(cyclic)
      false
  """
  @spec is_dag?(map()) :: boolean()
  def is_dag?(graph) when is_map(graph) do
    detect_cycles(graph) == []
  end

  @doc """
  Calculate the depth of the graph (longest path).

  ## Examples

      iex> graph = %{"a" => ["b"], "b" => ["c"], "c" => []}
      iex> CNS.Topology.depth(graph)
      2
  """
  @spec depth(map()) :: non_neg_integer()
  def depth(graph) when is_map(graph) do
    nodes = Map.keys(graph)

    if Enum.empty?(nodes) do
      0
    else
      # Find root nodes (no incoming edges)
      roots = find_roots(graph)

      if Enum.empty?(roots) do
        # Graph has cycles, use any node
        max_depth_from(graph, hd(nodes), MapSet.new())
      else
        # Calculate max depth from all roots
        Enum.map(roots, &max_depth_from(graph, &1, MapSet.new()))
        |> Enum.max(fn -> 0 end)
      end
    end
  end

  @doc """
  Find root nodes (nodes with no incoming edges).

  ## Examples

      iex> graph = %{"a" => ["b", "c"], "b" => [], "c" => []}
      iex> roots = CNS.Topology.find_roots(graph)
      iex> "a" in roots
      true
  """
  @spec find_roots(map()) :: [String.t()]
  def find_roots(graph) when is_map(graph) do
    all_nodes = MapSet.new(Map.keys(graph))
    children = graph |> Map.values() |> List.flatten() |> MapSet.new()

    MapSet.difference(all_nodes, children)
    |> MapSet.to_list()
  end

  @doc """
  Find leaf nodes (nodes with no outgoing edges).

  ## Examples

      iex> graph = %{"a" => ["b"], "b" => []}
      iex> leaves = CNS.Topology.find_leaves(graph)
      iex> "b" in leaves
      true
  """
  @spec find_leaves(map()) :: [String.t()]
  def find_leaves(graph) when is_map(graph) do
    Enum.filter(graph, fn {_node, children} -> Enum.empty?(children) end)
    |> Enum.map(fn {node, _} -> node end)
  end

  @doc """
  Calculate connectivity metrics for the graph.

  ## Examples

      iex> graph = %{"a" => ["b"], "b" => ["c"], "c" => []}
      iex> metrics = CNS.Topology.connectivity(graph)
      iex> metrics.density >= 0.0
      true
  """
  @spec connectivity(map()) :: map()
  def connectivity(graph) when is_map(graph) do
    n = map_size(graph)
    edges = count_edges(graph)

    # Density = actual edges / possible edges
    max_edges = if n > 1, do: n * (n - 1), else: 0
    density = if max_edges > 0, do: edges / max_edges, else: 0.0

    %{
      nodes: n,
      edges: edges,
      density: Float.round(density, 4),
      components: count_components(graph),
      roots: length(find_roots(graph)),
      leaves: length(find_leaves(graph))
    }
  end

  @doc """
  Get all paths between two nodes.

  ## Examples

      iex> graph = %{"a" => ["b", "c"], "b" => ["d"], "c" => ["d"], "d" => []}
      iex> paths = CNS.Topology.all_paths(graph, "a", "d")
      iex> length(paths)
      2
  """
  @spec all_paths(map(), String.t(), String.t()) :: [[String.t()]]
  def all_paths(graph, start_node, end_node) do
    find_paths(graph, start_node, end_node, [start_node], MapSet.new([start_node]))
  end

  @doc """
  Topological sort of the graph (if DAG).

  Returns nodes in topological order.

  ## Examples

      iex> graph = %{"a" => ["b"], "b" => ["c"], "c" => []}
      iex> {:ok, sorted} = CNS.Topology.topological_sort(graph)
      iex> length(sorted)
      3
  """
  @spec topological_sort(map()) :: {:ok, [String.t()]} | {:error, :has_cycle}
  def topological_sort(graph) when is_map(graph) do
    if is_dag?(graph) do
      sorted = kahn_sort(graph)
      {:ok, sorted}
    else
      {:error, :has_cycle}
    end
  end

  # Private functions

  defp count_edges(graph) do
    graph
    |> Map.values()
    |> Enum.map(&length/1)
    |> Enum.sum()
  end

  defp count_components(graph) do
    nodes = Map.keys(graph)

    if Enum.empty?(nodes) do
      0
    else
      {_, count} =
        Enum.reduce(nodes, {MapSet.new(), 0}, fn node, {visited, count} ->
          if MapSet.member?(visited, node) do
            {visited, count}
          else
            new_visited = bfs(graph, node, visited)
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
        children = Map.get(graph, node, [])

        {new_queue, new_visited} =
          Enum.reduce(children, {rest, visited}, fn child, {q, v} ->
            if MapSet.member?(v, child) do
              {q, v}
            else
              {:queue.in(child, q), MapSet.put(v, child)}
            end
          end)

        do_bfs(graph, new_queue, new_visited)
    end
  end

  defp find_cycles_from(graph, current, path, visited) do
    children = Map.get(graph, current, [])
    start_node = hd(path)

    Enum.flat_map(children, fn child ->
      cond do
        child == start_node and length(path) > 1 ->
          # Found a cycle back to start
          [path ++ [child]]

        MapSet.member?(visited, child) ->
          # Already visited, but not back to start - not a cycle from this path
          []

        true ->
          # Continue exploring
          find_cycles_from(graph, child, path ++ [child], MapSet.put(visited, child))
      end
    end)
  end

  defp normalize_cycle(cycle) do
    # Normalize cycle to start with smallest element
    min_elem = Enum.min(cycle)
    min_idx = Enum.find_index(cycle, &(&1 == min_elem))

    # Remove the duplicate end element if present
    cycle_no_dup = if List.last(cycle) == hd(cycle), do: Enum.drop(cycle, -1), else: cycle

    # Rotate to start with min element
    {before, after_min} = Enum.split(cycle_no_dup, min_idx)
    after_min ++ before
  end

  @spec max_depth_from(map(), String.t(), MapSet.t()) :: non_neg_integer()
  defp max_depth_from(graph, node, visited) do
    cond do
      # Safety limit
      MapSet.size(visited) > 1000 ->
        0

      MapSet.member?(visited, node) ->
        0

      true ->
        children = Map.get(graph, node, [])

        if Enum.empty?(children) do
          0
        else
          new_visited = MapSet.put(visited, node)

          child_depths =
            Enum.map(children, &max_depth_from(graph, &1, new_visited))

          1 + Enum.max(child_depths, fn -> 0 end)
        end
    end
  end

  defp find_paths(_graph, current, target, path, _visited) when current == target do
    [path]
  end

  defp find_paths(graph, current, target, path, visited) do
    children = Map.get(graph, current, [])

    Enum.flat_map(children, fn child ->
      if MapSet.member?(visited, child) do
        []
      else
        find_paths(graph, child, target, path ++ [child], MapSet.put(visited, child))
      end
    end)
  end

  defp kahn_sort(graph) do
    # Calculate in-degrees
    in_degrees =
      Map.keys(graph)
      |> Map.new(fn node -> {node, 0} end)

    in_degrees =
      Enum.reduce(graph, in_degrees, fn {_node, children}, degrees ->
        Enum.reduce(children, degrees, fn child, acc ->
          Map.update(acc, child, 1, &(&1 + 1))
        end)
      end)

    # Start with nodes that have no incoming edges
    queue =
      in_degrees
      |> Enum.filter(fn {_node, degree} -> degree == 0 end)
      |> Enum.map(fn {node, _} -> node end)

    do_kahn_sort(graph, queue, in_degrees, [])
  end

  defp do_kahn_sort(_graph, [], _in_degrees, result) do
    Enum.reverse(result)
  end

  defp do_kahn_sort(graph, [node | rest], in_degrees, result) do
    children = Map.get(graph, node, [])

    # Decrease in-degree for children
    {new_in_degrees, new_queue_additions} =
      Enum.reduce(children, {in_degrees, []}, fn child, {degrees, additions} ->
        new_degree = Map.get(degrees, child, 1) - 1
        new_degrees = Map.put(degrees, child, new_degree)

        if new_degree == 0 do
          {new_degrees, [child | additions]}
        else
          {new_degrees, additions}
        end
      end)

    do_kahn_sort(graph, rest ++ new_queue_additions, new_in_degrees, [node | result])
  end
end

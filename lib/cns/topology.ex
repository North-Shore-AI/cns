defmodule CNS.Topology do
  @moduledoc """
  Topology facade for CNS claim networks.

  This module keeps the CNS public API thin and delegates the heavy lifting to
  `ex_topology` (for Betti numbers, invariants, fragility, and persistent
  homology) and `libgraph` (for graph representation). CNS code focuses on
  mapping SNO structures into graphs/embeddings and interpreting the results.
  """

  alias CNS.{Provenance, SNO}
  alias CNS.Topology.{Adapter, Persistence, Surrogates}
  alias ExTopology.Graph, as: TopoGraph

  @type graph_like :: Graph.t() | map() | [SNO.t()]

  @doc """
  Build a directed Graph.t from a list of SNOs or an existing graph-like
  structure (graph struct or adjacency map).
  """
  @spec build_graph(graph_like()) :: Graph.t()
  def build_graph(%Graph{} = graph), do: graph

  def build_graph(snos) when is_list(snos) do
    Enum.reduce(snos, Graph.new(type: :directed), fn %SNO{id: id, provenance: prov}, g ->
      g = Graph.add_vertex(g, id)

      parent_ids =
        case prov do
          %Provenance{parent_ids: parents} when is_list(parents) -> parents
          _ -> []
        end

      Enum.reduce(parent_ids, g, fn parent_id, acc ->
        acc
        |> Graph.add_vertex(parent_id)
        |> Graph.add_edge(parent_id, id)
      end)
    end)
  end

  def build_graph(graph_map) when is_map(graph_map), do: map_to_graph(graph_map)

  @doc """
  Compute graph invariants (β₀, β₁, Euler characteristic, vertices, edges).
  """
  @spec invariants(graph_like()) :: map()
  def invariants(input) do
    input
    |> build_graph()
    |> TopoGraph.invariants()
  end

  @doc """
  Convenience wrapper returning only β₀/β₁.
  """
  @spec betti_numbers(graph_like()) :: %{b0: non_neg_integer(), b1: non_neg_integer()}
  def betti_numbers(input) do
    inv = invariants(input)
    %{b0: inv.beta_zero, b1: inv.beta_one}
  end

  @doc """
  Detect cycles using strongly connected components.

  Returns components that contain a self-loop or more than one node.
  """
  @spec detect_cycles(graph_like()) :: [[any()]]
  def detect_cycles(input) do
    graph = build_graph(input)

    graph
    |> Graph.strong_components()
    |> Enum.filter(fn comp ->
      length(comp) > 1 or has_self_loop?(graph, comp)
    end)
  end

  @doc """
  Check if the graph is a DAG.
  """
  @spec dag?(graph_like()) :: boolean()
  def dag?(input), do: input |> build_graph() |> Graph.is_acyclic?()

  @doc """
  Depth of the graph (longest path length).
  """
  @spec depth(graph_like()) :: non_neg_integer()
  def depth(input) do
    graph = build_graph(input)

    cond do
      Graph.num_vertices(graph) == 0 ->
        0

      Graph.is_acyclic?(graph) ->
        roots = find_roots(graph)
        roots |> Enum.map(&max_depth_from(graph, &1, %{})) |> Enum.max(fn -> 0 end)

      true ->
        # For cyclic graphs, still attempt a finite depth via visited guard.
        graph
        |> Graph.vertices()
        |> Enum.map(&max_depth_from(graph, &1, %{}))
        |> Enum.max(fn -> 0 end)
    end
  end

  @doc """
  Root nodes (no incoming edges).
  """
  @spec find_roots(graph_like()) :: [any()]
  def find_roots(input) do
    graph = build_graph(input)
    Graph.vertices(graph) |> Enum.filter(fn v -> Graph.in_degree(graph, v) == 0 end)
  end

  @doc """
  Leaf nodes (no outgoing edges).
  """
  @spec find_leaves(graph_like()) :: [any()]
  def find_leaves(input) do
    graph = build_graph(input)
    Graph.vertices(graph) |> Enum.filter(fn v -> Graph.out_degree(graph, v) == 0 end)
  end

  @doc """
  Basic connectivity metrics (nodes, edges, density, components).
  """
  @spec connectivity(graph_like()) :: %{
          nodes: non_neg_integer(),
          edges: non_neg_integer(),
          density: float(),
          components: non_neg_integer()
        }
  def connectivity(input) do
    graph = build_graph(input)
    inv = TopoGraph.invariants(graph)

    v = Graph.num_vertices(graph)
    e = TopoGraph.num_edges(graph)
    density = if v <= 1, do: 0.0, else: Float.round(e / (v * (v - 1)), 4)

    %{nodes: v, edges: e, density: density, components: inv.beta_zero}
  end

  @doc """
  Enumerate all simple paths between two nodes.
  """
  @spec all_paths(graph_like(), any(), any()) :: [[any()]]
  def all_paths(input, start_node, end_node) do
    graph = build_graph(input)
    do_all_paths(graph, start_node, end_node, [start_node], MapSet.new([start_node]))
  end

  @doc """
  Topological sort for DAGs.
  """
  @spec topological_sort(graph_like()) :: {:ok, [any()]} | {:error, :has_cycle}
  def topological_sort(input) do
    graph = build_graph(input)

    case Graph.topsort(graph) do
      false -> {:error, :has_cycle}
      list -> {:ok, list}
    end
  end

  @doc """
  Analyze a claim network (SNO list) and return topology stats.
  """
  @spec analyze_claim_network([SNO.t()], keyword()) :: map()
  def analyze_claim_network(snos, _opts \\ []) do
    graph = build_graph(snos)
    inv = TopoGraph.invariants(graph)
    cycles = detect_cycles(graph)

    %{
      beta1: inv.beta_one,
      dag?: Graph.is_acyclic?(graph),
      sno_count: length(snos),
      cycles: cycles,
      link_count: TopoGraph.num_edges(graph)
    }
  end

  @doc """
  Detect circular reasoning events.
  """
  @spec detect_circular_reasoning(SNO.t() | [SNO.t()]) ::
          {:ok, [map()]} | {:error, term()}
  def detect_circular_reasoning(sno_or_snos) do
    snos = List.wrap(sno_or_snos)
    analysis = analyze_claim_network(snos)

    if analysis.beta1 > 0 do
      {:ok, [%{type: :circular_reasoning, beta1: analysis.beta1, cycles: analysis.cycles}]}
    else
      {:ok, []}
    end
  end

  @doc """
  Lightweight fragility estimate (embedding variance surrogate).
  """
  @spec fragility([SNO.t()] | SNO.t(), keyword()) :: float()
  def fragility(snos_or_sno, opts \\ []) do
    snos = List.wrap(snos_or_sno)
    embeddings = Adapter.sno_embeddings(snos, Keyword.get(opts, :embedding_opts, []))
    Surrogates.compute_fragility_surrogate(embeddings, opts)
  end

  @doc """
  β₁ with selectable mode.

  Currently both modes use graph invariants; `:exact` is reserved for future
  persistent homology-backed implementations.
  """
  @spec beta1(graph_like(), keyword()) :: non_neg_integer()
  def beta1(snos_or_graph, opts \\ []) do
    _mode = Keyword.get(opts, :mode, :surrogate)
    invariants(snos_or_graph).beta_one
  end

  @doc """
  Convenience wrapper for surrogate metrics (β₁ + fragility).
  """
  @spec surrogates(SNO.t() | [SNO.t()], keyword()) :: %{
          beta1: non_neg_integer(),
          fragility: float()
        }
  def surrogates(snos_or_sno, opts \\ []) do
    %{
      beta1: beta1(snos_or_sno),
      fragility: fragility(snos_or_sno, opts)
    }
  end

  @doc """
  Full persistent homology analysis.
  """
  @spec tda([SNO.t()], keyword()) :: Persistence.persistence_result()
  def tda(snos, opts \\ []), do: Persistence.compute(snos, opts)

  # -- Private helpers ------------------------------------------------------

  defp map_to_graph(map) do
    Enum.reduce(map, Graph.new(type: :directed), fn {node, children}, g ->
      g =
        g
        |> Graph.add_vertex(node)
        |> add_missing_children(children)

      Enum.reduce(children, g, fn child, acc -> Graph.add_edge(acc, node, child) end)
    end)
  end

  defp add_missing_children(graph, children) do
    Enum.reduce(children, graph, fn child, acc -> Graph.add_vertex(acc, child) end)
  end

  defp has_self_loop?(graph, component) do
    Enum.any?(component, fn v -> Graph.edge(graph, v, v) != nil end)
  end

  defp max_depth_from(graph, node, visited) when is_map(visited) do
    if Map.has_key?(visited, node) do
      0
    else
      calculate_max_child_depth(graph, node, Map.put(visited, node, true))
    end
  end

  defp calculate_max_child_depth(graph, node, visited) do
    children = Graph.out_neighbors(graph, node)

    if Enum.empty?(children) do
      0
    else
      child_depths = Enum.map(children, fn child -> max_depth_from(graph, child, visited) end)
      1 + Enum.max(child_depths, fn -> 0 end)
    end
  end

  defp do_all_paths(_graph, current, target, path, _visited) when current == target, do: [path]

  defp do_all_paths(graph, current, target, path, visited) do
    Graph.out_neighbors(graph, current)
    |> Enum.flat_map(fn child ->
      if MapSet.member?(visited, child) do
        []
      else
        do_all_paths(graph, child, target, path ++ [child], MapSet.put(visited, child))
      end
    end)
  end
end

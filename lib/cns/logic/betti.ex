defmodule CNS.Logic.Betti do
  @moduledoc """
  Compute Betti numbers and cycle diagnostics for CLAIM/RELATION graphs
  to detect logical inconsistencies.

  The first Betti number (beta1) measures the number of independent cycles
  in the reasoning graph. High beta1 indicates circular reasoning patterns.
  """

  alias CNS.Topology
  alias ExTopology.Graph, as: TopoGraph

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

    components = TopoGraph.beta_zero(graph)
    beta1 = TopoGraph.beta_one(graph)
    cycles = Topology.detect_cycles(graph)

    conflict = polarity_conflict?(relations)

    %GraphStats{
      nodes: Graph.num_vertices(graph),
      edges: Graph.num_edges(graph),
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
  def find_cycles(graph), do: Topology.detect_cycles(graph)

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
end

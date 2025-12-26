defmodule CNS.Graph.Builder do
  @moduledoc """
  Build reasoning graphs from SNOs.

  Creates directed graphs representing:
  - Claim-to-evidence relationships
  - Evidence-to-evidence citations
  - Claim hierarchies

  ## Examples

      iex> sno = CNS.SNO.new("Claim", evidence: [CNS.Evidence.new("Src", "Content")])
      iex> {:ok, graph} = CNS.Graph.Builder.from_sno(sno)
      iex> Graph.num_vertices(graph) > 0
      true
  """

  alias CNS.{Evidence, SNO}

  @type vertex_id :: String.t()
  @type edge_label :: :supports | :cites | :contradicts | :child_of

  @doc """
  Build a graph from an SNO.
  """
  @spec from_sno(SNO.t()) :: {:ok, Graph.t()} | {:error, term()}
  def from_sno(%SNO{} = sno) do
    graph = Graph.new(type: :directed)

    graph =
      graph
      |> add_claim_vertex(sno)
      |> add_evidence_vertices(sno)
      |> add_child_vertices(sno)
      |> add_support_edges(sno)
      |> add_child_edges(sno)

    {:ok, graph}
  rescue
    e -> {:error, Exception.message(e)}
  end

  @doc """
  Build a graph from multiple SNOs.
  """
  @spec from_sno_list([SNO.t()]) :: {:ok, Graph.t()} | {:error, term()}
  def from_sno_list(snos) when is_list(snos) do
    graph =
      Enum.reduce(snos, Graph.new(type: :directed), fn sno, acc ->
        {:ok, g} = from_sno(sno)
        merge_graphs(acc, g)
      end)

    {:ok, graph}
  rescue
    e -> {:error, Exception.message(e)}
  end

  @doc """
  Add an edge between two vertices.
  """
  @spec add_edge(Graph.t(), vertex_id(), vertex_id(), edge_label()) :: Graph.t()
  def add_edge(graph, from, to, label) do
    Graph.add_edge(graph, from, to, label: label)
  end

  @doc """
  Get all vertices of a specific type.
  """
  @spec vertices_of_type(Graph.t(), :claim | :evidence) :: [vertex_id()]
  def vertices_of_type(graph, type) do
    graph
    |> Graph.vertices()
    |> Enum.filter(fn v ->
      case Graph.vertex_labels(graph, v) do
        [^type | _] -> true
        _ -> false
      end
    end)
  end

  @doc """
  Get edges with a specific label.
  """
  @spec edges_with_label(Graph.t(), edge_label()) :: [Graph.Edge.t()]
  def edges_with_label(graph, label) do
    graph
    |> Graph.edges()
    |> Enum.filter(fn edge ->
      edge.label == label
    end)
  end

  # Private functions

  defp add_claim_vertex(graph, %SNO{id: id, claim: claim, confidence: confidence}) do
    Graph.add_vertex(graph, id,
      type: :claim,
      text: claim,
      confidence: confidence
    )
  end

  defp add_evidence_vertices(graph, %SNO{evidence: evidence}) do
    Enum.reduce(evidence, graph, fn %Evidence{
                                      id: id,
                                      source: source,
                                      validity: validity,
                                      content: content
                                    },
                                    g ->
      Graph.add_vertex(g, id,
        type: :evidence,
        source: source,
        validity: validity,
        content: content
      )
    end)
  end

  defp add_child_vertices(graph, %SNO{children: children}) do
    Enum.reduce(children, graph, fn child_sno, g ->
      {:ok, child_graph} = from_sno(child_sno)
      merge_graphs(g, child_graph)
    end)
  end

  defp add_support_edges(graph, %SNO{id: claim_id, evidence: evidence}) do
    Enum.reduce(evidence, graph, fn %Evidence{id: ev_id}, g ->
      Graph.add_edge(g, ev_id, claim_id, label: :supports)
    end)
  end

  defp add_child_edges(graph, %SNO{id: parent_id, children: children}) do
    Enum.reduce(children, graph, fn %SNO{id: child_id}, g ->
      Graph.add_edge(g, child_id, parent_id, label: :child_of)
    end)
  end

  defp merge_graphs(g1, g2) do
    # Add all vertices from g2 to g1
    g1 =
      Enum.reduce(Graph.vertices(g2), g1, fn v, acc ->
        labels = Graph.vertex_labels(g2, v)
        Graph.add_vertex(acc, v, labels)
      end)

    # Add all edges from g2 to g1
    Enum.reduce(Graph.edges(g2), g1, fn edge, acc ->
      Graph.add_edge(acc, edge.v1, edge.v2, label: edge.label)
    end)
  end
end

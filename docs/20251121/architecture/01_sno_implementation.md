# Structured Narrative Objects (SNO) Implementation

## Version
- Date: 2025-11-21
- Status: Design Document
- Author: Architecture Team

## Overview

SNOs are the core data structure in CNS, representing dialectical reasoning graphs with evidence linking and trust scoring.

## Module Structure

```
lib/cns/
  sno.ex                      # Main SNO struct and API
  sno/
    graph.ex                  # Graph representation and operations
    evidence.ex               # Evidence linking and citation
    trust.ex                  # Trust scoring and propagation
    temporal.ex               # Temporal evolution tracking
    serialization.ex          # JSON/binary serialization
```

## SNO Struct Definition

```elixir
defmodule CNS.SNO do
  @moduledoc """
  Structured Narrative Object - a dialectical reasoning graph.

  An SNO represents a synthesis of conflicting information with:
  - Thesis/antithesis/synthesis structure
  - Evidence with citations
  - Graph representation of claim relationships
  - Trust scores for evidence quality
  """

  alias CNS.SNO.{Graph, Evidence, Trust}

  @type claim :: String.t()
  @type evidence_id :: String.t()
  @type node_id :: String.t()
  @type edge_type :: :supports | :contradicts | :elaborates | :cites | :derives

  @type t :: %__MODULE__{
    id: String.t(),
    thesis: claim(),
    antithesis: claim(),
    synthesis: claim() | nil,
    evidence: %{evidence_id() => Evidence.t()},
    graph: Graph.t(),
    trust_scores: %{node_id() => float()},
    metadata: map(),
    created_at: DateTime.t(),
    updated_at: DateTime.t(),
    version: pos_integer()
  }

  defstruct [
    :id,
    :thesis,
    :antithesis,
    :synthesis,
    evidence: %{},
    graph: nil,
    trust_scores: %{},
    metadata: %{},
    created_at: nil,
    updated_at: nil,
    version: 1
  ]

  @doc """
  Creates a new SNO from thesis and antithesis.
  """
  @spec new(keyword()) :: t()
  def new(opts) do
    thesis = Keyword.fetch!(opts, :thesis)
    antithesis = Keyword.fetch!(opts, :antithesis)
    evidence = Keyword.get(opts, :evidence, [])

    id = generate_id()
    now = DateTime.utc_now()

    # Initialize graph with thesis and antithesis as nodes
    graph = Graph.new()
    |> Graph.add_node("thesis", %{type: :thesis, content: thesis})
    |> Graph.add_node("antithesis", %{type: :antithesis, content: antithesis})
    |> Graph.add_edge("thesis", "antithesis", :contradicts)

    # Process evidence
    {evidence_map, graph} = process_evidence(evidence, graph)

    %__MODULE__{
      id: id,
      thesis: thesis,
      antithesis: antithesis,
      synthesis: nil,
      evidence: evidence_map,
      graph: graph,
      trust_scores: %{},
      metadata: Keyword.get(opts, :metadata, %{}),
      created_at: now,
      updated_at: now,
      version: 1
    }
  end

  @doc """
  Adds evidence to the SNO.
  """
  @spec add_evidence(t(), Evidence.t() | map()) :: t()
  def add_evidence(%__MODULE__{} = sno, evidence) do
    evidence = Evidence.normalize(evidence)
    evidence_id = evidence.id

    # Add to evidence map
    new_evidence = Map.put(sno.evidence, evidence_id, evidence)

    # Add to graph
    new_graph = sno.graph
    |> Graph.add_node(evidence_id, %{type: :evidence, content: evidence.content})

    # Link to claims it supports/contradicts
    new_graph = Enum.reduce(evidence.supports, new_graph, fn claim_id, g ->
      Graph.add_edge(g, evidence_id, claim_id, :supports)
    end)

    new_graph = Enum.reduce(evidence.contradicts, new_graph, fn claim_id, g ->
      Graph.add_edge(g, evidence_id, claim_id, :contradicts)
    end)

    %{sno |
      evidence: new_evidence,
      graph: new_graph,
      updated_at: DateTime.utc_now(),
      version: sno.version + 1
    }
  end

  @doc """
  Sets the synthesis for this SNO.
  """
  @spec set_synthesis(t(), claim(), keyword()) :: t()
  def set_synthesis(%__MODULE__{} = sno, synthesis, opts \\ []) do
    citations = Keyword.get(opts, :citations, [])

    # Add synthesis node to graph
    new_graph = sno.graph
    |> Graph.add_node("synthesis", %{type: :synthesis, content: synthesis})
    |> Graph.add_edge("synthesis", "thesis", :derives)
    |> Graph.add_edge("synthesis", "antithesis", :derives)

    # Add citation edges
    new_graph = Enum.reduce(citations, new_graph, fn evidence_id, g ->
      Graph.add_edge(g, "synthesis", evidence_id, :cites)
    end)

    %{sno |
      synthesis: synthesis,
      graph: new_graph,
      updated_at: DateTime.utc_now(),
      version: sno.version + 1
    }
  end

  @doc """
  Computes trust scores for all nodes.
  """
  @spec compute_trust_scores(t()) :: t()
  def compute_trust_scores(%__MODULE__{} = sno) do
    scores = Trust.compute(sno.graph, sno.evidence)
    %{sno | trust_scores: scores}
  end

  @doc """
  Gets topology metrics for the SNO graph.
  """
  @spec topology_metrics(t()) :: {:ok, map()} | {:error, term()}
  def topology_metrics(%__MODULE__{graph: graph}) do
    CNS.Topology.Analyzer.analyze(graph)
  end

  @doc """
  Validates the SNO structure.
  """
  @spec validate(t()) :: {:ok, t()} | {:error, [String.t()]}
  def validate(%__MODULE__{} = sno) do
    errors = []

    # Check required fields
    errors = if is_nil(sno.thesis), do: ["thesis is required" | errors], else: errors
    errors = if is_nil(sno.antithesis), do: ["antithesis is required" | errors], else: errors

    # Check graph connectivity
    errors = if Graph.disconnected?(sno.graph) do
      ["graph has disconnected components" | errors]
    else
      errors
    end

    # Check for cycles (except valid derivation cycles)
    errors = if Graph.has_invalid_cycles?(sno.graph) do
      ["graph has invalid cycles" | errors]
    else
      errors
    end

    # Check evidence citations
    errors = if sno.synthesis do
      invalid = find_invalid_citations(sno)
      if invalid != [] do
        ["invalid citations: #{inspect(invalid)}" | errors]
      else
        errors
      end
    else
      errors
    end

    if errors == [] do
      {:ok, sno}
    else
      {:error, errors}
    end
  end

  @doc """
  Serializes SNO to JSON.
  """
  @spec to_json(t()) :: String.t()
  def to_json(%__MODULE__{} = sno) do
    Jason.encode!(%{
      id: sno.id,
      thesis: sno.thesis,
      antithesis: sno.antithesis,
      synthesis: sno.synthesis,
      evidence: serialize_evidence(sno.evidence),
      graph: Graph.to_map(sno.graph),
      trust_scores: sno.trust_scores,
      metadata: sno.metadata,
      version: sno.version
    })
  end

  @doc """
  Deserializes SNO from JSON.
  """
  @spec from_json(String.t()) :: {:ok, t()} | {:error, term()}
  def from_json(json) do
    case Jason.decode(json) do
      {:ok, data} -> {:ok, from_map(data)}
      {:error, _} = error -> error
    end
  end

  # Private functions

  defp generate_id do
    :crypto.strong_rand_bytes(16) |> Base.encode16(case: :lower)
  end

  defp process_evidence(evidence_list, graph) do
    Enum.reduce(evidence_list, {%{}, graph}, fn ev, {map, g} ->
      evidence = Evidence.normalize(ev)
      new_map = Map.put(map, evidence.id, evidence)
      new_g = Graph.add_node(g, evidence.id, %{type: :evidence, content: evidence.content})
      {new_map, new_g}
    end)
  end

  defp find_invalid_citations(sno) do
    cited_ids = Graph.edges_from(sno.graph, "synthesis")
    |> Enum.filter(fn {_, _, type} -> type == :cites end)
    |> Enum.map(fn {_, target, _} -> target end)

    Enum.filter(cited_ids, fn id ->
      not Map.has_key?(sno.evidence, id)
    end)
  end

  defp serialize_evidence(evidence_map) do
    Map.new(evidence_map, fn {id, ev} ->
      {id, Evidence.to_map(ev)}
    end)
  end

  defp from_map(data) do
    %__MODULE__{
      id: data["id"],
      thesis: data["thesis"],
      antithesis: data["antithesis"],
      synthesis: data["synthesis"],
      evidence: deserialize_evidence(data["evidence"]),
      graph: Graph.from_map(data["graph"]),
      trust_scores: data["trust_scores"],
      metadata: data["metadata"],
      version: data["version"]
    }
  end

  defp deserialize_evidence(evidence_data) do
    Map.new(evidence_data, fn {id, ev} ->
      {id, Evidence.from_map(ev)}
    end)
  end
end
```

## Graph Representation

```elixir
defmodule CNS.SNO.Graph do
  @moduledoc """
  Graph representation for SNO using libgraph.
  """

  @type t :: Graph.t()
  @type node_id :: String.t()
  @type node_data :: map()
  @type edge_type :: atom()

  @doc """
  Creates a new empty graph.
  """
  @spec new() :: t()
  def new do
    Graph.new(type: :directed)
  end

  @doc """
  Adds a node with metadata.
  """
  @spec add_node(t(), node_id(), node_data()) :: t()
  def add_node(graph, id, data) do
    Graph.add_vertex(graph, id, data)
  end

  @doc """
  Adds a typed edge between nodes.
  """
  @spec add_edge(t(), node_id(), node_id(), edge_type()) :: t()
  def add_edge(graph, from, to, type) do
    Graph.add_edge(graph, from, to, label: type)
  end

  @doc """
  Gets all nodes of a specific type.
  """
  @spec nodes_of_type(t(), atom()) :: [node_id()]
  def nodes_of_type(graph, type) do
    graph
    |> Graph.vertices()
    |> Enum.filter(fn v ->
      data = Graph.vertex_labels(graph, v)
      data[:type] == type
    end)
  end

  @doc """
  Gets edges from a specific node.
  """
  @spec edges_from(t(), node_id()) :: [{node_id(), node_id(), edge_type()}]
  def edges_from(graph, node_id) do
    Graph.out_edges(graph, node_id)
    |> Enum.map(fn edge ->
      {edge.v1, edge.v2, edge.label}
    end)
  end

  @doc """
  Gets edges to a specific node.
  """
  @spec edges_to(t(), node_id()) :: [{node_id(), node_id(), edge_type()}]
  def edges_to(graph, node_id) do
    Graph.in_edges(graph, node_id)
    |> Enum.map(fn edge ->
      {edge.v1, edge.v2, edge.label}
    end)
  end

  @doc """
  Checks if graph has disconnected components.
  """
  @spec disconnected?(t()) :: boolean()
  def disconnected?(graph) do
    components = Graph.components(graph)
    length(components) > 1
  end

  @doc """
  Checks for invalid cycles (not derivation chains).
  """
  @spec has_invalid_cycles?(t()) :: boolean()
  def has_invalid_cycles?(graph) do
    case Graph.is_acyclic?(graph) do
      true -> false
      false ->
        # Check if cycles are only in valid derivation patterns
        cycles = find_cycles(graph)
        Enum.any?(cycles, &invalid_cycle?/1)
    end
  end

  @doc """
  Gets the subgraph supporting a specific claim.
  """
  @spec support_subgraph(t(), node_id()) :: t()
  def support_subgraph(graph, claim_id) do
    # Find all evidence nodes that support this claim
    supporting = edges_to(graph, claim_id)
    |> Enum.filter(fn {_, _, type} -> type == :supports end)
    |> Enum.map(fn {from, _, _} -> from end)

    # Create subgraph
    vertices = [claim_id | supporting]
    Graph.subgraph(graph, vertices)
  end

  @doc """
  Converts graph to a serializable map.
  """
  @spec to_map(t()) :: map()
  def to_map(graph) do
    vertices = Graph.vertices(graph)
    |> Enum.map(fn v ->
      %{id: v, data: Graph.vertex_labels(graph, v)}
    end)

    edges = Graph.edges(graph)
    |> Enum.map(fn edge ->
      %{from: edge.v1, to: edge.v2, type: edge.label}
    end)

    %{vertices: vertices, edges: edges}
  end

  @doc """
  Creates graph from serialized map.
  """
  @spec from_map(map()) :: t()
  def from_map(data) do
    graph = new()

    # Add vertices
    graph = Enum.reduce(data["vertices"], graph, fn v, g ->
      add_node(g, v["id"], v["data"])
    end)

    # Add edges
    Enum.reduce(data["edges"], graph, fn e, g ->
      add_edge(g, e["from"], e["to"], String.to_atom(e["type"]))
    end)
  end

  # Private functions

  defp find_cycles(graph) do
    # Use DFS to find cycles
    Graph.vertices(graph)
    |> Enum.flat_map(&find_cycles_from(graph, &1, [], MapSet.new()))
  end

  defp find_cycles_from(_graph, _vertex, _path, _visited) do
    # Simplified: return empty for now
    []
  end

  defp invalid_cycle?(_cycle) do
    # Check if cycle represents circular reasoning
    false
  end
end
```

## Evidence Struct

```elixir
defmodule CNS.SNO.Evidence do
  @moduledoc """
  Evidence item with source tracking and trust metadata.
  """

  @type t :: %__MODULE__{
    id: String.t(),
    content: String.t(),
    source: String.t(),
    source_type: :document | :study | :expert | :database,
    supports: [String.t()],
    contradicts: [String.t()],
    trust_score: float() | nil,
    metadata: map()
  }

  defstruct [
    :id,
    :content,
    :source,
    source_type: :document,
    supports: [],
    contradicts: [],
    trust_score: nil,
    metadata: %{}
  ]

  @doc """
  Normalizes evidence input to struct.
  """
  @spec normalize(map() | t()) :: t()
  def normalize(%__MODULE__{} = evidence), do: evidence
  def normalize(map) when is_map(map) do
    %__MODULE__{
      id: Map.get(map, :id) || Map.get(map, "id") || generate_id(),
      content: Map.get(map, :content) || Map.get(map, "content"),
      source: Map.get(map, :source) || Map.get(map, "source"),
      source_type: normalize_source_type(map),
      supports: Map.get(map, :supports) || Map.get(map, "supports") || [],
      contradicts: Map.get(map, :contradicts) || Map.get(map, "contradicts") || [],
      trust_score: Map.get(map, :trust_score) || Map.get(map, "trust_score"),
      metadata: Map.get(map, :metadata) || Map.get(map, "metadata") || %{}
    }
  end

  @doc """
  Converts evidence to serializable map.
  """
  @spec to_map(t()) :: map()
  def to_map(%__MODULE__{} = ev) do
    %{
      id: ev.id,
      content: ev.content,
      source: ev.source,
      source_type: Atom.to_string(ev.source_type),
      supports: ev.supports,
      contradicts: ev.contradicts,
      trust_score: ev.trust_score,
      metadata: ev.metadata
    }
  end

  @doc """
  Creates evidence from serialized map.
  """
  @spec from_map(map()) :: t()
  def from_map(data) do
    normalize(data)
  end

  defp generate_id do
    :crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)
  end

  defp normalize_source_type(map) do
    case Map.get(map, :source_type) || Map.get(map, "source_type") do
      nil -> :document
      type when is_atom(type) -> type
      type when is_binary(type) -> String.to_atom(type)
    end
  end
end
```

## Trust Scoring

```elixir
defmodule CNS.SNO.Trust do
  @moduledoc """
  Trust scoring and propagation for SNO nodes.
  """

  alias CNS.SNO.{Graph, Evidence}

  @doc """
  Computes trust scores for all nodes in the graph.
  """
  @spec compute(Graph.t(), %{String.t() => Evidence.t()}) :: %{String.t() => float()}
  def compute(graph, evidence_map) do
    # Initialize with evidence trust scores
    initial_scores = initialize_scores(graph, evidence_map)

    # Propagate trust through graph
    propagate(graph, initial_scores, iterations: 10)
  end

  defp initialize_scores(graph, evidence_map) do
    Graph.nodes_of_type(graph, :all)
    |> Enum.map(fn node_id ->
      score = case Map.get(evidence_map, node_id) do
        nil -> 0.5  # Default for non-evidence nodes
        evidence -> evidence.trust_score || compute_evidence_trust(evidence)
      end
      {node_id, score}
    end)
    |> Map.new()
  end

  defp compute_evidence_trust(evidence) do
    base_score = case evidence.source_type do
      :study -> 0.8
      :document -> 0.6
      :expert -> 0.7
      :database -> 0.75
    end

    # Adjust based on metadata
    adjustments = [
      peer_reviewed_adjustment(evidence),
      recency_adjustment(evidence),
      citation_count_adjustment(evidence)
    ]

    adjusted = Enum.reduce(adjustments, base_score, &(&1 + &2))
    min(max(adjusted, 0.0), 1.0)
  end

  defp peer_reviewed_adjustment(evidence) do
    if evidence.metadata[:peer_reviewed], do: 0.1, else: 0.0
  end

  defp recency_adjustment(evidence) do
    case evidence.metadata[:year] do
      nil -> 0.0
      year when year >= 2020 -> 0.05
      year when year >= 2015 -> 0.0
      _ -> -0.05
    end
  end

  defp citation_count_adjustment(evidence) do
    case evidence.metadata[:citation_count] do
      nil -> 0.0
      count when count > 100 -> 0.1
      count when count > 10 -> 0.05
      _ -> 0.0
    end
  end

  defp propagate(graph, scores, opts) do
    iterations = Keyword.get(opts, :iterations, 10)
    damping = Keyword.get(opts, :damping, 0.85)

    Enum.reduce(1..iterations, scores, fn _iteration, current_scores ->
      # Update each node based on incoming edges
      Graph.vertices(graph)
      |> Enum.map(fn node_id ->
        incoming = Graph.edges_to(graph, node_id)

        new_score = if incoming == [] do
          current_scores[node_id]
        else
          # Weighted average of incoming node scores
          incoming_contribution = Enum.map(incoming, fn {from, _, edge_type} ->
            weight = edge_weight(edge_type)
            current_scores[from] * weight
          end)
          |> Enum.sum()
          |> Kernel./(length(incoming))

          # Combine with damping
          damping * incoming_contribution + (1 - damping) * current_scores[node_id]
        end

        {node_id, new_score}
      end)
      |> Map.new()
    end)
  end

  defp edge_weight(:supports), do: 1.0
  defp edge_weight(:contradicts), do: -0.5
  defp edge_weight(:cites), do: 0.8
  defp edge_weight(:derives), do: 0.6
  defp edge_weight(:elaborates), do: 0.4
  defp edge_weight(_), do: 0.5
end
```

## Temporal Evolution

```elixir
defmodule CNS.SNO.Temporal do
  @moduledoc """
  Tracks temporal evolution of SNOs.
  """

  @type version_entry :: %{
    version: pos_integer(),
    timestamp: DateTime.t(),
    changes: [change()],
    sno_snapshot: CNS.SNO.t() | nil
  }

  @type change ::
    {:add_evidence, String.t()} |
    {:remove_evidence, String.t()} |
    {:set_synthesis, String.t()} |
    {:update_trust, map()}

  @doc """
  Records a change to the SNO history.
  """
  @spec record_change(CNS.SNO.t(), change()) :: {CNS.SNO.t(), version_entry()}
  def record_change(sno, change) do
    entry = %{
      version: sno.version,
      timestamp: DateTime.utc_now(),
      changes: [change],
      sno_snapshot: nil  # Optional full snapshot
    }

    {sno, entry}
  end

  @doc """
  Gets the evolution timeline for an SNO.
  """
  @spec get_timeline(String.t()) :: [version_entry()]
  def get_timeline(_sno_id) do
    # Query from storage
    []
  end

  @doc """
  Reconstructs SNO at a specific version.
  """
  @spec at_version(String.t(), pos_integer()) :: {:ok, CNS.SNO.t()} | {:error, term()}
  def at_version(_sno_id, _version) do
    # Replay changes to reconstruct
    {:error, :not_implemented}
  end
end
```

## Usage Examples

```elixir
# Create a new SNO
sno = CNS.SNO.new(
  thesis: "Vaccine X is safe and effective for all age groups",
  antithesis: "Vaccine X has significant side effects in elderly patients",
  evidence: [
    %{
      id: "e1",
      content: "Clinical trial with 10,000 participants showed 95% efficacy",
      source: "NEJM",
      source_type: :study,
      supports: ["thesis"],
      metadata: %{peer_reviewed: true, year: 2023}
    },
    %{
      id: "e2",
      content: "Post-market surveillance found increased myocarditis risk in 65+ patients",
      source: "FDA Report",
      source_type: :database,
      supports: ["antithesis"],
      metadata: %{year: 2024}
    }
  ]
)

# Add more evidence
sno = CNS.SNO.add_evidence(sno, %{
  id: "e3",
  content: "Meta-analysis confirms efficacy but recommends caution for elderly",
  source: "Cochrane Review",
  source_type: :study,
  supports: ["thesis", "antithesis"],
  metadata: %{peer_reviewed: true, year: 2024, citation_count: 150}
})

# Set synthesis
sno = CNS.SNO.set_synthesis(sno,
  "Vaccine X is effective for most populations but requires age-specific risk assessment for patients over 65",
  citations: ["e1", "e2", "e3"]
)

# Compute trust scores
sno = CNS.SNO.compute_trust_scores(sno)

# Validate
{:ok, sno} = CNS.SNO.validate(sno)

# Get topology metrics
{:ok, metrics} = CNS.SNO.topology_metrics(sno)
IO.inspect(metrics)
# %{
#   betti_0: 1,  # Connected
#   betti_1: 0,  # No cycles
#   nodes: 6,
#   edges: 8,
#   density: 0.27
# }

# Serialize
json = CNS.SNO.to_json(sno)
{:ok, restored} = CNS.SNO.from_json(json)
```

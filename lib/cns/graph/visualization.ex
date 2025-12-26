defmodule CNS.Graph.Visualization do
  @moduledoc """
  Export reasoning graphs to visualization formats.

  Supports:
  - DOT format (Graphviz)
  - Mermaid format
  - Plain text representation
  """

  @doc """
  Export graph to DOT format for Graphviz.
  """
  @spec to_dot(Graph.t(), keyword()) :: String.t()
  def to_dot(graph, opts \\ []) do
    name = Keyword.get(opts, :name, "reasoning_graph")
    rankdir = Keyword.get(opts, :rankdir, "TB")

    vertices = format_dot_vertices(graph)
    edges = format_dot_edges(graph)

    """
    digraph #{name} {
      rankdir=#{rankdir};
      node [shape=box, style=rounded];

    #{vertices}
    #{edges}
    }
    """
  end

  @doc """
  Export graph to Mermaid format for markdown rendering.
  """
  @spec to_mermaid(Graph.t(), keyword()) :: String.t()
  def to_mermaid(graph, opts \\ []) do
    direction = Keyword.get(opts, :direction, "TD")

    vertices = format_mermaid_vertices(graph)
    edges = format_mermaid_edges(graph)

    """
    graph #{direction}
    #{vertices}
    #{edges}
    """
  end

  @doc """
  Generate a plain text representation of the graph.
  """
  @spec to_text(Graph.t()) :: String.t()
  def to_text(graph) do
    vertex_count = Graph.num_vertices(graph)
    edge_count = Graph.num_edges(graph)

    vertices_str =
      graph
      |> Graph.vertices()
      |> Enum.map_join("\n", fn v ->
        labels = Graph.vertex_labels(graph, v)
        type = Keyword.get(labels, :type, :unknown)
        "  - #{v} (#{type})"
      end)

    edges_str =
      graph
      |> Graph.edges()
      |> Enum.map_join("\n", fn edge ->
        "  - #{edge.v1} --[#{edge.label || ""}]--> #{edge.v2}"
      end)

    """
    Graph Summary:
      Vertices: #{vertex_count}
      Edges: #{edge_count}

    Vertices:
    #{vertices_str}

    Edges:
    #{edges_str}
    """
  end

  @doc """
  Export to a specific file format.
  """
  @spec export(Graph.t(), String.t(), keyword()) :: {:ok, String.t()} | {:error, term()}
  def export(graph, path, opts \\ []) do
    format = Keyword.get(opts, :format) || detect_format(path)

    content =
      case format do
        :dot -> to_dot(graph, opts)
        :mermaid -> to_mermaid(graph, opts)
        :txt -> to_text(graph)
        _ -> to_text(graph)
      end

    case File.write(path, content) do
      :ok -> {:ok, path}
      error -> error
    end
  end

  # Private functions

  defp format_dot_vertices(graph) do
    graph
    |> Graph.vertices()
    |> Enum.map_join("\n", fn v ->
      labels = Graph.vertex_labels(graph, v)
      type = Keyword.get(labels, :type, :unknown)
      text = Keyword.get(labels, :text) || Keyword.get(labels, :content) || ""

      # Truncate text for display
      display_text = truncate(text, 30)

      style =
        case type do
          :claim -> "fillcolor=lightblue, style=\"rounded,filled\""
          :evidence -> "fillcolor=lightyellow, style=\"rounded,filled\""
          _ -> ""
        end

      id = sanitize_id(v)
      ~s(  #{id} [label="#{display_text}", #{style}];)
    end)
  end

  defp format_dot_edges(graph) do
    graph
    |> Graph.edges()
    |> Enum.map_join("\n", fn edge ->
      v1 = sanitize_id(edge.v1)
      v2 = sanitize_id(edge.v2)
      label = edge.label || ""

      style =
        case label do
          :supports -> "color=green"
          :contradicts -> "color=red, style=dashed"
          :child_of -> "color=blue"
          :cites -> "color=gray"
          _ -> ""
        end

      ~s(  #{v1} -> #{v2} [label="#{label}", #{style}];)
    end)
  end

  defp format_mermaid_vertices(graph) do
    graph
    |> Graph.vertices()
    |> Enum.map_join("\n", fn v ->
      labels = Graph.vertex_labels(graph, v)
      type = Keyword.get(labels, :type, :unknown)
      text = Keyword.get(labels, :text) || Keyword.get(labels, :content) || to_string(v)

      display_text = truncate(text, 20)
      id = sanitize_mermaid_id(v)

      case type do
        :claim -> "    #{id}[\"#{display_text}\"]"
        :evidence -> "    #{id}([\"#{display_text}\"])"
        _ -> "    #{id}[\"#{display_text}\"]"
      end
    end)
  end

  defp format_mermaid_edges(graph) do
    graph
    |> Graph.edges()
    |> Enum.map_join("\n", fn edge ->
      v1 = sanitize_mermaid_id(edge.v1)
      v2 = sanitize_mermaid_id(edge.v2)
      label = edge.label || ""

      arrow =
        case label do
          :supports -> "-->|supports|"
          :contradicts -> "-.->|contradicts|"
          :child_of -> "-->|child|"
          :cites -> "-.->|cites|"
          _ -> "-->"
        end

      "    #{v1} #{arrow} #{v2}"
    end)
  end

  defp sanitize_id(id) do
    id
    |> to_string()
    |> String.replace(~r/[^a-zA-Z0-9_]/, "_")
    |> then(fn s ->
      if String.match?(s, ~r/^[0-9]/) do
        "v_" <> s
      else
        s
      end
    end)
  end

  defp sanitize_mermaid_id(id) do
    id
    |> to_string()
    |> String.replace(~r/[^a-zA-Z0-9]/, "")
    |> then(fn s ->
      if String.match?(s, ~r/^[0-9]/) do
        "v" <> s
      else
        s
      end
    end)
  end

  defp truncate(text, max) do
    text = text |> String.replace("\"", "'") |> String.replace("\n", " ")

    if String.length(text) > max do
      String.slice(text, 0, max) <> "..."
    else
      text
    end
  end

  defp detect_format(path) do
    case Path.extname(path) do
      ".dot" -> :dot
      ".gv" -> :dot
      ".md" -> :mermaid
      ".mermaid" -> :mermaid
      _ -> :txt
    end
  end
end

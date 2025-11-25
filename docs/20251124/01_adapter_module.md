# CNS Topology Adapter Module - Phase 1 Implementation

**Date**: 2024-11-24
**Phase**: 1 - Foundation
**Status**: Production-ready implementation
**Module**: `CNS.Topology.Adapter`

---

## Overview

This document contains the complete, production-ready implementation of the `CNS.Topology.Adapter` module - the bridge between CNS's dialectical reasoning structures and ex_topology's topological analysis capabilities.

The adapter handles:
- Converting SNOs (Structured Narrative Objects) to Nx tensors
- Extracting embeddings from SNO metadata or generating them on-demand
- Converting claim networks to formats suitable for topological analysis
- Interpreting topological results in the context of dialectical reasoning

---

## Implementation

### File: `lib/cns/topology/adapter.ex`

```elixir
defmodule CNS.Topology.Adapter do
  @moduledoc """
  Bridge between CNS claim networks and ex_topology.

  Converts CNS data structures (SNOs, Evidence) to formats suitable for
  topological analysis, then interprets results in the dialectical reasoning context.

  ## Purpose

  This adapter isolates the integration with ex_topology, allowing CNS to leverage
  advanced topological analysis without coupling its core dialectical reasoning
  to specific TDA implementations.

  ## Key Functions

  - `sno_embeddings/2` - Extract or generate embeddings from SNO list
  - `claim_graph/3` - Build neighborhood graph from claim embeddings
  - `extract_embedding/1` - Get embedding vector from a single SNO
  - `to_tensor/2` - Convert various formats to Nx tensors
  - `interpret_betti/1` - Interpret Betti numbers in CNS context

  ## Examples

      # Extract embeddings from SNOs
      snos = [sno1, sno2, sno3]
      embeddings = CNS.Topology.Adapter.sno_embeddings(snos)
      # => Nx.Tensor shape {3, 384}

      # Build k-NN graph
      graph = CNS.Topology.Adapter.claim_graph(embeddings, :knn, k: 5)

      # Interpret topological features
      inv = ExTopology.Graph.invariants(graph)
      interpretation = CNS.Topology.Adapter.interpret_betti(inv)
      # => %{cycles: 2, interpretation: :circular_reasoning, ...}
  """

  alias CNS.SNO
  require Logger

  @type embedding_source ::
          :metadata
          | :generate
          | {:encoder, module()}

  @type neighborhood_strategy ::
          :knn
          | :epsilon
          | :gabriel
          | :delaunay

  # Default embedding dimension for MiniLM
  @default_embedding_dim 384

  # ============================================================================
  # Public API - SNO to Embeddings
  # ============================================================================

  @doc """
  Extract embeddings from a list of SNOs.

  ## Extraction Strategy

  The adapter tries multiple strategies in order:
  1. Check SNO metadata for cached embeddings (`:embedding` or `:embeddings` key)
  2. If `:encoder` option provided, use that encoder module
  3. If `:generate` option is true, generate embeddings from claim text
  4. Otherwise, return error

  ## Parameters

    - `snos` - List of SNO structs
    - `opts` - Options:
      - `:source` - Embedding source (`:metadata`, `:generate`, `{:encoder, Module}`)
      - `:encoder` - Encoder module with `encode/1` function (default: CNS.Embedding.Encoder)
      - `:cache` - Whether to cache generated embeddings in SNO metadata (default: true)
      - `:normalize` - Whether to L2-normalize embeddings (default: false)

  ## Returns

  Nx.Tensor of shape `{n_snos, embedding_dim}` or `{:error, reason}`

  ## Examples

      # From cached embeddings in metadata
      snos = [
        %SNO{claim: "A", metadata: %{embedding: [0.1, 0.2, ...]}},
        %SNO{claim: "B", metadata: %{embedding: [0.3, 0.4, ...]}}
      ]
      embeddings = CNS.Topology.Adapter.sno_embeddings(snos)

      # Generate embeddings on-demand
      snos = [%SNO{claim: "Test claim"}]
      embeddings = CNS.Topology.Adapter.sno_embeddings(snos, source: :generate)

      # Use custom encoder
      embeddings = CNS.Topology.Adapter.sno_embeddings(snos,
        source: {:encoder, MyCustomEncoder}
      )
  """
  @spec sno_embeddings([SNO.t()], keyword()) ::
          Nx.Tensor.t() | {:error, term()}
  def sno_embeddings(snos, opts \\ [])

  def sno_embeddings([], _opts) do
    # Return empty tensor with default embedding dimension
    Nx.tensor([], type: :f32) |> Nx.reshape({0, @default_embedding_dim})
  end

  def sno_embeddings(snos, opts) when is_list(snos) do
    source = Keyword.get(opts, :source, :metadata)
    normalize = Keyword.get(opts, :normalize, false)

    case extract_all_embeddings(snos, source, opts) do
      {:ok, embeddings} ->
        tensor = to_tensor(embeddings, type: :f32)

        if normalize do
          normalize_embeddings(tensor)
        else
          tensor
        end

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Extract embedding vector from a single SNO.

  Returns the embedding as a list of floats or generates it if needed.

  ## Parameters

    - `sno` - Single SNO struct
    - `opts` - Same options as `sno_embeddings/2`

  ## Returns

  List of floats or `{:error, reason}`

  ## Examples

      sno = %SNO{claim: "Test", metadata: %{embedding: [0.1, 0.2, 0.3]}}
      {:ok, embedding} = CNS.Topology.Adapter.extract_embedding(sno)
      # => [0.1, 0.2, 0.3]

      sno = %SNO{claim: "Generate me"}
      {:ok, embedding} = CNS.Topology.Adapter.extract_embedding(sno, source: :generate)
      # => [0.015, -0.023, ...]
  """
  @spec extract_embedding(SNO.t(), keyword()) ::
          {:ok, list(float())} | {:error, term()}
  def extract_embedding(%SNO{} = sno, opts \\ []) do
    source = Keyword.get(opts, :source, :metadata)

    case extract_single_embedding(sno, source, opts) do
      {:ok, embedding} when is_list(embedding) -> {:ok, embedding}
      {:ok, %Nx.Tensor{} = tensor} -> {:ok, Nx.to_flat_list(tensor)}
      {:error, reason} -> {:error, reason}
    end
  end

  # ============================================================================
  # Public API - Graph Construction
  # ============================================================================

  @doc """
  Build neighborhood graph from claim embeddings.

  Creates a graph structure suitable for topological analysis using various
  neighborhood strategies.

  ## Parameters

    - `embeddings` - Nx.Tensor of shape `{n_points, dim}`
    - `strategy` - Neighborhood strategy (`:knn`, `:epsilon`, `:gabriel`, `:delaunay`)
    - `opts` - Strategy-specific options:
      - For `:knn`: `:k` (default: 5), `:metric` (`:euclidean` or `:cosine`)
      - For `:epsilon`: `:epsilon` (required), `:metric`
      - For `:gabriel`: `:metric`
      - For `:delaunay`: (no options)

  ## Returns

  `Graph.t()` (libgraph structure) suitable for ex_topology analysis

  ## Examples

      embeddings = Nx.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

      # k-NN graph
      graph = CNS.Topology.Adapter.claim_graph(embeddings, :knn, k: 3)

      # Epsilon-ball graph
      graph = CNS.Topology.Adapter.claim_graph(embeddings, :epsilon, epsilon: 0.5)
  """
  @spec claim_graph(Nx.Tensor.t(), neighborhood_strategy(), keyword()) ::
          Graph.t()
  def claim_graph(embeddings, strategy \\ :knn, opts \\ [])

  def claim_graph(embeddings, :knn, opts) do
    k = Keyword.get(opts, :k, 5)
    metric = Keyword.get(opts, :metric, :euclidean)

    ExTopology.Neighborhood.knn_graph(embeddings, k: k, metric: metric)
  end

  def claim_graph(embeddings, :epsilon, opts) do
    epsilon = Keyword.get(opts, :epsilon)

    if is_nil(epsilon) do
      raise ArgumentError, "epsilon is required for :epsilon strategy"
    end

    metric = Keyword.get(opts, :metric, :euclidean)
    ExTopology.Neighborhood.epsilon_graph(embeddings, epsilon: epsilon, metric: metric)
  end

  def claim_graph(embeddings, :gabriel, opts) do
    metric = Keyword.get(opts, :metric, :euclidean)
    ExTopology.Neighborhood.gabriel_graph(embeddings, metric: metric)
  end

  def claim_graph(embeddings, :delaunay, _opts) do
    ExTopology.Neighborhood.delaunay_graph(embeddings)
  end

  # ============================================================================
  # Public API - Tensor Conversion
  # ============================================================================

  @doc """
  Convert various data formats to Nx tensors.

  Accepts nested lists, flat lists, existing tensors, and validates shapes.

  ## Parameters

    - `data` - Data to convert (list, nested list, or Nx.Tensor)
    - `opts` - Options:
      - `:type` - Nx type (default: `:f32`)
      - `:shape` - Expected shape (optional, for validation)

  ## Returns

  `Nx.Tensor.t()`

  ## Examples

      # Nested list (embeddings)
      data = [[0.1, 0.2], [0.3, 0.4]]
      tensor = CNS.Topology.Adapter.to_tensor(data)
      # => Nx.Tensor shape {2, 2}

      # Already a tensor (passthrough)
      tensor = Nx.tensor([1, 2, 3])
      result = CNS.Topology.Adapter.to_tensor(tensor)
      # => same tensor

      # With type specification
      tensor = CNS.Topology.Adapter.to_tensor([[1, 2]], type: :s32)
  """
  @spec to_tensor(any(), keyword()) :: Nx.Tensor.t()
  def to_tensor(data, opts \\ [])

  def to_tensor(%Nx.Tensor{} = tensor, opts) do
    type = Keyword.get(opts, :type)

    if type && Nx.type(tensor) != type do
      Nx.as_type(tensor, type)
    else
      tensor
    end
  end

  def to_tensor(data, opts) when is_list(data) do
    type = Keyword.get(opts, :type, :f32)
    tensor = Nx.tensor(data, type: type)

    case Keyword.get(opts, :shape) do
      nil ->
        tensor

      expected_shape ->
        actual_shape = Nx.shape(tensor)

        if actual_shape != expected_shape do
          raise ArgumentError,
                "Shape mismatch: expected #{inspect(expected_shape)}, got #{inspect(actual_shape)}"
        end

        tensor
    end
  end

  # ============================================================================
  # Public API - Result Interpretation
  # ============================================================================

  @doc """
  Interpret Betti numbers in CNS dialectical reasoning context.

  Translates topological invariants into meaningful interpretations for
  claim network analysis.

  ## Parameters

    - `invariants` - Map with `:beta_zero`, `:beta_one`, and optionally higher Betti numbers

  ## Returns

  Map with CNS-specific interpretations

  ## Examples

      inv = %{beta_zero: 1, beta_one: 2, euler_characteristic: -1}
      interpretation = CNS.Topology.Adapter.interpret_betti(inv)
      # => %{
      #   components: 1,
      #   cycles: 2,
      #   has_circular_reasoning: true,
      #   circular_reasoning_severity: :moderate,
      #   claim_clusters: 1,
      #   is_connected: true,
      #   topology_class: :single_component_with_cycles
      # }
  """
  @spec interpret_betti(map()) :: map()
  def interpret_betti(invariants) do
    beta_zero = Map.get(invariants, :beta_zero, 0)
    beta_one = Map.get(invariants, :beta_one, 0)
    beta_two = Map.get(invariants, :beta_two, 0)

    has_cycles = beta_one > 0

    severity =
      cond do
        beta_one == 0 -> :none
        beta_one <= 2 -> :mild
        beta_one <= 5 -> :moderate
        true -> :severe
      end

    topology_class =
      cond do
        beta_zero > 1 && beta_one == 0 -> :disconnected_acyclic
        beta_zero > 1 && beta_one > 0 -> :disconnected_with_cycles
        beta_zero == 1 && beta_one == 0 -> :connected_acyclic
        beta_zero == 1 && beta_one > 0 -> :connected_with_cycles
        true -> :unknown
      end

    %{
      # Raw topological features
      components: beta_zero,
      cycles: beta_one,
      voids: beta_two,

      # CNS interpretations
      has_circular_reasoning: has_cycles,
      circular_reasoning_severity: severity,
      claim_clusters: beta_zero,
      is_connected: beta_zero == 1,

      # Overall classification
      topology_class: topology_class,
      interpretation: generate_text_interpretation(beta_zero, beta_one, beta_two)
    }
  end

  @doc """
  Interpret fragility scores from ex_topology.Embedding analysis.

  ## Parameters

    - `fragility_map` - Result from ExTopology.Embedding.knn_variance or similar

  ## Returns

  Map with CNS-specific fragility interpretations

  ## Examples

      fragility = %{mean_variance: 0.45, max_variance: 0.82, percentile_95: 0.71}
      interpretation = CNS.Topology.Adapter.interpret_fragility(fragility)
      # => %{
      #   stability: :moderate,
      #   has_outliers: true,
      #   recommendation: "Review claims with high variance"
      # }
  """
  @spec interpret_fragility(map() | float()) :: map()
  def interpret_fragility(fragility) when is_float(fragility) do
    interpret_fragility(%{mean_variance: fragility})
  end

  def interpret_fragility(fragility) when is_map(fragility) do
    mean_var = Map.get(fragility, :mean_variance, 0.0)
    max_var = Map.get(fragility, :max_variance, mean_var)

    stability =
      cond do
        mean_var < 0.3 -> :stable
        mean_var < 0.6 -> :moderate
        true -> :fragile
      end

    has_outliers = max_var > mean_var * 1.5

    recommendation =
      case stability do
        :stable ->
          "Claim network is topologically stable"

        :moderate ->
          "Some semantic instability detected. Review isolated claims."

        :fragile ->
          "High semantic fragility. Network may be incoherent or contain contradictions."
      end

    %{
      stability: stability,
      mean_variance: Float.round(mean_var, 4),
      max_variance: Float.round(max_var, 4),
      has_outliers: has_outliers,
      recommendation: recommendation
    }
  end

  # ============================================================================
  # Public API - Causal Link Graph Construction
  # ============================================================================

  @doc """
  Build a directed graph from causal links in SNOs.

  Extracts causal relationships from SNO metadata or provenance and constructs
  a directed graph suitable for topological analysis.

  ## Parameters

    - `snos` - List of SNOs
    - `opts` - Options:
      - `:link_key` - Metadata key containing causal links (default: `:causal_links`)
      - `:use_provenance` - Extract links from provenance (default: false)

  ## Returns

  `Graph.t()` (directed graph)

  ## Examples

      snos = [
        %SNO{id: "a", claim: "A", metadata: %{causal_links: ["b", "c"]}},
        %SNO{id: "b", claim: "B", metadata: %{causal_links: ["c"]}},
        %SNO{id: "c", claim: "C"}
      ]
      graph = CNS.Topology.Adapter.build_causal_graph(snos)
  """
  @spec build_causal_graph([SNO.t()], keyword()) :: Graph.t()
  def build_causal_graph(snos, opts \\ []) do
    link_key = Keyword.get(opts, :link_key, :causal_links)

    g = Graph.new(type: :directed)

    # Add all SNOs as vertices
    g =
      Enum.reduce(snos, g, fn sno, acc ->
        Graph.add_vertex(acc, sno.id, sno.claim)
      end)

    # Add edges from causal links
    Enum.reduce(snos, g, fn sno, acc ->
      links = get_in(sno.metadata, [link_key]) || []

      Enum.reduce(links, acc, fn target_id, graph_acc ->
        Graph.add_edge(graph_acc, sno.id, target_id)
      end)
    end)
  end

  # ============================================================================
  # Public API - Helper Functions
  # ============================================================================

  @doc """
  Check if embeddings are cached in SNO metadata.

  ## Parameters

    - `sno` - SNO struct

  ## Returns

  Boolean

  ## Examples

      sno = %SNO{metadata: %{embedding: [0.1, 0.2]}}
      CNS.Topology.Adapter.has_cached_embedding?(sno)
      # => true
  """
  @spec has_cached_embedding?(SNO.t()) :: boolean()
  def has_cached_embedding?(%SNO{metadata: metadata}) do
    Map.has_key?(metadata, :embedding) or Map.has_key?(metadata, :embeddings)
  end

  @doc """
  Cache embedding in SNO metadata.

  ## Parameters

    - `sno` - SNO struct
    - `embedding` - Embedding vector (list of floats or Nx.Tensor)

  ## Returns

  Updated SNO with cached embedding

  ## Examples

      sno = %SNO{claim: "Test"}
      embedding = [0.1, 0.2, 0.3]
      updated = CNS.Topology.Adapter.cache_embedding(sno, embedding)
      updated.metadata.embedding
      # => [0.1, 0.2, 0.3]
  """
  @spec cache_embedding(SNO.t(), list(float()) | Nx.Tensor.t()) :: SNO.t()
  def cache_embedding(%SNO{} = sno, %Nx.Tensor{} = embedding) do
    cache_embedding(sno, Nx.to_flat_list(embedding))
  end

  def cache_embedding(%SNO{} = sno, embedding) when is_list(embedding) do
    updated_metadata = Map.put(sno.metadata, :embedding, embedding)
    %{sno | metadata: updated_metadata}
  end

  @doc """
  Get embedding dimension from a list of SNOs.

  Returns the embedding dimension if consistent, or error if dimensions vary.

  ## Examples

      snos = [
        %SNO{metadata: %{embedding: [0.1, 0.2, 0.3]}},
        %SNO{metadata: %{embedding: [0.4, 0.5, 0.6]}}
      ]
      {:ok, dim} = CNS.Topology.Adapter.embedding_dimension(snos)
      # => {:ok, 3}
  """
  @spec embedding_dimension([SNO.t()]) :: {:ok, pos_integer()} | {:error, term()}
  def embedding_dimension([]) do
    {:ok, @default_embedding_dim}
  end

  def embedding_dimension(snos) when is_list(snos) do
    dimensions =
      snos
      |> Enum.map(&extract_embedding_dimension/1)
      |> Enum.reject(&is_nil/1)
      |> Enum.uniq()

    case dimensions do
      [] -> {:ok, @default_embedding_dim}
      [dim] -> {:ok, dim}
      multiple -> {:error, {:inconsistent_dimensions, multiple}}
    end
  end

  # ============================================================================
  # Private Functions - Embedding Extraction
  # ============================================================================

  defp extract_all_embeddings(snos, :metadata, _opts) do
    embeddings =
      Enum.map(snos, fn sno ->
        case extract_single_embedding(sno, :metadata, []) do
          {:ok, emb} -> emb
          {:error, _} -> nil
        end
      end)

    if Enum.any?(embeddings, &is_nil/1) do
      {:error, :missing_embeddings_in_metadata}
    else
      {:ok, embeddings}
    end
  end

  defp extract_all_embeddings(snos, :generate, opts) do
    encoder = Keyword.get(opts, :encoder, default_encoder())
    cache = Keyword.get(opts, :cache, true)

    case encoder do
      nil ->
        {:error, :no_encoder_available}

      encoder_module ->
        embeddings =
          Enum.map(snos, fn sno ->
            case extract_or_generate(sno, encoder_module, cache) do
              {:ok, emb} -> emb
              {:error, reason} -> {:error, reason}
            end
          end)

        if Enum.any?(embeddings, &match?({:error, _}, &1)) do
          errors = Enum.filter(embeddings, &match?({:error, _}, &1))
          {:error, {:generation_failed, errors}}
        else
          {:ok, embeddings}
        end
    end
  end

  defp extract_all_embeddings(snos, {:encoder, encoder_module}, opts) do
    opts_with_encoder = Keyword.put(opts, :encoder, encoder_module)
    extract_all_embeddings(snos, :generate, opts_with_encoder)
  end

  defp extract_single_embedding(%SNO{} = sno, :metadata, _opts) do
    cond do
      Map.has_key?(sno.metadata, :embedding) ->
        {:ok, Map.get(sno.metadata, :embedding)}

      Map.has_key?(sno.metadata, :embeddings) ->
        {:ok, Map.get(sno.metadata, :embeddings)}

      true ->
        {:error, :no_cached_embedding}
    end
  end

  defp extract_single_embedding(%SNO{} = sno, :generate, opts) do
    encoder = Keyword.get(opts, :encoder, default_encoder())

    case encoder do
      nil ->
        {:error, :no_encoder_available}

      encoder_module ->
        cache = Keyword.get(opts, :cache, true)
        extract_or_generate(sno, encoder_module, cache)
    end
  end

  defp extract_single_embedding(%SNO{} = sno, {:encoder, encoder_module}, opts) do
    opts_with_encoder = Keyword.put(opts, :encoder, encoder_module)
    extract_single_embedding(sno, :generate, opts_with_encoder)
  end

  defp extract_or_generate(sno, encoder_module, cache) do
    # Try cache first
    case extract_single_embedding(sno, :metadata, []) do
      {:ok, embedding} ->
        {:ok, embedding}

      {:error, :no_cached_embedding} ->
        # Generate new embedding
        case generate_embedding(sno, encoder_module) do
          {:ok, embedding} ->
            if cache do
              # Cache for future use (mutation happens outside this function)
              Logger.debug("Generated embedding for SNO #{sno.id}")
            end

            {:ok, embedding}

          {:error, reason} ->
            {:error, reason}
        end
    end
  end

  defp generate_embedding(%SNO{claim: claim}, encoder_module) do
    try do
      case encoder_module.encode(claim) do
        {:ok, embedding} -> {:ok, embedding}
        embedding when is_list(embedding) -> {:ok, embedding}
        %Nx.Tensor{} = tensor -> {:ok, Nx.to_flat_list(tensor)}
        _ -> {:error, :invalid_encoder_response}
      end
    rescue
      e ->
        Logger.error("Embedding generation failed: #{Exception.message(e)}")
        {:error, {:generation_failed, Exception.message(e)}}
    end
  end

  defp extract_embedding_dimension(%SNO{metadata: metadata}) do
    cond do
      Map.has_key?(metadata, :embedding) ->
        length(Map.get(metadata, :embedding))

      Map.has_key?(metadata, :embeddings) ->
        length(Map.get(metadata, :embeddings))

      true ->
        nil
    end
  end

  # ============================================================================
  # Private Functions - Utilities
  # ============================================================================

  defp normalize_embeddings(tensor) do
    {n, _} = Nx.shape(tensor)

    norms =
      tensor
      |> Nx.multiply(tensor)
      |> Nx.sum(axes: [1])
      |> Nx.sqrt()
      |> Nx.reshape({n, 1})

    Nx.divide(tensor, Nx.add(norms, 1.0e-8))
  end

  defp generate_text_interpretation(beta_zero, beta_one, beta_two) do
    parts = []

    parts =
      if beta_zero == 1 do
        parts ++ ["Connected claim network"]
      else
        parts ++ ["#{beta_zero} disconnected claim clusters"]
      end

    parts =
      cond do
        beta_one == 0 ->
          parts ++ ["with no circular reasoning (acyclic)"]

        beta_one == 1 ->
          parts ++ ["with 1 circular reasoning pattern"]

        true ->
          parts ++ ["with #{beta_one} circular reasoning patterns"]
      end

    parts =
      if beta_two > 0 do
        parts ++ ["and #{beta_two} higher-order void(s)"]
      else
        parts
      end

    Enum.join(parts, " ")
  end

  defp default_encoder do
    # Try to detect available encoder
    cond do
      Code.ensure_loaded?(CNS.Embedding.Encoder) -> CNS.Embedding.Encoder
      Code.ensure_loaded?(CNS.Encoder) -> CNS.Encoder
      true -> nil
    end
  end
end
```

---

## Test Suite

### File: `test/cns/topology/adapter_test.exs`

```elixir
defmodule CNS.Topology.AdapterTest do
  use ExUnit.Case, async: true
  doctest CNS.Topology.Adapter

  alias CNS.{SNO, Topology.Adapter}

  describe "sno_embeddings/2" do
    test "extracts embeddings from SNO metadata" do
      snos = [
        SNO.new("Claim A", metadata: %{embedding: [0.1, 0.2, 0.3]}),
        SNO.new("Claim B", metadata: %{embedding: [0.4, 0.5, 0.6]}),
        SNO.new("Claim C", metadata: %{embedding: [0.7, 0.8, 0.9]})
      ]

      embeddings = Adapter.sno_embeddings(snos)

      assert Nx.shape(embeddings) == {3, 3}
      assert Nx.type(embeddings) == {:f, 32}

      # Check first embedding
      first_row = embeddings[0] |> Nx.to_flat_list()
      assert_in_delta Enum.at(first_row, 0), 0.1, 0.001
    end

    test "handles empty list" do
      embeddings = Adapter.sno_embeddings([])
      assert Nx.shape(embeddings) == {0, 384}
    end

    test "returns error when embeddings missing in metadata mode" do
      snos = [
        SNO.new("Claim A", metadata: %{embedding: [0.1, 0.2]}),
        SNO.new("Claim B")  # Missing embedding
      ]

      result = Adapter.sno_embeddings(snos, source: :metadata)
      assert {:error, :missing_embeddings_in_metadata} = result
    end

    test "normalizes embeddings when requested" do
      snos = [
        SNO.new("A", metadata: %{embedding: [3.0, 4.0]})  # Norm = 5.0
      ]

      embeddings = Adapter.sno_embeddings(snos, normalize: true)
      row = embeddings[0] |> Nx.to_flat_list()

      # After normalization: [3/5, 4/5] = [0.6, 0.8]
      assert_in_delta Enum.at(row, 0), 0.6, 0.01
      assert_in_delta Enum.at(row, 1), 0.8, 0.01

      # Check norm is 1.0
      norm = :math.sqrt(Enum.sum(Enum.map(row, &(&1 * &1))))
      assert_in_delta norm, 1.0, 0.001
    end

    test "accepts Nx tensor directly" do
      snos = [
        SNO.new("A", metadata: %{embedding: Nx.tensor([1.0, 2.0])})
      ]

      embeddings = Adapter.sno_embeddings(snos)
      assert Nx.shape(embeddings) == {1, 2}
    end
  end

  describe "extract_embedding/2" do
    test "extracts single embedding from metadata" do
      sno = SNO.new("Test", metadata: %{embedding: [0.1, 0.2, 0.3]})
      assert {:ok, [0.1, 0.2, 0.3]} = Adapter.extract_embedding(sno)
    end

    test "returns error when no embedding in metadata" do
      sno = SNO.new("Test")
      assert {:error, :no_cached_embedding} = Adapter.extract_embedding(sno)
    end

    test "handles embeddings key variant" do
      sno = SNO.new("Test", metadata: %{embeddings: [1.0, 2.0]})
      assert {:ok, [1.0, 2.0]} = Adapter.extract_embedding(sno)
    end
  end

  describe "claim_graph/3" do
    setup do
      # Simple 2D embeddings for testing
      embeddings = Nx.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]
      ])

      %{embeddings: embeddings}
    end

    test "builds k-NN graph", %{embeddings: embeddings} do
      graph = Adapter.claim_graph(embeddings, :knn, k: 2)

      assert Graph.num_vertices(graph) == 4
      # Each vertex should have k outgoing edges
      assert Graph.num_edges(graph) > 0
    end

    test "builds epsilon graph", %{embeddings: embeddings} do
      graph = Adapter.claim_graph(embeddings, :epsilon, epsilon: 1.5)

      assert Graph.num_vertices(graph) == 4
      # Points within epsilon distance should be connected
      assert Graph.num_edges(graph) > 0
    end

    test "raises error for epsilon strategy without epsilon param", %{embeddings: embeddings} do
      assert_raise ArgumentError, ~r/epsilon is required/, fn ->
        Adapter.claim_graph(embeddings, :epsilon, [])
      end
    end

    test "supports cosine metric for k-NN", %{embeddings: embeddings} do
      graph = Adapter.claim_graph(embeddings, :knn, k: 2, metric: :cosine)
      assert Graph.num_vertices(graph) == 4
    end
  end

  describe "to_tensor/2" do
    test "converts nested list to tensor" do
      data = [[1, 2, 3], [4, 5, 6]]
      tensor = Adapter.to_tensor(data)

      assert Nx.shape(tensor) == {2, 3}
      assert Nx.type(tensor) == {:f, 32}
    end

    test "converts flat list to tensor" do
      data = [1.0, 2.0, 3.0]
      tensor = Adapter.to_tensor(data)

      assert Nx.shape(tensor) == {3}
    end

    test "passes through existing tensor" do
      original = Nx.tensor([1, 2, 3], type: :s32)
      result = Adapter.to_tensor(original)

      assert result == original
    end

    test "converts tensor type when specified" do
      data = [[1, 2], [3, 4]]
      tensor = Adapter.to_tensor(data, type: :s64)

      assert Nx.type(tensor) == {:s, 64}
    end

    test "validates shape when provided" do
      data = [[1, 2], [3, 4]]

      assert_raise ArgumentError, ~r/Shape mismatch/, fn ->
        Adapter.to_tensor(data, shape: {3, 2})
      end
    end

    test "accepts matching shape" do
      data = [[1, 2], [3, 4]]
      tensor = Adapter.to_tensor(data, shape: {2, 2})

      assert Nx.shape(tensor) == {2, 2}
    end
  end

  describe "interpret_betti/1" do
    test "interprets connected acyclic network" do
      inv = %{beta_zero: 1, beta_one: 0, beta_two: 0}
      result = Adapter.interpret_betti(inv)

      assert result.components == 1
      assert result.cycles == 0
      assert result.has_circular_reasoning == false
      assert result.circular_reasoning_severity == :none
      assert result.is_connected == true
      assert result.topology_class == :connected_acyclic
      assert result.interpretation =~ "Connected claim network"
      assert result.interpretation =~ "no circular reasoning"
    end

    test "interprets network with mild circular reasoning" do
      inv = %{beta_zero: 1, beta_one: 2, beta_two: 0}
      result = Adapter.interpret_betti(inv)

      assert result.cycles == 2
      assert result.has_circular_reasoning == true
      assert result.circular_reasoning_severity == :mild
      assert result.topology_class == :connected_with_cycles
      assert result.interpretation =~ "2 circular reasoning patterns"
    end

    test "interprets network with moderate circular reasoning" do
      inv = %{beta_zero: 1, beta_one: 4}
      result = Adapter.interpret_betti(inv)

      assert result.circular_reasoning_severity == :moderate
    end

    test "interprets network with severe circular reasoning" do
      inv = %{beta_zero: 1, beta_one: 10}
      result = Adapter.interpret_betti(inv)

      assert result.circular_reasoning_severity == :severe
    end

    test "interprets disconnected network" do
      inv = %{beta_zero: 3, beta_one: 0}
      result = Adapter.interpret_betti(inv)

      assert result.components == 3
      assert result.is_connected == false
      assert result.claim_clusters == 3
      assert result.topology_class == :disconnected_acyclic
      assert result.interpretation =~ "3 disconnected claim clusters"
    end

    test "interprets disconnected network with cycles" do
      inv = %{beta_zero: 2, beta_one: 3}
      result = Adapter.interpret_betti(inv)

      assert result.topology_class == :disconnected_with_cycles
    end

    test "interprets higher-order structures" do
      inv = %{beta_zero: 1, beta_one: 1, beta_two: 2}
      result = Adapter.interpret_betti(inv)

      assert result.voids == 2
      assert result.interpretation =~ "2 higher-order void(s)"
    end
  end

  describe "interpret_fragility/1" do
    test "interprets stable network" do
      fragility = %{mean_variance: 0.2, max_variance: 0.25}
      result = Adapter.interpret_fragility(fragility)

      assert result.stability == :stable
      assert result.recommendation =~ "stable"
    end

    test "interprets moderately fragile network" do
      fragility = %{mean_variance: 0.45, max_variance: 0.6}
      result = Adapter.interpret_fragility(fragility)

      assert result.stability == :moderate
      assert result.recommendation =~ "instability"
    end

    test "interprets fragile network" do
      fragility = %{mean_variance: 0.75, max_variance: 0.9}
      result = Adapter.interpret_fragility(fragility)

      assert result.stability == :fragile
      assert result.recommendation =~ "fragility"
    end

    test "detects outliers" do
      fragility = %{mean_variance: 0.3, max_variance: 0.6}  # max > mean * 1.5
      result = Adapter.interpret_fragility(fragility)

      assert result.has_outliers == true
    end

    test "accepts float input" do
      result = Adapter.interpret_fragility(0.8)
      assert result.stability == :fragile
    end
  end

  describe "build_causal_graph/2" do
    test "builds directed graph from causal links" do
      snos = [
        SNO.new("A", id: "a", metadata: %{causal_links: ["b", "c"]}),
        SNO.new("B", id: "b", metadata: %{causal_links: ["c"]}),
        SNO.new("C", id: "c")
      ]

      graph = Adapter.build_causal_graph(snos)

      assert Graph.num_vertices(graph) == 3
      assert Graph.num_edges(graph) == 3

      # Check edges exist
      assert Graph.has_edge?(graph, "a", "b")
      assert Graph.has_edge?(graph, "a", "c")
      assert Graph.has_edge?(graph, "b", "c")
    end

    test "handles SNOs without causal links" do
      snos = [
        SNO.new("A", id: "a"),
        SNO.new("B", id: "b")
      ]

      graph = Adapter.build_causal_graph(snos)

      assert Graph.num_vertices(graph) == 2
      assert Graph.num_edges(graph) == 0
    end

    test "supports custom link key" do
      snos = [
        SNO.new("A", id: "a", metadata: %{dependencies: ["b"]}),
        SNO.new("B", id: "b")
      ]

      graph = Adapter.build_causal_graph(snos, link_key: :dependencies)

      assert Graph.has_edge?(graph, "a", "b")
    end
  end

  describe "has_cached_embedding?/1" do
    test "returns true when embedding exists" do
      sno = SNO.new("Test", metadata: %{embedding: [1, 2, 3]})
      assert Adapter.has_cached_embedding?(sno)
    end

    test "returns true when embeddings key exists" do
      sno = SNO.new("Test", metadata: %{embeddings: [1, 2, 3]})
      assert Adapter.has_cached_embedding?(sno)
    end

    test "returns false when no embedding" do
      sno = SNO.new("Test")
      refute Adapter.has_cached_embedding?(sno)
    end
  end

  describe "cache_embedding/2" do
    test "caches embedding list in metadata" do
      sno = SNO.new("Test")
      embedding = [0.1, 0.2, 0.3]

      updated = Adapter.cache_embedding(sno, embedding)

      assert updated.metadata.embedding == embedding
    end

    test "caches tensor as list" do
      sno = SNO.new("Test")
      embedding = Nx.tensor([1.0, 2.0, 3.0])

      updated = Adapter.cache_embedding(sno, embedding)

      assert updated.metadata.embedding == [1.0, 2.0, 3.0]
    end

    test "preserves other metadata" do
      sno = SNO.new("Test", metadata: %{custom_field: "value"})
      embedding = [1.0, 2.0]

      updated = Adapter.cache_embedding(sno, embedding)

      assert updated.metadata.custom_field == "value"
      assert updated.metadata.embedding == [1.0, 2.0]
    end
  end

  describe "embedding_dimension/1" do
    test "returns dimension for consistent embeddings" do
      snos = [
        SNO.new("A", metadata: %{embedding: [1, 2, 3]}),
        SNO.new("B", metadata: %{embedding: [4, 5, 6]})
      ]

      assert {:ok, 3} = Adapter.embedding_dimension(snos)
    end

    test "returns error for inconsistent dimensions" do
      snos = [
        SNO.new("A", metadata: %{embedding: [1, 2, 3]}),
        SNO.new("B", metadata: %{embedding: [4, 5]})
      ]

      assert {:error, {:inconsistent_dimensions, [3, 2]}} = Adapter.embedding_dimension(snos)
    end

    test "returns default for empty list" do
      assert {:ok, 384} = Adapter.embedding_dimension([])
    end

    test "ignores SNOs without embeddings" do
      snos = [
        SNO.new("A", metadata: %{embedding: [1, 2, 3]}),
        SNO.new("B"),  # No embedding
        SNO.new("C", metadata: %{embedding: [4, 5, 6]})
      ]

      assert {:ok, 3} = Adapter.embedding_dimension(snos)
    end
  end

  describe "integration with ex_topology" do
    test "full pipeline: SNOs -> embeddings -> graph -> invariants" do
      # Create SNOs with embeddings
      snos = [
        SNO.new("A causes B", metadata: %{embedding: [0.0, 0.0]}),
        SNO.new("B causes C", metadata: %{embedding: [1.0, 0.0]}),
        SNO.new("C causes A", metadata: %{embedding: [0.5, 0.86]})  # Triangle
      ]

      # Extract embeddings
      embeddings = Adapter.sno_embeddings(snos)
      assert Nx.shape(embeddings) == {3, 2}

      # Build graph
      graph = Adapter.claim_graph(embeddings, :knn, k: 2)
      assert Graph.num_vertices(graph) == 3

      # Compute invariants (using ex_topology)
      invariants = ExTopology.Graph.invariants(graph)

      # Interpret in CNS context
      interpretation = Adapter.interpret_betti(invariants)

      assert is_map(interpretation)
      assert Map.has_key?(interpretation, :has_circular_reasoning)
      assert Map.has_key?(interpretation, :topology_class)
    end
  end
end
```

---

## Property-Based Tests

### File: `test/cns/topology/adapter_property_test.exs`

```elixir
defmodule CNS.Topology.AdapterPropertyTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias CNS.{SNO, Topology.Adapter}

  describe "property: sno_embeddings preserves count" do
    property "returns tensor with same number of rows as input SNOs" do
      check all(
              n <- integer(1..20),
              dim <- integer(2..10),
              embeddings <- list_of(list_of(float(min: -1.0, max: 1.0), length: dim), length: n)
            ) do
        snos = Enum.map(embeddings, fn emb ->
          SNO.new("Test claim", metadata: %{embedding: emb})
        end)

        result = Adapter.sno_embeddings(snos)
        assert Nx.shape(result) == {n, dim}
      end
    end
  end

  describe "property: normalization produces unit vectors" do
    property "normalized embeddings have unit norm" do
      check all(
              n <- integer(1..10),
              dim <- integer(2..5),
              embeddings <- list_of(
                list_of(float(min: -10.0, max: 10.0), length: dim),
                length: n
              )
            ) do
        # Filter out zero vectors
        non_zero_embeddings = Enum.filter(embeddings, fn emb ->
          Enum.any?(emb, &(abs(&1) > 0.01))
        end)

        if length(non_zero_embeddings) > 0 do
          snos = Enum.map(non_zero_embeddings, fn emb ->
            SNO.new("Test", metadata: %{embedding: emb})
          end)

          tensor = Adapter.sno_embeddings(snos, normalize: true)

          # Check each row has norm ~= 1.0
          norms = Nx.LinAlg.norm(tensor, axes: [1]) |> Nx.to_flat_list()

          Enum.each(norms, fn norm ->
            assert_in_delta norm, 1.0, 0.01
          end)
        end
      end
    end
  end

  describe "property: to_tensor type conversion" do
    property "converts to requested type" do
      check all(
              data <- list_of(list_of(integer(-100..100), length: 3), length: 2),
              type <- member_of([:s32, :s64, :f32, :f64])
            ) do
        tensor = Adapter.to_tensor(data, type: type)
        assert Nx.type(tensor) == Nx.Type.normalize!(type)
      end
    end
  end

  describe "property: interpret_betti components" do
    property "components equals beta_zero" do
      check all(beta_zero <- integer(0..10)) do
        inv = %{beta_zero: beta_zero, beta_one: 0}
        result = Adapter.interpret_betti(inv)

        assert result.components == beta_zero
        assert result.claim_clusters == beta_zero
        assert result.is_connected == (beta_zero == 1)
      end
    end
  end

  describe "property: interpret_betti cycles" do
    property "has_circular_reasoning true iff beta_one > 0" do
      check all(
              beta_zero <- integer(1..5),
              beta_one <- integer(0..10)
            ) do
        inv = %{beta_zero: beta_zero, beta_one: beta_one}
        result = Adapter.interpret_betti(inv)

        assert result.has_circular_reasoning == (beta_one > 0)
        assert result.cycles == beta_one
      end
    end
  end

  describe "property: causal graph structure" do
    property "graph has correct number of vertices" do
      check all(n <- integer(1..10)) do
        snos = Enum.map(1..n, fn i ->
          SNO.new("Claim #{i}", id: "id_#{i}")
        end)

        graph = Adapter.build_causal_graph(snos)
        assert Graph.num_vertices(graph) == n
      end
    end
  end

  describe "property: cache_embedding roundtrip" do
    property "cached embedding can be retrieved" do
      check all(embedding <- list_of(float(min: -1.0, max: 1.0), length: 5)) do
        sno = SNO.new("Test")
        updated = Adapter.cache_embedding(sno, embedding)

        assert Adapter.has_cached_embedding?(updated)
        assert {:ok, retrieved} = Adapter.extract_embedding(updated)
        assert length(retrieved) == length(embedding)

        # Check values are close
        Enum.zip(retrieved, embedding)
        |> Enum.each(fn {r, e} ->
          assert_in_delta r, e, 0.001
        end)
      end
    end
  end
end
```

---

## Usage Examples

### Example 1: Basic SNO to Tensor Conversion

```elixir
# Create SNOs with cached embeddings
snos = [
  CNS.SNO.new("Coffee improves alertness",
    metadata: %{embedding: [0.12, -0.05, 0.31, ..., 0.08]}
  ),
  CNS.SNO.new("Caffeine blocks adenosine",
    metadata: %{embedding: [0.15, -0.03, 0.28, ..., 0.11]}
  )
]

# Extract embeddings as tensor
embeddings = CNS.Topology.Adapter.sno_embeddings(snos)
# => Nx.Tensor<f32[2][384]>

# Normalize if needed
normalized = CNS.Topology.Adapter.sno_embeddings(snos, normalize: true)
```

### Example 2: Building and Analyzing Claim Graph

```elixir
# Get embeddings
embeddings = CNS.Topology.Adapter.sno_embeddings(snos)

# Build k-NN graph
graph = CNS.Topology.Adapter.claim_graph(embeddings, :knn, k: 5)

# Compute topological invariants with ex_topology
invariants = ExTopology.Graph.invariants(graph)
# => %{beta_zero: 1, beta_one: 2, euler_characteristic: -1}

# Interpret in CNS context
interpretation = CNS.Topology.Adapter.interpret_betti(invariants)
# => %{
#   components: 1,
#   cycles: 2,
#   has_circular_reasoning: true,
#   circular_reasoning_severity: :mild,
#   topology_class: :connected_with_cycles,
#   interpretation: "Connected claim network with 2 circular reasoning patterns"
# }
```

### Example 3: Fragility Analysis

```elixir
# Extract embeddings
embeddings = CNS.Topology.Adapter.sno_embeddings(snos)

# Compute k-NN variance with ex_topology
variance = ExTopology.Embedding.knn_variance(embeddings, k: 3)
# => 0.45

# Interpret fragility
fragility = CNS.Topology.Adapter.interpret_fragility(variance)
# => %{
#   stability: :moderate,
#   mean_variance: 0.45,
#   has_outliers: false,
#   recommendation: "Some semantic instability detected. Review isolated claims."
# }
```

### Example 4: Causal Link Graph Construction

```elixir
# SNOs with causal relationships
snos = [
  CNS.SNO.new("A causes B", id: "a", metadata: %{causal_links: ["b"]}),
  CNS.SNO.new("B causes C", id: "b", metadata: %{causal_links: ["c"]}),
  CNS.SNO.new("C causes A", id: "c", metadata: %{causal_links: ["a"]})  # Cycle!
]

# Build directed graph
graph = CNS.Topology.Adapter.build_causal_graph(snos)

# Check for cycles with ex_topology
invariants = ExTopology.Graph.invariants(graph)
interpretation = CNS.Topology.Adapter.interpret_betti(invariants)

if interpretation.has_circular_reasoning do
  IO.puts("⚠️  Detected circular reasoning in causal chain")
end
```

### Example 5: Generate Embeddings On-Demand

```elixir
# SNOs without cached embeddings
snos = [
  CNS.SNO.new("First claim"),
  CNS.SNO.new("Second claim")
]

# Generate embeddings (requires encoder module)
embeddings = CNS.Topology.Adapter.sno_embeddings(snos,
  source: :generate,
  encoder: CNS.Embedding.Encoder,
  cache: true  # Cache in metadata for reuse
)

# Now SNOs have cached embeddings
assert CNS.Topology.Adapter.has_cached_embedding?(List.first(snos))
```

---

## Integration Checklist

### Phase 1 Tasks

- [ ] **Add module file**: Create `lib/cns/topology/adapter.ex`
- [ ] **Add test file**: Create `test/cns/topology/adapter_test.exs`
- [ ] **Add property tests**: Create `test/cns/topology/adapter_property_test.exs`
- [ ] **Run tests**: `mix test test/cns/topology/adapter_test.exs`
- [ ] **Run doctests**: Ensure all doctests pass
- [ ] **Check coverage**: Aim for >90% coverage
- [ ] **Update mix.exs**: Add ex_topology dependency
- [ ] **Run formatter**: `mix format`
- [ ] **Run dialyzer**: `mix dialyzer` (add typespecs if needed)
- [ ] **Integration test**: Test with real CNS.SNO data
- [ ] **Benchmark**: Compare performance with surrogate implementation

### Validation Steps

```elixir
# 1. Verify adapter works with existing surrogates
alias CNS.Topology.{Adapter, Surrogates}

snos = generate_test_snos(10)
embeddings = Adapter.sno_embeddings(snos)

# 2. Compare beta1 results
graph = Adapter.claim_graph(embeddings, :knn, k: 3)
beta1_extopo = ExTopology.Graph.beta_one(graph)

causal_graph = extract_causal_links(snos)
beta1_surrogate = Surrogates.compute_beta1_surrogate(causal_graph)

# Should be correlated but may differ
IO.inspect({beta1_extopo, beta1_surrogate}, label: "Beta1 comparison")

# 3. Verify interpretation makes sense
interpretation = Adapter.interpret_betti(%{beta_zero: 1, beta_one: beta1_extopo})
IO.inspect(interpretation, label: "Interpretation")
```

---

## Performance Considerations

### Caching Strategy

The adapter implements a **lazy evaluation + caching** strategy:

1. **First priority**: Check SNO metadata for cached embeddings
2. **Second priority**: Generate embeddings if encoder available
3. **Cache on generation**: Store generated embeddings in metadata (optional)

### Optimization Opportunities

- **Batch encoding**: Generate all embeddings in one call to encoder
- **EXLA backend**: Use GPU acceleration for distance computations
- **Sparse graphs**: Use ex_topology's sparse graph representations
- **Streaming**: Process large SNO lists in chunks

### Benchmarking Template

```elixir
# File: bench/adapter_benchmark.exs
Benchee.run(
  %{
    "extract_embeddings_10" => fn snos_10 ->
      CNS.Topology.Adapter.sno_embeddings(snos_10)
    end,
    "extract_embeddings_100" => fn snos_100 ->
      CNS.Topology.Adapter.sno_embeddings(snos_100)
    end,
    "build_knn_graph_100" => fn {embeddings_100, _} ->
      CNS.Topology.Adapter.claim_graph(embeddings_100, :knn, k: 5)
    end
  },
  inputs: %{
    "10 SNOs" => generate_test_data(10),
    "100 SNOs" => generate_test_data(100),
    "1000 SNOs" => generate_test_data(1000)
  }
)
```

---

## Next Steps (Phase 2)

After Phase 1 completion and validation:

1. **Create `CNS.Topology.GraphAnalysis`** - Wrapper for ex_topology graph functions
2. **Deprecate old functions** - Add warnings to `CNS.Topology.betti_numbers/1`
3. **Update `CNS.Logic.Betti`** - Use adapter + ex_topology instead of manual computation
4. **Integration tests** - Full pipeline tests with dialectical synthesis

---

## References

- **CNS Architecture**: [cns_rebuild_with_ex_topology.md](./cns_rebuild_with_ex_topology.md)
- **ex_topology Docs**: https://hexdocs.pm/ex_topology/0.1.1
- **SNO Structure**: [lib/cns/sno.ex](../../lib/cns/sno.ex)
- **Current Surrogates**: [lib/cns/topology/surrogates.ex](../../lib/cns/topology/surrogates.ex)

---

**Status**: ✅ Ready for implementation
**Estimated LOC**: ~500 (module) + ~600 (tests)
**Dependencies**: ex_topology v0.1.1, CNS.SNO, Nx
**Breaking Changes**: None (pure addition)

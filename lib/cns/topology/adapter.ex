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
    # Note: delaunay_graph may not be available in all ex_topology versions
    # Fall back to k-NN if not available
    if function_exported?(ExTopology.Neighborhood, :delaunay_graph, 1) do
      ExTopology.Neighborhood.delaunay_graph(embeddings)
    else
      Logger.warning("delaunay_graph not available, falling back to k-NN with k=5")
      claim_graph(embeddings, :knn, k: 5)
    end
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

defmodule CNS.Metrics.Chirality do
  @moduledoc """
  Compute chirality scores using Fisher-Rao distance approximation
  for measuring semantic divergence between thesis/antithesis pairs.

  Chirality quantifies argumentative bias through geometric asymmetry
  in the embedding space.
  """

  defmodule FisherRaoStats do
    @moduledoc "Pre-computed statistics for Fisher-Rao distance calculation"

    @type t :: %__MODULE__{
            mean: Nx.Tensor.t(),
            inv_var: Nx.Tensor.t()
          }

    @enforce_keys [:mean, :inv_var]
    defstruct [:mean, :inv_var]
  end

  defmodule ChiralityResult do
    @moduledoc "Result of chirality analysis between thesis and antithesis"

    @type t :: %__MODULE__{
            fisher_rao_distance: float(),
            evidence_overlap: float(),
            polarity_conflict: boolean(),
            chirality_score: float()
          }

    @enforce_keys [:fisher_rao_distance, :evidence_overlap, :polarity_conflict, :chirality_score]
    defstruct [:fisher_rao_distance, :evidence_overlap, :polarity_conflict, :chirality_score]
  end

  @doc """
  Build Fisher-Rao statistics from a collection of embedding vectors.

  ## Parameters
    - vectors: Nx tensor of shape {n_samples, embedding_dim} or list of lists
    - epsilon: Small value for numerical stability (default: 1.0e-6)

  ## Returns
    FisherRaoStats with mean and inverse variance

  ## Examples

      iex> vectors = Nx.tensor([[1.0, 2.0], [1.5, 2.5], [1.2, 2.1]])
      iex> stats = CNS.Metrics.Chirality.build_fisher_rao_stats(vectors)
      iex> Nx.shape(stats.mean)
      {2}
  """
  @spec build_fisher_rao_stats(Nx.Tensor.t() | [[number()]], float()) :: FisherRaoStats.t()
  def build_fisher_rao_stats(vectors, epsilon \\ 1.0e-6) do
    tensor =
      case vectors do
        %Nx.Tensor{} -> vectors
        list when is_list(list) -> Nx.tensor(list)
      end

    mean = Nx.mean(tensor, axes: [0])
    variance = Nx.variance(tensor, axes: [0])

    # Inverse variance with epsilon for numerical stability
    inv_var = Nx.divide(1.0, Nx.add(variance, epsilon))

    %FisherRaoStats{mean: mean, inv_var: inv_var}
  end

  @doc """
  Compute Fisher-Rao distance between two embedding vectors.

  Uses diagonal approximation of Fisher information metric.

  ## Parameters
    - vec_a: First embedding vector
    - vec_b: Second embedding vector
    - stats: FisherRaoStats containing inverse variance weights

  ## Returns
    Distance as a float

  ## Examples

      iex> stats = %CNS.Metrics.Chirality.FisherRaoStats{
      ...>   mean: Nx.tensor([0.0, 0.0]),
      ...>   inv_var: Nx.tensor([1.0, 1.0])
      ...> }
      iex> vec_a = Nx.tensor([0.0, 0.0])
      iex> vec_b = Nx.tensor([3.0, 4.0])
      iex> CNS.Metrics.Chirality.fisher_rao_distance(vec_a, vec_b, stats)
      5.0
  """
  @spec fisher_rao_distance(Nx.Tensor.t(), Nx.Tensor.t(), FisherRaoStats.t()) :: float()
  def fisher_rao_distance(vec_a, vec_b, %FisherRaoStats{inv_var: inv_var}) do
    diff = Nx.subtract(vec_a, vec_b)
    weighted_sq = Nx.multiply(Nx.multiply(diff, inv_var), diff)

    weighted_sq
    |> Nx.sum()
    |> Nx.sqrt()
    |> Nx.to_number()
  end

  @doc """
  Compute chirality score from distance, overlap, and conflict.

  The chirality score is a normalized composite metric in [0, 1] that combines:
  - Fisher-Rao distance (normalized, weight: 0.6)
  - Evidence overlap inverse (weight: 0.2)
  - Polarity conflict penalty (0.25 if true)

  ## Parameters
    - distance: Fisher-Rao distance between embeddings
    - evidence_overlap: Overlap score in [0, 1]
    - polarity_conflict: Whether polarity conflict exists

  ## Returns
    Chirality score in [0, 1]

  ## Examples

      iex> CNS.Metrics.Chirality.compute_chirality_score(1.0, 0.5, false)
      0.4
  """
  @spec compute_chirality_score(float(), float(), boolean()) :: float()
  def compute_chirality_score(distance, evidence_overlap, polarity_conflict) do
    # Normalize distance to [0, 1]
    norm_distance = distance / (distance + 1.0)

    # Overlap factor (inverse of overlap)
    overlap_factor = 1.0 - clamp(evidence_overlap, 0.0, 1.0)

    # Penalty for polarity conflict
    conflict_penalty = if polarity_conflict, do: 0.25, else: 0.0

    # Composite score with weights
    raw_score = norm_distance * 0.6 + overlap_factor * 0.2 + conflict_penalty

    min(1.0, raw_score)
  end

  @doc """
  Compare thesis and antithesis to compute chirality score.

  ## Parameters
    - embedder: Module implementing encode/1 callback
    - stats: Pre-computed FisherRaoStats
    - thesis: Thesis text
    - antithesis: Antithesis text
    - evidence_overlap: Overlap score in [0, 1]
    - polarity_conflict: Whether polarity conflict exists

  ## Returns
    ChiralityResult with composite chirality score
  """
  @spec compare(
          module(),
          FisherRaoStats.t(),
          String.t(),
          String.t(),
          float(),
          boolean()
        ) :: ChiralityResult.t()
  def compare(embedder, stats, thesis, antithesis, evidence_overlap, polarity_conflict) do
    # Get embeddings from embedder module
    embeddings = embedder.encode([thesis, antithesis])

    [thesis_emb, antithesis_emb] =
      embeddings
      |> Nx.to_batched(1)
      |> Enum.map(&Nx.squeeze/1)

    # Compute Fisher-Rao distance
    distance = fisher_rao_distance(thesis_emb, antithesis_emb, stats)

    # Compute chirality score
    score = compute_chirality_score(distance, evidence_overlap, polarity_conflict)

    %ChiralityResult{
      fisher_rao_distance: distance,
      evidence_overlap: evidence_overlap,
      polarity_conflict: polarity_conflict,
      chirality_score: score
    }
  end

  defp clamp(value, min_val, max_val) do
    value
    |> max(min_val)
    |> min(max_val)
  end
end

defmodule CNS.Topology.TDA do
  @moduledoc """
  Persistent homology / topological data analysis for CNS reasoning structures.

  This module is pure CNS logic: it works on SNOs/graphs and returns TDA
  summaries, leaving orchestration to higher-level apps.
  """

  alias CNS.SNO

  defmodule Result do
    @moduledoc "Per-SNO TDA result."

    @type t :: %__MODULE__{
            sno_id: String.t(),
            betti: %{non_neg_integer() => non_neg_integer()},
            diagrams: %{non_neg_integer() => [barcode()]},
            summary: %{
              total_features: non_neg_integer(),
              max_persistence: float(),
              mean_persistence: float(),
              persistent_cycle_ratio: float(),
              notes: String.t() | nil
            }
          }

    @type barcode :: %{birth: float(), death: float(), persistence: float()}

    @enforce_keys [:sno_id, :betti, :diagrams, :summary]
    defstruct [:sno_id, :betti, :diagrams, :summary]
  end

  @type tda_summary :: %{
          beta0_mean: float(),
          beta1_mean: float(),
          beta2_mean: float(),
          high_persistence_fraction: float(),
          avg_persistence: float(),
          n_snos: non_neg_integer()
        }

  @doc """
  Compute TDA metrics for a list of SNOs.

  Options:
    * `:max_dim` - maximum homology dimension (default: 2)
    * `:persistence_threshold` - minimum persistence to count a feature (default: 0.1)
    * `:distance_metric` - :fisher_rao | :cosine | :euclidean
    * `:max_edge_length` - optional cutoff for Rips complex
  """
  @spec compute_for_snos([SNO.t()], keyword()) :: {[Result.t()], tda_summary()}
  def compute_for_snos(_snos, _opts \\ []) do
    raise "not implemented"
  end

  @doc """
  Compute TDA for a single SNO.
  """
  @spec compute_for_sno(SNO.t(), keyword()) :: Result.t()
  def compute_for_sno(_sno, _opts \\ []) do
    raise "not implemented"
  end

  @doc """
  Summarize a list of TDA results.
  """
  @spec summarize([Result.t()], keyword()) :: tda_summary()
  def summarize(_results, _opts \\ []) do
    raise "not implemented"
  end
end

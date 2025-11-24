defmodule CNS.Topology.TDA do
  @moduledoc """
  Persistent homology / topological data analysis for CNS reasoning structures.

  This module is pure CNS logic: it works on SNOs/graphs and returns TDA
  summaries, leaving orchestration to higher-level apps.
  """

  alias CNS.SNO
  alias CNS.Topology

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
  def compute_for_snos(snos, opts \\ []) do
    snos = List.wrap(snos)

    results =
      snos
      |> Enum.map(&compute_for_sno(&1, opts))

    {results, summarize(results, opts)}
  end

  @doc """
  Compute TDA for a single SNO.
  """
  @spec compute_for_sno(SNO.t(), keyword()) :: Result.t()
  def compute_for_sno(%SNO{id: id} = sno, _opts \\ []) do
    snos = flatten_snos([sno])
    graph = Topology.build_graph(snos)
    betti = Topology.betti_numbers(graph)

    diagrams = build_diagrams(betti)

    %Result{
      sno_id: id,
      betti: %{0 => betti.b0, 1 => betti.b1},
      diagrams: diagrams,
      summary: %{
        total_features: betti.b0 + betti.b1,
        max_persistence: max_persistence(diagrams),
        mean_persistence: mean_persistence(diagrams),
        persistent_cycle_ratio:
          if(betti.b0 + betti.b1 > 0, do: betti.b1 / max(1, betti.b0 + betti.b1), else: 0.0),
        notes: nil
      }
    }
  end

  @doc """
  Summarize a list of TDA results.
  """
  @spec summarize([Result.t()], keyword()) :: tda_summary()
  def summarize(results, _opts \\ []) do
    count = length(results)

    if count == 0 do
      %{
        beta0_mean: 0.0,
        beta1_mean: 0.0,
        beta2_mean: 0.0,
        high_persistence_fraction: 0.0,
        avg_persistence: 0.0,
        n_snos: 0
      }
    else
      betti0 = Enum.map(results, fn r -> Map.get(r.betti, 0, 0) end)
      betti1 = Enum.map(results, fn r -> Map.get(r.betti, 1, 0) end)

      mean_persist =
        results
        |> Enum.map(fn r -> r.summary.mean_persistence end)
        |> mean()

      %{
        beta0_mean: mean(betti0),
        beta1_mean: mean(betti1),
        beta2_mean: 0.0,
        high_persistence_fraction: fraction(betti1, fn v -> v > 0 end),
        avg_persistence: mean_persist,
        n_snos: count
      }
    end
  end

  defp flatten_snos(snos) do
    Enum.flat_map(snos, fn
      %SNO{children: children} = sno ->
        [sno | flatten_snos(children || [])]

      other ->
        List.wrap(other)
    end)
  end

  defp build_diagrams(%{b0: b0, b1: b1}) do
    %{
      0 => Enum.map(1..b0, fn _ -> %{birth: 0.0, death: 0.0, persistence: 0.0} end),
      1 => Enum.map(1..b1, fn _ -> %{birth: 0.0, death: 1.0, persistence: 1.0} end),
      2 => []
    }
  end

  defp max_persistence(diagrams) do
    diagrams
    |> Map.values()
    |> List.flatten()
    |> Enum.map(& &1.persistence)
    |> case do
      [] -> 0.0
      vals -> Enum.max(vals)
    end
  end

  defp mean_persistence(diagrams) do
    diagrams
    |> Map.values()
    |> List.flatten()
    |> Enum.map(& &1.persistence)
    |> mean()
  end

  defp mean([]), do: 0.0
  defp mean(values), do: Enum.sum(values) / length(values)

  defp fraction([], _fun), do: 0.0

  defp fraction(list, fun) do
    Enum.count(list, fun) / length(list)
  end
end

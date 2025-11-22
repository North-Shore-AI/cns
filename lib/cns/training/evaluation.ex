defmodule CNS.Training.Evaluation do
  @moduledoc """
  Evaluation harness for CNS claim extraction models.

  Computes metrics on predictions vs gold standard data
  including precision, recall, F1, and accuracy.
  """

  alias CNS.Schema.Parser

  defmodule Metrics do
    @moduledoc "Evaluation metrics for model performance"

    @type t :: %__MODULE__{
            precision: float(),
            recall: float(),
            f1: float(),
            accuracy: float()
          }

    @enforce_keys [:precision, :recall, :f1, :accuracy]
    defstruct [:precision, :recall, :f1, :accuracy]
  end

  defmodule EvalConfig do
    @moduledoc "Configuration for evaluation runs"

    @type t :: %__MODULE__{
            batch_size: pos_integer(),
            max_samples: pos_integer() | nil
          }

    defstruct batch_size: 32,
              max_samples: nil
  end

  defmodule EvalResult do
    @moduledoc "Result of an evaluation run"

    @type t :: %__MODULE__{
            metrics: Metrics.t(),
            num_samples: non_neg_integer(),
            errors: [String.t()]
          }

    @enforce_keys [:metrics, :num_samples, :errors]
    defstruct [:metrics, :num_samples, :errors]
  end

  @doc """
  Compute basic metrics from predictions and gold labels.

  ## Examples

      iex> CNS.Training.Evaluation.compute_metrics(["A", "B"], ["A", "B"])
      %CNS.Training.Evaluation.Metrics{precision: 1.0, recall: 1.0, f1: 1.0, accuracy: 1.0}
  """
  @spec compute_metrics([String.t()], [String.t()]) :: Metrics.t()
  def compute_metrics(predictions, gold) do
    if length(predictions) == 0 or length(gold) == 0 do
      %Metrics{precision: 0.0, recall: 0.0, f1: 0.0, accuracy: 0.0}
    else
      correct =
        Enum.zip(predictions, gold)
        |> Enum.count(fn {pred, g} -> pred == g end)

      total = length(gold)
      accuracy = correct / total

      # For simple accuracy-based metrics
      precision = accuracy
      recall = accuracy
      f1 = compute_f1(precision, recall)

      %Metrics{
        precision: precision,
        recall: recall,
        f1: f1,
        accuracy: accuracy
      }
    end
  end

  @doc """
  Evaluate claim extraction predictions against gold standard.

  ## Parameters
    - predictions: List of model output strings
    - gold: List of gold standard strings
    - config: Evaluation configuration

  ## Returns
    EvalResult with metrics and metadata
  """
  @spec evaluate_claims([String.t()], [String.t()], EvalConfig.t()) :: EvalResult.t()
  def evaluate_claims(predictions, gold, %EvalConfig{} = _config) do
    metrics = compute_metrics(predictions, gold)

    %EvalResult{
      metrics: metrics,
      num_samples: length(predictions),
      errors: []
    }
  end

  @doc """
  Compute F1 score from precision and recall.

  ## Examples

      iex> CNS.Training.Evaluation.compute_f1(0.8, 0.6)
      0.6857142857142857
  """
  @spec compute_f1(float(), float()) :: float()
  def compute_f1(precision, recall) do
    if precision + recall == 0 do
      0.0
    else
      2 * precision * recall / (precision + recall)
    end
  end

  @doc """
  Extract claim texts from formatted output.

  ## Examples

      iex> CNS.Training.Evaluation.extract_claims_from_output("CLAIM[c1]: Test")
      ["Test"]
  """
  @spec extract_claims_from_output(String.t()) :: [String.t()]
  def extract_claims_from_output(output) do
    output
    |> Parser.parse_claims()
    |> Map.values()
    |> Enum.map(& &1.text)
  end

  @doc """
  Extract relations from formatted output.

  ## Examples

      iex> CNS.Training.Evaluation.extract_relations_from_output("RELATION: c2 supports c1")
      [{"c2", "supports", "c1"}]
  """
  @spec extract_relations_from_output(String.t()) :: [{String.t(), String.t(), String.t()}]
  def extract_relations_from_output(output) do
    Parser.parse_relations(output)
  end

  @doc """
  Evaluate claim and relation extraction with detailed metrics.

  Returns separate metrics for claims and relations.
  """
  @spec evaluate_detailed([String.t()], [String.t()]) :: map()
  def evaluate_detailed(predictions, gold) do
    claim_results =
      Enum.zip(predictions, gold)
      |> Enum.map(fn {pred, g} ->
        pred_claims = extract_claims_from_output(pred) |> MapSet.new()
        gold_claims = extract_claims_from_output(g) |> MapSet.new()
        compute_set_metrics(pred_claims, gold_claims)
      end)

    relation_results =
      Enum.zip(predictions, gold)
      |> Enum.map(fn {pred, g} ->
        pred_rels = extract_relations_from_output(pred) |> MapSet.new()
        gold_rels = extract_relations_from_output(g) |> MapSet.new()
        compute_set_metrics(pred_rels, gold_rels)
      end)

    %{
      claim_metrics: aggregate_metrics(claim_results),
      relation_metrics: aggregate_metrics(relation_results)
    }
  end

  defp compute_set_metrics(predicted, gold) do
    if MapSet.size(gold) == 0 do
      %{precision: 0.0, recall: 0.0, f1: 0.0}
    else
      tp = MapSet.intersection(predicted, gold) |> MapSet.size()
      fp = MapSet.difference(predicted, gold) |> MapSet.size()
      fn_count = MapSet.difference(gold, predicted) |> MapSet.size()

      precision = if tp + fp == 0, do: 0.0, else: tp / (tp + fp)
      recall = if tp + fn_count == 0, do: 0.0, else: tp / (tp + fn_count)
      f1 = compute_f1(precision, recall)

      %{precision: precision, recall: recall, f1: f1}
    end
  end

  defp aggregate_metrics(results) do
    if Enum.empty?(results) do
      %{precision: 0.0, recall: 0.0, f1: 0.0}
    else
      n = length(results)

      avg_precision = Enum.map(results, & &1.precision) |> Enum.sum() |> Kernel./(n)
      avg_recall = Enum.map(results, & &1.recall) |> Enum.sum() |> Kernel./(n)
      avg_f1 = Enum.map(results, & &1.f1) |> Enum.sum() |> Kernel./(n)

      %{precision: avg_precision, recall: avg_recall, f1: avg_f1}
    end
  end
end

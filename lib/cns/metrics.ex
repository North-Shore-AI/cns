defmodule CNS.Metrics do
  @moduledoc """
  Quality metrics for CNS pipeline evaluation.

  Provides metrics for measuring synthesis quality including:
  - Entailment scoring
  - Citation accuracy
  - Pass rate calculation
  - Chirality computation
  - Fisher-Rao distance

  ## CNS 3.0 Quality Targets

  - Schema compliance >= 95%
  - Citation accuracy >= 95%
  - Mean entailment >= 0.50
  """

  alias CNS.{SNO, Evidence, Challenge}

  @doc """
  Calculate overall quality score for a synthesis result.

  ## Examples

      iex> synthesis = CNS.SNO.new("Result", confidence: 0.9)
      iex> metrics = CNS.Metrics.quality_score(synthesis)
      iex> metrics.overall > 0
      true
  """
  @spec quality_score(SNO.t()) :: map()
  def quality_score(%SNO{} = sno) do
    evidence_quality = evidence_quality_score(sno)
    confidence = sno.confidence
    coverage = evidence_coverage(sno)

    overall = confidence * 0.4 + evidence_quality * 0.4 + coverage * 0.2

    %{
      overall: Float.round(overall, 4),
      confidence: confidence,
      evidence_quality: evidence_quality,
      coverage: coverage
    }
  end

  @doc """
  Calculate entailment score between premise and conclusion.

  Measures how well the conclusion follows from the premise.

  ## Examples

      iex> premise = CNS.SNO.new("Coffee contains caffeine")
      iex> conclusion = CNS.SNO.new("Coffee has stimulant effects from caffeine")
      iex> score = CNS.Metrics.entailment(premise, conclusion)
      iex> score >= 0.0 and score <= 1.0
      true
  """
  @spec entailment(SNO.t(), SNO.t()) :: float()
  def entailment(%SNO{} = premise, %SNO{} = conclusion) do
    # Extract key terms
    premise_terms = extract_terms(premise.claim)
    conclusion_terms = extract_terms(conclusion.claim)

    # Calculate overlap
    overlap = term_overlap(premise_terms, conclusion_terms)

    # Weight by confidence
    weighted = overlap * (premise.confidence + conclusion.confidence) / 2

    Float.round(weighted, 4)
  end

  @doc """
  Calculate citation accuracy for a set of SNOs.

  ## Examples

      iex> e1 = CNS.Evidence.new("Valid", "", validity: 0.9)
      iex> e2 = CNS.Evidence.new("Invalid", "", validity: 0.3)
      iex> sno = CNS.SNO.new("Claim", evidence: [e1, e2])
      iex> accuracy = CNS.Metrics.citation_accuracy([sno])
      iex> accuracy > 0
      true
  """
  @spec citation_accuracy([SNO.t()]) :: float()
  def citation_accuracy(snos) when is_list(snos) do
    all_evidence = Enum.flat_map(snos, & &1.evidence)

    if Enum.empty?(all_evidence) do
      0.0
    else
      valid_count = Enum.count(all_evidence, &(&1.validity >= 0.7))
      Float.round(valid_count / length(all_evidence), 4)
    end
  end

  @doc """
  Calculate pass rate based on quality threshold.

  ## Examples

      iex> s1 = CNS.SNO.new("A", confidence: 0.9)
      iex> s2 = CNS.SNO.new("B", confidence: 0.4)
      iex> rate = CNS.Metrics.pass_rate([s1, s2], 0.5)
      iex> rate
      0.5
  """
  @spec pass_rate([SNO.t()], float()) :: float()
  def pass_rate(snos, threshold) when is_list(snos) do
    if Enum.empty?(snos) do
      0.0
    else
      passed = Enum.count(snos, &(&1.confidence >= threshold))
      Float.round(passed / length(snos), 4)
    end
  end

  @doc """
  Calculate chirality score for challenges.

  Chirality measures the degree of contradiction/tension requiring resolution.

  ## Examples

      iex> c1 = CNS.Challenge.new("id", :contradiction, "Test", severity: :high, confidence: 0.8)
      iex> c2 = CNS.Challenge.new("id", :scope, "Test", severity: :low, confidence: 0.5)
      iex> score = CNS.Metrics.chirality([c1, c2])
      iex> score > 0
      true
  """
  @spec chirality([Challenge.t()]) :: float()
  def chirality(challenges) when is_list(challenges) do
    if Enum.empty?(challenges) do
      0.0
    else
      scores = Enum.map(challenges, &Challenge.chirality_score/1)
      Float.round(Enum.sum(scores) / length(scores), 4)
    end
  end

  @doc """
  Calculate Fisher-Rao distance between two distributions.

  Used for measuring divergence between claim confidence distributions.

  ## Examples

      iex> dist1 = [0.2, 0.5, 0.3]
      iex> dist2 = [0.3, 0.4, 0.3]
      iex> distance = CNS.Metrics.fisher_rao_distance(dist1, dist2)
      iex> distance >= 0.0
      true
  """
  @spec fisher_rao_distance([float()], [float()]) :: float()
  def fisher_rao_distance(dist1, dist2) when is_list(dist1) and is_list(dist2) do
    if length(dist1) != length(dist2) do
      raise ArgumentError, "Distributions must have same length"
    end

    # Normalize distributions
    norm1 = normalize_distribution(dist1)
    norm2 = normalize_distribution(dist2)

    # Calculate Fisher-Rao distance using sqrt transformation
    inner_product =
      Enum.zip(norm1, norm2)
      |> Enum.map(fn {p, q} -> :math.sqrt(p) * :math.sqrt(q) end)
      |> Enum.sum()

    # Clamp to valid range for acos
    clamped = max(-1.0, min(1.0, inner_product))
    distance = 2 * :math.acos(clamped)

    Float.round(distance, 4)
  end

  @doc """
  Calculate schema compliance for SNOs.

  Checks that SNOs have required fields and valid values.

  ## Examples

      iex> sno = CNS.SNO.new("Valid claim", confidence: 0.8)
      iex> compliance = CNS.Metrics.schema_compliance([sno])
      iex> compliance
      1.0
  """
  @spec schema_compliance([SNO.t()]) :: float()
  def schema_compliance(snos) when is_list(snos) do
    if Enum.empty?(snos) do
      1.0
    else
      valid_count =
        Enum.count(snos, fn sno ->
          case SNO.validate(sno) do
            {:ok, _} -> true
            _ -> false
          end
        end)

      Float.round(valid_count / length(snos), 4)
    end
  end

  @doc """
  Calculate mean entailment across multiple synthesis operations.

  ## Examples

      iex> results = [
      ...>   {CNS.SNO.new("P1"), CNS.SNO.new("C1")},
      ...>   {CNS.SNO.new("P2"), CNS.SNO.new("C2")}
      ...> ]
      iex> mean = CNS.Metrics.mean_entailment(results)
      iex> mean >= 0.0
      true
  """
  @spec mean_entailment([{SNO.t(), SNO.t()}]) :: float()
  def mean_entailment(premise_conclusion_pairs) when is_list(premise_conclusion_pairs) do
    if Enum.empty?(premise_conclusion_pairs) do
      0.0
    else
      scores =
        Enum.map(premise_conclusion_pairs, fn {premise, conclusion} ->
          entailment(premise, conclusion)
        end)

      Float.round(Enum.sum(scores) / length(scores), 4)
    end
  end

  @doc """
  Calculate convergence delta between iterations.

  ## Examples

      iex> prev = CNS.SNO.new("Old", confidence: 0.6)
      iex> curr = CNS.SNO.new("New", confidence: 0.8)
      iex> delta = CNS.Metrics.convergence_delta(prev, curr)
      iex> delta
      0.2
  """
  @spec convergence_delta(SNO.t(), SNO.t()) :: float()
  def convergence_delta(%SNO{confidence: prev_conf}, %SNO{confidence: curr_conf}) do
    Float.round(curr_conf - prev_conf, 4)
  end

  @doc """
  Check if metrics meet CNS 3.0 quality targets.

  ## Examples

      iex> metrics = %{schema_compliance: 0.96, citation_accuracy: 0.97, mean_entailment: 0.55}
      iex> CNS.Metrics.meets_targets?(metrics)
      true
  """
  @spec meets_targets?(map()) :: boolean()
  def meets_targets?(metrics) when is_map(metrics) do
    targets = CNS.Config.quality_targets()

    Map.get(metrics, :schema_compliance, 0) >= targets.schema_compliance and
      Map.get(metrics, :citation_accuracy, 0) >= targets.citation_accuracy and
      Map.get(metrics, :mean_entailment, 0) >= targets.mean_entailment
  end

  @doc """
  Generate comprehensive metrics report.
  """
  @spec report([SNO.t()], [Challenge.t()]) :: map()
  def report(snos, challenges \\ []) do
    %{
      count: length(snos),
      schema_compliance: schema_compliance(snos),
      citation_accuracy: citation_accuracy(snos),
      mean_confidence: mean_confidence(snos),
      chirality: chirality(challenges),
      pass_rate_50: pass_rate(snos, 0.5),
      pass_rate_85: pass_rate(snos, 0.85),
      evidence_count: Enum.sum(Enum.map(snos, &length(&1.evidence)))
    }
  end

  # Private functions

  defp evidence_quality_score(%SNO{evidence: []}) do
    0.0
  end

  defp evidence_quality_score(%SNO{evidence: evidence}) do
    scores = Enum.map(evidence, &Evidence.score/1)
    Float.round(Enum.sum(scores) / length(scores), 4)
  end

  defp evidence_coverage(%SNO{evidence: evidence}) do
    min(1.0, length(evidence) / 3.0)
  end

  defp extract_terms(text) do
    stop_words =
      ~w(the a an is are was were be been have has do does will would could should may might must shall can)

    text
    |> String.downcase()
    |> String.replace(~r/[^\w\s]/, "")
    |> String.split(~r/\s+/, trim: true)
    |> Enum.reject(&(&1 in stop_words))
    |> Enum.reject(&(String.length(&1) < 3))
  end

  defp term_overlap(terms1, terms2) do
    set1 = MapSet.new(terms1)
    set2 = MapSet.new(terms2)

    if MapSet.size(set1) == 0 or MapSet.size(set2) == 0 do
      0.0
    else
      intersection = MapSet.intersection(set1, set2) |> MapSet.size()
      union = MapSet.union(set1, set2) |> MapSet.size()
      intersection / union
    end
  end

  defp normalize_distribution(dist) do
    sum = Enum.sum(dist)

    if sum == 0 do
      n = length(dist)
      List.duplicate(1.0 / n, n)
    else
      Enum.map(dist, &(&1 / sum))
    end
  end

  defp mean_confidence([]), do: 0.0

  defp mean_confidence(snos) do
    total = Enum.sum(Enum.map(snos, & &1.confidence))
    Float.round(total / length(snos), 4)
  end
end

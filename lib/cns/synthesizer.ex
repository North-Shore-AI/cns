defmodule CNS.Synthesizer do
  @moduledoc """
  Synthesizer agent for reconciling conflicting claims into coherent synthesis.

  The Synthesizer integrates thesis and antithesis into a nuanced synthesis,
  preserving valid insights from both sides while maintaining coherence.

  ## Examples

      iex> thesis = CNS.SNO.new("Coffee improves focus", confidence: 0.8)
      iex> antithesis = CNS.SNO.new("Coffee causes anxiety", confidence: 0.7)
      iex> {:ok, synthesis} = CNS.Synthesizer.synthesize(thesis, antithesis)
      iex> synthesis.confidence >= 0.7
      true
  """

  alias CNS.{SNO, Evidence, Provenance, Challenge, Config}

  @doc """
  Synthesize thesis and antithesis into a coherent synthesis.

  ## Options

  * `:preserve_nuance` - Maintain complexity in output (default: true)
  * `:citation_validity_weight` - Weight for evidence quality (default: 0.4)
  * `:coherence_threshold` - Minimum synthesis quality (default: 0.8)

  ## Examples

      iex> thesis = CNS.SNO.new("A is true")
      iex> antithesis = CNS.SNO.new("A has exceptions")
      iex> {:ok, synthesis} = CNS.Synthesizer.synthesize(thesis, antithesis)
      iex> is_binary(synthesis.claim)
      true
  """
  @spec synthesize(SNO.t(), SNO.t(), keyword()) :: {:ok, SNO.t()} | {:error, term()}
  def synthesize(%SNO{} = thesis, %SNO{} = antithesis, opts \\ []) do
    preserve_nuance = Keyword.get(opts, :preserve_nuance, true)
    citation_weight = Keyword.get(opts, :citation_validity_weight, 0.4)

    # Generate synthesized claim
    claim = generate_synthesis_claim(thesis, antithesis, preserve_nuance)

    # Merge evidence from both sides
    evidence = merge_evidence(thesis.evidence, antithesis.evidence, citation_weight)

    # Calculate synthesis confidence
    confidence = calculate_synthesis_confidence(thesis, antithesis, evidence)

    # Create provenance
    provenance =
      Provenance.new(:synthesizer,
        parent_ids: [thesis.id, antithesis.id],
        transformation: "dialectical_synthesis",
        model_id: Keyword.get(opts, :model_id)
      )

    # Record synthesis history
    history_entry = %{
      thesis_id: thesis.id,
      antithesis_id: antithesis.id,
      timestamp: DateTime.utc_now(),
      confidence_delta: confidence - (thesis.confidence + antithesis.confidence) / 2
    }

    synthesis =
      SNO.new(claim,
        confidence: confidence,
        evidence: evidence,
        provenance: provenance,
        synthesis_history: [history_entry],
        metadata: %{
          thesis_claim: thesis.claim,
          antithesis_claim: antithesis.claim
        }
      )

    {:ok, synthesis}
  rescue
    e -> {:error, Exception.message(e)}
  end

  @doc """
  Ground a claim with additional evidence.

  ## Examples

      iex> sno = CNS.SNO.new("Claim")
      iex> evidence = [CNS.Evidence.new("New source", "Data")]
      iex> {:ok, grounded} = CNS.Synthesizer.ground_evidence(sno, evidence)
      iex> length(grounded.evidence) > 0
      true
  """
  @spec ground_evidence(SNO.t(), [Evidence.t()], keyword()) :: {:ok, SNO.t()} | {:error, term()}
  def ground_evidence(%SNO{} = sno, evidence, opts \\ []) when is_list(evidence) do
    validity_threshold = Keyword.get(opts, :validity_threshold, 0.5)

    # Filter evidence by validity
    valid_evidence = Enum.filter(evidence, &(&1.validity >= validity_threshold))

    # Merge with existing evidence
    merged = merge_evidence(sno.evidence, valid_evidence, 0.5)

    # Update confidence based on evidence quality
    evidence_boost = calculate_evidence_boost(valid_evidence)
    new_confidence = min(1.0, sno.confidence + evidence_boost)

    grounded = %{sno | evidence: merged, confidence: Float.round(new_confidence, 4)}

    {:ok, grounded}
  rescue
    e -> {:error, Exception.message(e)}
  end

  @doc """
  Resolve conflicts between claims based on challenges.

  ## Examples

      iex> thesis = CNS.SNO.new("A", confidence: 0.8)
      iex> antithesis = CNS.SNO.new("Not A", confidence: 0.7)
      iex> challenges = [CNS.Challenge.new(thesis.id, :contradiction, "Conflict")]
      iex> {:ok, resolved} = CNS.Synthesizer.resolve_conflicts(thesis, antithesis, challenges)
      iex> is_binary(resolved.claim)
      true
  """
  @spec resolve_conflicts(SNO.t(), SNO.t(), [Challenge.t()], keyword()) ::
          {:ok, SNO.t()} | {:error, term()}
  def resolve_conflicts(%SNO{} = thesis, %SNO{} = antithesis, challenges, opts \\ []) do
    # Adjust confidence based on challenges
    thesis_adjusted = adjust_for_challenges(thesis, challenges)
    antithesis_adjusted = adjust_for_challenges(antithesis, challenges)

    # Synthesize with adjusted confidence
    synthesize(thesis_adjusted, antithesis_adjusted, opts)
  end

  @doc """
  Calculate coherence score for a synthesis.

  ## Examples

      iex> e = CNS.Evidence.new("Source", "Content", validity: 0.8)
      iex> sno = CNS.SNO.new("Coherent claim", evidence: [e], confidence: 0.9)
      iex> score = CNS.Synthesizer.coherence_score(sno)
      iex> score > 0.5
      true
  """
  @spec coherence_score(SNO.t()) :: float()
  def coherence_score(%SNO{} = sno) do
    # Base coherence from confidence
    confidence_factor = sno.confidence * 0.4

    # Evidence coverage factor
    evidence_factor = min(1.0, length(sno.evidence) / 3) * 0.3

    # Evidence quality factor
    quality_factor = SNO.evidence_score(sno) * 0.3

    Float.round(confidence_factor + evidence_factor + quality_factor, 4)
  end

  @doc """
  Calculate entailment score between claims.

  Measures how well the synthesis follows from the premises.

  ## Examples

      iex> thesis = CNS.SNO.new("Coffee has caffeine")
      iex> antithesis = CNS.SNO.new("Caffeine affects alertness")
      iex> synthesis = CNS.SNO.new("Coffee affects alertness through caffeine")
      iex> score = CNS.Synthesizer.entailment_score(thesis, antithesis, synthesis)
      iex> score > 0
      true
  """
  @spec entailment_score(SNO.t(), SNO.t(), SNO.t()) :: float()
  def entailment_score(%SNO{} = thesis, %SNO{} = antithesis, %SNO{} = synthesis) do
    thesis_words = extract_key_words(thesis.claim)
    antithesis_words = extract_key_words(antithesis.claim)
    synthesis_words = extract_key_words(synthesis.claim)

    # Calculate word overlap
    thesis_overlap = word_overlap(thesis_words, synthesis_words)
    antithesis_overlap = word_overlap(antithesis_words, synthesis_words)

    # Balance factor - synthesis should draw from both
    balance = 1 - abs(thesis_overlap - antithesis_overlap)

    # Combined score
    base_score = (thesis_overlap + antithesis_overlap) / 2
    Float.round(base_score * 0.7 + balance * 0.3, 4)
  end

  @doc """
  Process synthesis with full configuration.
  """
  @spec process(SNO.t(), SNO.t(), [Challenge.t()], Config.t()) :: {:ok, map()} | {:error, term()}
  def process(%SNO{} = thesis, %SNO{} = antithesis, challenges, %Config{} = config) do
    synthesizer_config = config.synthesizer

    with {:ok, synthesis} <-
           resolve_conflicts(thesis, antithesis, challenges, Map.to_list(synthesizer_config)) do
      coherence = coherence_score(synthesis)
      entailment = entailment_score(thesis, antithesis, synthesis)

      result = %{
        synthesis: synthesis,
        coherence_score: coherence,
        entailment_score: entailment,
        meets_threshold: coherence >= config.coherence_threshold,
        evidence_count: length(synthesis.evidence)
      }

      {:ok, result}
    end
  end

  # Private functions

  defp generate_synthesis_claim(
         %SNO{claim: thesis_claim},
         %SNO{claim: antithesis_claim},
         preserve_nuance
       ) do
    # Extract key concepts from both claims
    thesis_concepts = extract_key_words(thesis_claim)
    antithesis_concepts = extract_key_words(antithesis_claim)

    # Find common and unique concepts
    common = MapSet.intersection(MapSet.new(thesis_concepts), MapSet.new(antithesis_concepts))

    if preserve_nuance and MapSet.size(common) > 0 do
      common_str = common |> MapSet.to_list() |> Enum.join(", ")

      "While both perspectives address #{common_str}, " <>
        "a synthesis reveals that #{simplify_claim(thesis_claim)} while also acknowledging #{simplify_claim(antithesis_claim)}."
    else
      "A balanced view suggests that #{simplify_claim(thesis_claim)}, " <>
        "however #{simplify_claim(antithesis_claim)}, " <>
        "indicating a more nuanced understanding is needed."
    end
  end

  defp simplify_claim(claim) do
    claim
    |> String.downcase()
    |> String.replace(~r/^\s*(the|a|an)\s+/i, "")
    |> String.trim()
    |> String.slice(0, 100)
  end

  defp merge_evidence(evidence1, evidence2, weight) do
    all_evidence = evidence1 ++ evidence2

    # Deduplicate by source
    all_evidence
    |> Enum.uniq_by(& &1.source)
    |> Enum.sort_by(fn e -> -e.validity * weight - e.relevance * (1 - weight) end)
  end

  defp calculate_synthesis_confidence(%SNO{} = thesis, %SNO{} = antithesis, evidence) do
    # Base confidence is weighted average
    base = thesis.confidence * 0.4 + antithesis.confidence * 0.4

    # Evidence boost
    evidence_avg =
      if Enum.empty?(evidence) do
        0.0
      else
        Enum.sum(Enum.map(evidence, & &1.validity)) / length(evidence)
      end

    evidence_boost = evidence_avg * 0.2

    # Synthesis typically increases confidence through resolution
    synthesis_boost = 0.05

    min(1.0, Float.round(base + evidence_boost + synthesis_boost, 4))
  end

  defp calculate_evidence_boost(evidence) when is_list(evidence) do
    if Enum.empty?(evidence) do
      0.0
    else
      avg_validity = Enum.sum(Enum.map(evidence, & &1.validity)) / length(evidence)
      count_factor = min(1.0, length(evidence) / 5)
      Float.round(avg_validity * count_factor * 0.1, 4)
    end
  end

  defp adjust_for_challenges(%SNO{} = sno, challenges) do
    # Find challenges targeting this SNO
    relevant = Enum.filter(challenges, &(&1.target_id == sno.id))

    if Enum.empty?(relevant) do
      sno
    else
      # Calculate confidence penalty based on challenges
      penalty =
        relevant
        |> Enum.map(&Challenge.chirality_score/1)
        |> Enum.sum()
        |> Kernel.*(0.1)

      new_confidence = max(0.1, sno.confidence - penalty)
      %{sno | confidence: Float.round(new_confidence, 4)}
    end
  end

  defp extract_key_words(text) do
    # Remove common stop words and extract key terms
    stop_words =
      ~w(the a an is are was were be been being have has had do does did will would could should may might must shall can)

    text
    |> String.downcase()
    |> String.replace(~r/[^\w\s]/, "")
    |> String.split(~r/\s+/, trim: true)
    |> Enum.reject(&(&1 in stop_words))
    |> Enum.reject(&(String.length(&1) < 3))
  end

  defp word_overlap(words1, words2) do
    set1 = MapSet.new(words1)
    set2 = MapSet.new(words2)

    if MapSet.size(set1) == 0 or MapSet.size(set2) == 0 do
      0.0
    else
      intersection = MapSet.intersection(set1, set2) |> MapSet.size()
      union = MapSet.union(set1, set2) |> MapSet.size()
      intersection / union
    end
  end
end

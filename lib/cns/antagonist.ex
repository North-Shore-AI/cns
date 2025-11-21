defmodule CNS.Antagonist do
  @moduledoc """
  Antagonist agent for challenging claims with counter-evidence.

  The Antagonist identifies weaknesses in proposed claims, generates
  counter-arguments, and scores the severity of challenges.

  ## Examples

      iex> sno = CNS.SNO.new("All swans are white", confidence: 0.8)
      iex> {:ok, challenges} = CNS.Antagonist.challenge(sno)
      iex> Enum.all?(challenges, &match?(%CNS.Challenge{}, &1))
      true
  """

  alias CNS.{SNO, Challenge, Config}

  @doc """
  Generate challenges for a given claim.

  Analyzes the claim for potential weaknesses and generates
  challenges with severity scores.

  ## Options

  * `:max_challenges` - Maximum challenges to generate (default: 5)
  * `:critique_depth` - :quick, :standard, or :thorough (default: :standard)
  * `:evidence_search` - Search for counter-evidence (default: true)

  ## Examples

      iex> sno = CNS.SNO.new("Coffee is healthy")
      iex> {:ok, challenges} = CNS.Antagonist.challenge(sno)
      iex> length(challenges) > 0
      true
  """
  @spec challenge(SNO.t(), keyword()) :: {:ok, [Challenge.t()]} | {:error, term()}
  def challenge(%SNO{} = sno, opts \\ []) do
    max_challenges = Keyword.get(opts, :max_challenges, 5)
    _critique_depth = Keyword.get(opts, :critique_depth, :standard)

    challenges =
      []
      |> find_contradictions(sno)
      |> find_evidence_gaps(sno)
      |> find_scope_issues(sno)
      |> find_logical_issues(sno)
      |> generate_alternatives(sno)
      |> Enum.take(max_challenges)

    {:ok, challenges}
  rescue
    e -> {:error, Exception.message(e)}
  end

  @doc """
  Find potential contradictions in a claim.

  ## Examples

      iex> sno = CNS.SNO.new("Always and never simultaneously")
      iex> contradictions = CNS.Antagonist.find_contradictions(sno)
      iex> is_list(contradictions)
      true
  """
  @spec find_contradictions(SNO.t()) :: [Challenge.t()]
  def find_contradictions(%SNO{} = sno) do
    find_contradictions([], sno)
  end

  defp find_contradictions(challenges, %SNO{} = sno) do
    claim_lower = String.downcase(sno.claim)

    # Check for absolute terms that are often over-generalizations
    absolute_terms = [
      "always",
      "never",
      "all",
      "none",
      "everyone",
      "no one",
      "completely",
      "totally"
    ]

    contradictions =
      Enum.filter(absolute_terms, &String.contains?(claim_lower, &1))
      |> Enum.map(fn term ->
        Challenge.new(
          sno.id,
          :contradiction,
          "Claim uses absolute term '#{term}' which may not hold in all cases",
          severity: :medium,
          confidence: 0.6
        )
      end)

    # Check for internal contradictions
    internal = detect_internal_contradiction(sno.claim)

    challenges ++ contradictions ++ internal
  end

  @doc """
  Find evidence gaps in a claim.

  ## Examples

      iex> sno = CNS.SNO.new("This is true", evidence: [])
      iex> gaps = CNS.Antagonist.find_evidence_gaps(sno)
      iex> length(gaps) > 0
      true
  """
  @spec find_evidence_gaps(SNO.t()) :: [Challenge.t()]
  def find_evidence_gaps(%SNO{} = sno) do
    find_evidence_gaps([], sno)
  end

  defp find_evidence_gaps(challenges, %SNO{} = sno) do
    gaps = []

    # Check for missing evidence
    gaps =
      if Enum.empty?(sno.evidence) do
        [
          Challenge.new(sno.id, :evidence_gap, "Claim lacks supporting evidence",
            severity: :high,
            confidence: 0.9
          )
          | gaps
        ]
      else
        gaps
      end

    # Check for low validity evidence
    low_validity_evidence = Enum.filter(sno.evidence, &(&1.validity < 0.5))

    gaps =
      if length(low_validity_evidence) > 0 do
        [
          Challenge.new(
            sno.id,
            :evidence_gap,
            "#{length(low_validity_evidence)} evidence sources have low validity scores",
            severity: :medium,
            confidence: 0.7
          )
          | gaps
        ]
      else
        gaps
      end

    # Check average evidence score
    avg_score = SNO.evidence_score(sno)

    gaps =
      if avg_score > 0 and avg_score < 0.6 do
        [
          Challenge.new(
            sno.id,
            :evidence_gap,
            "Average evidence validity (#{Float.round(avg_score, 2)}) is below threshold",
            severity: :medium,
            confidence: 0.6
          )
          | gaps
        ]
      else
        gaps
      end

    challenges ++ gaps
  end

  @doc """
  Find scope limitation issues in a claim.
  """
  @spec find_scope_issues(SNO.t()) :: [Challenge.t()]
  def find_scope_issues(%SNO{} = sno) do
    find_scope_issues([], sno)
  end

  defp find_scope_issues(challenges, %SNO{} = sno) do
    claim_lower = String.downcase(sno.claim)

    issues = []

    # Check for broad generalizations
    generalization_markers = ["in general", "typically", "usually", "often", "most"]

    has_generalization = Enum.any?(generalization_markers, &String.contains?(claim_lower, &1))

    issues =
      if has_generalization do
        [
          Challenge.new(sno.id, :scope, "Claim makes broad generalization - scope may be too wide",
            severity: :low,
            confidence: 0.5
          )
          | issues
        ]
      else
        issues
      end

    # Check for missing context
    issues =
      if String.length(sno.claim) < 50 and sno.confidence > 0.7 do
        [
          Challenge.new(
            sno.id,
            :scope,
            "Short claim with high confidence may lack necessary context",
            severity: :low,
            confidence: 0.4
          )
          | issues
        ]
      else
        issues
      end

    challenges ++ issues
  end

  @doc """
  Find logical issues in a claim.
  """
  @spec find_logical_issues(SNO.t()) :: [Challenge.t()]
  def find_logical_issues(%SNO{} = sno) do
    find_logical_issues([], sno)
  end

  defp find_logical_issues(challenges, %SNO{} = sno) do
    claim_lower = String.downcase(sno.claim)

    issues = []

    # Check for causal language without evidence
    causal_markers = ["causes", "leads to", "results in", "because", "therefore", "thus"]

    has_causal = Enum.any?(causal_markers, &String.contains?(claim_lower, &1))

    issues =
      if has_causal and length(sno.evidence) < 2 do
        [
          Challenge.new(sno.id, :logical, "Causal claim requires stronger evidence support",
            severity: :medium,
            confidence: 0.6
          )
          | issues
        ]
      else
        issues
      end

    # Check for circular reasoning indicators
    issues =
      if String.contains?(claim_lower, "because it is") or
           String.contains?(claim_lower, "since it's") do
        [
          Challenge.new(sno.id, :logical, "Potential circular reasoning detected",
            severity: :high,
            confidence: 0.5
          )
          | issues
        ]
      else
        issues
      end

    challenges ++ issues
  end

  @doc """
  Generate alternative interpretations for a claim.
  """
  @spec generate_alternatives(SNO.t()) :: [Challenge.t()]
  def generate_alternatives(%SNO{} = sno) do
    generate_alternatives([], sno)
  end

  defp generate_alternatives(challenges, %SNO{} = sno) do
    # Always suggest considering alternatives for high-confidence claims
    if sno.confidence > 0.8 do
      alt =
        Challenge.new(
          sno.id,
          :alternative,
          "Consider alternative explanations for this high-confidence claim",
          severity: :low,
          confidence: 0.4
        )

      challenges ++ [alt]
    else
      challenges
    end
  end

  @doc """
  Calculate chirality score for a set of challenges.

  Chirality measures the degree of conflict/tension that needs resolution.

  ## Examples

      iex> challenges = [
      ...>   CNS.Challenge.new("id", :contradiction, "Test", severity: :high, confidence: 0.9)
      ...> ]
      iex> score = CNS.Antagonist.score_chirality(challenges)
      iex> score > 0
      true
  """
  @spec score_chirality([Challenge.t()]) :: float()
  def score_chirality(challenges) when is_list(challenges) do
    if Enum.empty?(challenges) do
      0.0
    else
      scores = Enum.map(challenges, &Challenge.chirality_score/1)
      Float.round(Enum.sum(scores) / length(scores), 4)
    end
  end

  @doc """
  Flag issues by severity level.

  Returns challenges grouped by severity.

  ## Examples

      iex> challenges = [
      ...>   CNS.Challenge.new("id", :contradiction, "High", severity: :high),
      ...>   CNS.Challenge.new("id", :scope, "Low", severity: :low)
      ...> ]
      iex> flagged = CNS.Antagonist.flag_issues(challenges)
      iex> length(flagged.high)
      1
  """
  @spec flag_issues([Challenge.t()]) :: %{
          high: [Challenge.t()],
          medium: [Challenge.t()],
          low: [Challenge.t()]
        }
  def flag_issues(challenges) do
    %{
      high: Enum.filter(challenges, &(&1.severity == :high)),
      medium: Enum.filter(challenges, &(&1.severity == :medium)),
      low: Enum.filter(challenges, &(&1.severity == :low))
    }
  end

  @doc """
  Process claims through antagonist with full configuration.
  """
  @spec process([SNO.t()], Config.t()) :: {:ok, map()} | {:error, term()}
  def process(claims, %Config{} = config) when is_list(claims) do
    antagonist_config = config.antagonist

    all_challenges =
      Enum.flat_map(claims, fn claim ->
        case challenge(claim, Map.to_list(antagonist_config)) do
          {:ok, challenges} -> challenges
          _ -> []
        end
      end)

    result = %{
      challenges: all_challenges,
      count: length(all_challenges),
      chirality_score: score_chirality(all_challenges),
      by_severity: flag_issues(all_challenges),
      critical_count: Enum.count(all_challenges, &Challenge.critical?/1)
    }

    {:ok, result}
  end

  # Private helper functions

  defp detect_internal_contradiction(claim) do
    claim_lower = String.downcase(claim)

    contradictory_pairs = [
      {"increase", "decrease"},
      {"improve", "worsen"},
      {"always", "never"},
      {"more", "less"},
      {"higher", "lower"}
    ]

    Enum.filter(contradictory_pairs, fn {a, b} ->
      String.contains?(claim_lower, a) and String.contains?(claim_lower, b)
    end)
    |> Enum.map(fn {a, b} ->
      Challenge.new(
        "internal",
        :contradiction,
        "Claim contains potentially contradictory terms: '#{a}' and '#{b}'",
        severity: :high,
        confidence: 0.7
      )
    end)
  end
end

defmodule CNS.Agents.Proposer do
  @moduledoc """
  Proposer agent for generating initial claims and hypotheses.

  The Proposer extracts structured claims from input text, assigns confidence
  scores, and identifies supporting evidence.

  ## Examples

      iex> {:ok, claims} = CNS.Agents.Proposer.extract_claims("Studies show coffee improves focus.")
      iex> length(claims) > 0
      true

      iex> {:ok, hypothesis} = CNS.Agents.Proposer.generate_hypothesis("Does exercise help sleep?")
      iex> is_binary(hypothesis.claim)
      true
  """

  alias CNS.{SNO, Evidence, Provenance, Config}

  @doc """
  Extract claims from input text.

  Parses text to identify distinct claims, assigning confidence scores
  based on linguistic markers and evidence quality.

  ## Options

  * `:min_confidence` - Filter claims below this threshold (default: 0.3)
  * `:max_claims` - Maximum number of claims to extract (default: 10)
  * `:extract_evidence` - Auto-extract evidence from text (default: true)

  ## Examples

      iex> text = "Research shows A. However, B is also true."
      iex> {:ok, claims} = CNS.Agents.Proposer.extract_claims(text)
      iex> Enum.all?(claims, &match?(%CNS.SNO{}, &1))
      true
  """
  @spec extract_claims(String.t(), keyword()) :: {:ok, [SNO.t()]} | {:error, term()}
  def extract_claims(text, opts \\ []) when is_binary(text) do
    min_confidence = Keyword.get(opts, :min_confidence, 0.3)
    max_claims = Keyword.get(opts, :max_claims, 10)
    extract_evidence = Keyword.get(opts, :extract_evidence, true)

    # Extract sentences that appear to make claims
    claims =
      text
      |> split_into_sentences()
      |> Enum.map(&analyze_sentence(&1, extract_evidence))
      |> Enum.reject(&is_nil/1)
      |> Enum.filter(fn sno -> sno.confidence >= min_confidence end)
      |> Enum.take(max_claims)

    {:ok, claims}
  rescue
    e -> {:error, Exception.message(e)}
  end

  @doc """
  Generate a hypothesis from a research question.

  ## Examples

      iex> {:ok, hypothesis} = CNS.Proposer.generate_hypothesis("Does X cause Y?")
      iex> hypothesis.claim =~ "X" or hypothesis.claim =~ "Y"
      true
  """
  @spec generate_hypothesis(String.t(), keyword()) :: {:ok, SNO.t()} | {:error, term()}
  def generate_hypothesis(question, opts \\ []) when is_binary(question) do
    # Convert question into hypothesis claim
    claim = question_to_hypothesis(question)
    confidence = Keyword.get(opts, :initial_confidence, 0.5)

    provenance =
      Provenance.new(:proposer,
        transformation: "hypothesis_generation",
        model_id: Keyword.get(opts, :model_id)
      )

    sno =
      SNO.new(claim,
        confidence: confidence,
        provenance: provenance,
        metadata: %{source_question: question}
      )

    {:ok, sno}
  rescue
    e -> {:error, Exception.message(e)}
  end

  @doc """
  Score the confidence of a claim based on linguistic markers.

  Returns a score between 0.0 and 1.0.

  ## Examples

      iex> CNS.Proposer.score_confidence("Studies conclusively show...")
      score when score > 0.7

      iex> CNS.Proposer.score_confidence("Some people think...")
      score when score < 0.5
  """
  @spec score_confidence(String.t()) :: float()
  def score_confidence(text) when is_binary(text) do
    base_score = 0.5

    # High confidence markers
    high_markers =
      ~w(conclusively proven demonstrate demonstrated establishes established confirms confirmed verified)

    high_boost = count_markers(text, high_markers) * 0.1

    # Medium confidence markers
    med_markers = ~w(shows indicates suggests research study evidence)
    med_boost = count_markers(text, med_markers) * 0.05

    # Low confidence markers (reduce score)
    low_markers = ~w(maybe perhaps possibly might could some)
    low_penalty = count_markers(text, low_markers) * 0.1

    score = base_score + high_boost + med_boost - low_penalty
    Float.round(max(0.0, min(1.0, score)), 2)
  end

  @doc """
  Extract evidence from text for a given claim.

  ## Examples

      iex> text = "According to Smith (2023), the effect is significant."
      iex> {:ok, evidence} = CNS.Proposer.extract_evidence(text)
      iex> length(evidence) > 0
      true
  """
  @spec extract_evidence(String.t(), keyword()) :: {:ok, [Evidence.t()]} | {:error, term()}
  def extract_evidence(text, _opts \\ []) when is_binary(text) do
    # Pattern matching for citations and sources
    evidence =
      []
      |> extract_citations(text)
      |> extract_study_references(text)
      |> Enum.uniq_by(& &1.source)

    {:ok, evidence}
  rescue
    e -> {:error, Exception.message(e)}
  end

  @doc """
  Process input through proposer with full configuration.

  ## Examples

      iex> config = %CNS.Config{}
      iex> {:ok, result} = CNS.Proposer.process("Input text", config)
      iex> Map.has_key?(result, :claims)
      true
  """
  @spec process(String.t(), Config.t()) :: {:ok, map()} | {:error, term()}
  def process(input, %Config{} = config) do
    proposer_config = config.proposer

    with {:ok, claims} <- extract_claims(input, Map.to_list(proposer_config)) do
      result = %{
        claims: claims,
        count: length(claims),
        avg_confidence: average_confidence(claims),
        total_evidence: count_evidence(claims)
      }

      {:ok, result}
    end
  end

  # Private functions

  defp split_into_sentences(text) do
    text
    |> String.replace(~r/([.!?])\s+/, "\\1\n")
    |> String.split("\n", trim: true)
    |> Enum.map(&String.trim/1)
    |> Enum.reject(&(String.length(&1) < 10))
  end

  defp analyze_sentence(sentence, extract_evidence?) do
    # Skip questions and fragments
    if String.ends_with?(sentence, "?") or String.length(sentence) < 15 do
      nil
    else
      confidence = score_confidence(sentence)

      evidence =
        if extract_evidence? do
          case extract_evidence(sentence) do
            {:ok, ev} -> ev
            _ -> []
          end
        else
          []
        end

      provenance = Provenance.new(:proposer, transformation: "claim_extraction")

      SNO.new(sentence,
        confidence: confidence,
        evidence: evidence,
        provenance: provenance
      )
    end
  end

  defp question_to_hypothesis(question) do
    question
    |> String.trim_trailing("?")
    |> String.replace(~r/^(Does|Do|Is|Are|Can|Will|Would|Should)\s+/i, "")
    |> String.capitalize()
    |> Kernel.<>(" is likely to be true based on available evidence.")
  end

  defp count_markers(text, markers) do
    text_lower = String.downcase(text)

    Enum.count(markers, fn marker ->
      String.contains?(text_lower, marker)
    end)
  end

  defp extract_citations(evidence_list, text) do
    # Match patterns like (Author, 2023), (Author 2023), or Author (2023)
    # Pattern 1: Author inside parens - (Smith, 2023) or (Smith 2023)
    pattern1 = ~r/\(([A-Z][a-z]+(?:\s+et\s+al\.)?)[,\s]+(\d{4})\)/
    # Pattern 2: Author outside parens - Smith (2023)
    pattern2 = ~r/([A-Z][a-z]+(?:\s+et\s+al\.)?)\s+\((\d{4})\)/

    citations1 =
      Regex.scan(pattern1, text)
      |> Enum.map(fn [_full, author, year] ->
        Evidence.new("#{author} (#{year})", "",
          retrieval_method: :citation,
          validity: 0.8
        )
      end)

    citations2 =
      Regex.scan(pattern2, text)
      |> Enum.map(fn [_full, author, year] ->
        Evidence.new("#{author} (#{year})", "",
          retrieval_method: :citation,
          validity: 0.8
        )
      end)

    evidence_list ++ citations1 ++ citations2
  end

  defp extract_study_references(evidence_list, text) do
    # Match patterns like "study shows", "research indicates"
    study_pattern =
      ~r/(study|research|analysis|survey|experiment)s?\s+(show|indicate|suggest|find|found|demonstrate)/i

    if Regex.match?(study_pattern, text) do
      ref =
        Evidence.new("Referenced study", text,
          retrieval_method: :inference,
          validity: 0.6
        )

      evidence_list ++ [ref]
    else
      evidence_list
    end
  end

  defp average_confidence([]), do: 0.0

  defp average_confidence(claims) do
    total = Enum.sum(Enum.map(claims, & &1.confidence))
    Float.round(total / length(claims), 4)
  end

  defp count_evidence(claims) do
    Enum.sum(Enum.map(claims, fn c -> length(c.evidence) end))
  end
end

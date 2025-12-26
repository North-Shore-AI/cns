defmodule CNS do
  @moduledoc """
  CNS (Chiral Narrative Synthesis) - Dialectical reasoning framework for automated knowledge discovery.

  CNS implements a three-agent dialectical reasoning system inspired by Hegelian dialectics:

  1. **Proposer** - Generates thesis claims from input
  2. **Antagonist** - Creates antithesis challenges
  3. **Synthesizer** - Reconciles into synthesis

  ## Quick Start

      # Define conflicting claims
      thesis = %CNS.SNO{
        claim: "Remote work increases productivity",
        evidence: [%CNS.Evidence{source: "Stanford Study 2023", validity: 0.85}],
        confidence: 0.75
      }

      antithesis = %CNS.SNO{
        claim: "Remote work decreases collaboration",
        evidence: [%CNS.Evidence{source: "Microsoft Research 2023", validity: 0.82}],
        confidence: 0.70
      }

      # Run dialectical synthesis
      {:ok, synthesis} = CNS.synthesize(thesis, antithesis)

  ## Configuration

      config = %CNS.Config{
        max_iterations: 5,
        convergence_threshold: 0.85
      }

      {:ok, result} = CNS.Pipeline.run("Research question?", config)
  """

  alias CNS.Agents.{Pipeline, Proposer, Synthesizer}
  alias CNS.{Config, SNO}

  @doc """
  Synthesize two conflicting claims into a coherent synthesis.

  ## Examples

      iex> thesis = %CNS.SNO{claim: "A is true", confidence: 0.8}
      iex> antithesis = %CNS.SNO{claim: "A is false", confidence: 0.7}
      iex> {:ok, synthesis} = CNS.synthesize(thesis, antithesis)
      iex> is_binary(synthesis.claim)
      true
  """
  @spec synthesize(SNO.t(), SNO.t(), keyword()) :: {:ok, SNO.t()} | {:error, term()}
  def synthesize(thesis, antithesis, opts \\ []) do
    Synthesizer.synthesize(thesis, antithesis, opts)
  end

  @doc """
  Run the full dialectical pipeline on input text.

  ## Examples

      iex> config = %CNS.Config{max_iterations: 3}
      iex> {:ok, result} = CNS.run("What are the effects of exercise?", config)
      iex> Map.has_key?(result, :final_synthesis)
      true
  """
  @spec run(String.t(), Config.t()) :: {:ok, map()} | {:error, term()}
  def run(input, config \\ %Config{}) do
    Pipeline.run(input, config)
  end

  @doc """
  Extract claims from text using the Proposer agent.

  ## Examples

      {:ok, claims} = CNS.extract_claims("The study found significant results.")

  ## Options

  - `:max_claims` - Maximum claims to extract (default: 10)
  - `:confidence_threshold` - Minimum confidence (default: 0.5)
  """
  @spec extract_claims(text :: String.t(), opts :: keyword()) ::
          {:ok, [SNO.t()]} | {:error, term()}
  defdelegate extract_claims(text, opts \\ []), to: Proposer

  @doc """
  Run full dialectical pipeline until convergence.

  Orchestrates Proposer → Antagonist → Synthesizer cycles.

  ## Examples

      config = CNS.Config.new(max_iterations: 5)
      {:ok, result} = CNS.run_pipeline("Research question here", config)
  """
  @spec run_pipeline(input :: String.t(), config :: Config.t()) ::
          {:ok, map()} | {:error, term()}
  defdelegate run_pipeline(input, config), to: Pipeline, as: :run

  @doc """
  Validate claim against evidence corpus.

  Uses semantic validation (NLI) and citation checking.
  Validates citations, computes entailment and similarity scores.

  ## Parameters

    - `sno` - The SNO to validate
    - `corpus` - List of corpus documents as maps with :id and :text/:abstract keys
    - `opts` - Options including:
      - `:gold_ids` - Set of expected evidence document IDs (default: empty)
      - `:gold_claim` - Expected claim text for similarity (default: sno.claim)
      - `:entailment_threshold` - Minimum entailment score (default: 0.75)
      - `:similarity_threshold` - Minimum similarity score (default: 0.7)

  ## Examples

      corpus = [%{id: "doc1", text: "Evidence text"}]
      {:ok, validation} = CNS.validate(claim, corpus)

  ## Returns

      {:ok, %{
        valid: boolean(),
        citation_valid: boolean(),
        entailment_score: float(),
        similarity_score: float(),
        sno: SNO.t()
      }}
  """
  @spec validate(SNO.t(), corpus :: [map()], opts :: keyword()) ::
          {:ok, map()} | {:error, term()}
  def validate(%SNO{} = sno, corpus, opts \\ []) when is_list(corpus) do
    alias CNS.Validation.Semantic
    alias CNS.Validation.Semantic.Config, as: SemanticConfig

    # Build corpus map from list
    corpus_map = build_corpus_map(corpus)

    # Get gold IDs from options or extract from SNO evidence
    gold_ids =
      opts
      |> Keyword.get(:gold_ids, extract_evidence_ids(sno))
      |> MapSet.new()

    # Get gold claim for comparison
    gold_claim = Keyword.get(opts, :gold_claim, sno.claim)

    # Build semantic config
    config = %SemanticConfig{
      entailment_threshold: Keyword.get(opts, :entailment_threshold, 0.75),
      similarity_threshold: Keyword.get(opts, :similarity_threshold, 0.7)
    }

    # Run semantic validation pipeline
    result =
      Semantic.validate_claim(
        config,
        sno.claim,
        gold_claim,
        sno.claim,
        corpus_map,
        gold_ids
      )

    {:ok,
     %{
       valid: result.overall_pass,
       citation_valid: result.citation_valid,
       entailment_score: result.entailment_score,
       similarity_score: result.semantic_similarity,
       entailment_pass: result.entailment_pass,
       similarity_pass: result.similarity_pass,
       cited_ids: result.cited_ids,
       missing_ids: result.missing_ids,
       sno: sno
     }}
  rescue
    e -> {:error, Exception.message(e)}
  end

  # Build corpus map from list of documents
  defp build_corpus_map(corpus) when is_list(corpus) do
    Enum.reduce(corpus, %{}, fn doc, acc ->
      id = Map.get(doc, :id) || Map.get(doc, "id")
      if id, do: Map.put(acc, to_string(id), doc), else: acc
    end)
  end

  # Extract evidence IDs from SNO
  defp extract_evidence_ids(%SNO{evidence: evidence}) do
    evidence
    |> Enum.map(fn e -> e.id end)
    |> Enum.reject(&is_nil/1)
  end

  @doc """
  Get the current CNS version.
  """
  @spec version() :: String.t()
  def version, do: "0.2.0"
end

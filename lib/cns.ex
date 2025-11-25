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

  alias CNS.{SNO, Config}
  alias CNS.Agents.{Proposer, Synthesizer, Pipeline}

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

  ## Examples

      corpus = [%{id: "doc1", text: "..."}]
      {:ok, validation} = CNS.validate(claim, corpus)
  """
  @spec validate(SNO.t(), corpus :: [map()], opts :: keyword()) ::
          {:ok, map()} | {:error, term()}
  def validate(sno, _corpus, _opts \\ []) do
    # TODO: Implement proper validation wrapper
    # For now, just return a success tuple
    {:ok, %{valid: true, sno: sno}}
  end

  @doc """
  Get the current CNS version.
  """
  @spec version() :: String.t()
  def version, do: "0.2.0"
end

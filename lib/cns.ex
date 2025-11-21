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

  alias CNS.{SNO, Synthesizer, Pipeline, Config}

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
  Get the current CNS version.
  """
  @spec version() :: String.t()
  def version, do: "0.1.0"
end

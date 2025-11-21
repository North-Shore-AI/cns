defmodule CNS.Pipeline do
  @moduledoc """
  Pipeline orchestration for the three-agent dialectical reasoning process.

  The Pipeline coordinates Proposer, Antagonist, and Synthesizer agents,
  managing iteration and convergence towards stable synthesis.

  ## Examples

      iex> config = %CNS.Config{max_iterations: 3}
      iex> {:ok, result} = CNS.Pipeline.run("Does exercise improve sleep?", config)
      iex> Map.has_key?(result, :final_synthesis)
      true
  """

  alias CNS.{SNO, Proposer, Antagonist, Synthesizer, Config}

  @type pipeline_result :: %{
          final_synthesis: SNO.t(),
          iterations: non_neg_integer(),
          convergence_score: float(),
          evidence_chain: [CNS.Evidence.t()],
          challenges_resolved: non_neg_integer(),
          metrics: map()
        }

  @doc """
  Run the full dialectical pipeline on input text.

  ## Examples

      iex> config = %CNS.Config{}
      iex> {:ok, result} = CNS.Pipeline.run("Research question", config)
      iex> result.iterations >= 1
      true
  """
  @spec run(String.t(), Config.t()) :: {:ok, pipeline_result()} | {:error, term()}
  def run(input, %Config{} = config \\ %Config{}) do
    # Initialize pipeline state
    state = %{
      input: input,
      config: config,
      iteration: 0,
      claims: [],
      challenges: [],
      synthesis: nil,
      converged: false,
      metrics: %{}
    }

    # Execute pipeline loop
    case execute_loop(state) do
      {:ok, final_state} ->
        result = build_result(final_state)
        {:ok, result}

      {:error, reason} ->
        {:error, reason}
    end
  rescue
    e -> {:error, Exception.message(e)}
  end

  @doc """
  Configure pipeline with custom settings.

  ## Examples

      iex> config = CNS.Pipeline.configure(max_iterations: 10)
      iex> config.max_iterations
      10
  """
  @spec configure(keyword()) :: Config.t()
  def configure(opts \\ []) do
    Config.new(opts)
  end

  @doc """
  Check if pipeline has converged.

  ## Examples

      iex> synthesis = CNS.SNO.new("Result", confidence: 0.9)
      iex> config = %CNS.Config{convergence_threshold: 0.85}
      iex> CNS.Pipeline.converged?(synthesis, config)
      true
  """
  @spec converged?(SNO.t(), Config.t()) :: boolean()
  def converged?(%SNO{} = synthesis, %Config{} = config) do
    synthesis.confidence >= config.convergence_threshold and
      Synthesizer.coherence_score(synthesis) >= config.coherence_threshold and
      SNO.evidence_score(synthesis) >= config.evidence_threshold * 0.5
  end

  @doc """
  Execute a single iteration of the pipeline.

  ## Examples

      iex> claims = [CNS.SNO.new("Claim 1"), CNS.SNO.new("Claim 2")]
      iex> config = %CNS.Config{}
      iex> {:ok, result} = CNS.Pipeline.iterate(claims, config)
      iex> Map.has_key?(result, :synthesis)
      true
  """
  @spec iterate([SNO.t()], Config.t()) :: {:ok, map()} | {:error, term()}
  def iterate(claims, %Config{} = _config) when is_list(claims) do
    if length(claims) < 2 do
      # Not enough claims for dialectical synthesis
      case claims do
        [single] -> {:ok, %{synthesis: single, challenges: []}}
        [] -> {:error, "No claims to process"}
      end
    else
      # Take first two claims as thesis/antithesis
      [thesis | [antithesis | _rest]] = claims

      # Generate challenges
      with {:ok, thesis_challenges} <- Antagonist.challenge(thesis),
           {:ok, antithesis_challenges} <- Antagonist.challenge(antithesis) do
        all_challenges = thesis_challenges ++ antithesis_challenges

        # Synthesize
        with {:ok, synthesis} <- Synthesizer.resolve_conflicts(thesis, antithesis, all_challenges) do
          {:ok,
           %{
             synthesis: synthesis,
             challenges: all_challenges,
             thesis: thesis,
             antithesis: antithesis
           }}
        end
      end
    end
  end

  @doc """
  Run pipeline with async execution for parallel processing.
  """
  @spec run_async(String.t(), Config.t()) :: Task.t()
  def run_async(input, %Config{} = config \\ %Config{}) do
    Task.async(fn -> run(input, config) end)
  end

  @doc """
  Get pipeline status and metrics.
  """
  @spec status(map()) :: map()
  def status(state) when is_map(state) do
    %{
      iteration: state.iteration,
      converged: state.converged,
      claim_count: length(state.claims),
      challenge_count: length(state.challenges),
      has_synthesis: not is_nil(state.synthesis)
    }
  end

  # Private functions

  defp execute_loop(%{iteration: iteration, config: %{max_iterations: max}} = state)
       when iteration >= max do
    # Max iterations reached
    {:ok, %{state | converged: false}}
  end

  defp execute_loop(%{converged: true} = state) do
    {:ok, state}
  end

  defp execute_loop(state) do
    with {:ok, state} <- step_propose(state),
         {:ok, state} <- step_challenge(state),
         {:ok, state} <- step_synthesize(state),
         {:ok, state} <- check_convergence(state) do
      if state.converged do
        {:ok, state}
      else
        # Continue with synthesis as new input
        new_state = %{
          state
          | iteration: state.iteration + 1,
            claims: if(state.synthesis, do: [state.synthesis | state.claims], else: state.claims)
        }

        execute_loop(new_state)
      end
    end
  end

  defp step_propose(%{iteration: 0, input: input, config: _config} = state) do
    case Proposer.extract_claims(input) do
      {:ok, claims} when length(claims) >= 1 ->
        {:ok, %{state | claims: claims}}

      {:ok, []} ->
        # Generate hypothesis if no claims found
        case Proposer.generate_hypothesis(input) do
          {:ok, hypothesis} -> {:ok, %{state | claims: [hypothesis]}}
          error -> error
        end

      error ->
        error
    end
  end

  defp step_propose(state) do
    # Subsequent iterations use existing claims
    {:ok, state}
  end

  defp step_challenge(%{claims: claims, config: config} = state) do
    {:ok, result} = Antagonist.process(claims, config)
    {:ok, %{state | challenges: result.challenges}}
  end

  defp step_synthesize(%{claims: claims} = state) when length(claims) < 2 do
    # Not enough for synthesis
    synthesis = List.first(claims) || SNO.new("No synthesis possible", confidence: 0.0)
    {:ok, %{state | synthesis: synthesis}}
  end

  defp step_synthesize(%{claims: claims, challenges: challenges, config: config} = state) do
    [thesis | [antithesis | _]] = claims

    case Synthesizer.process(thesis, antithesis, challenges, config) do
      {:ok, result} ->
        metrics =
          Map.merge(state.metrics, %{
            coherence: result.coherence_score,
            entailment: result.entailment_score
          })

        {:ok, %{state | synthesis: result.synthesis, metrics: metrics}}

      error ->
        error
    end
  end

  defp check_convergence(%{synthesis: nil} = state) do
    {:ok, %{state | converged: false}}
  end

  defp check_convergence(%{synthesis: synthesis, config: config} = state) do
    converged = converged?(synthesis, config)
    {:ok, %{state | converged: converged}}
  end

  defp build_result(state) do
    final_synthesis = state.synthesis || SNO.new("Pipeline incomplete", confidence: 0.0)

    evidence_chain =
      state.claims
      |> Enum.flat_map(& &1.evidence)
      |> Enum.uniq_by(& &1.source)

    %{
      final_synthesis: final_synthesis,
      iterations: max(1, state.iteration),
      convergence_score: final_synthesis.confidence,
      evidence_chain: evidence_chain,
      challenges_resolved: length(state.challenges),
      converged: state.converged,
      metrics:
        Map.merge(state.metrics, %{
          total_claims: length(state.claims),
          total_challenges: length(state.challenges),
          final_confidence: final_synthesis.confidence,
          evidence_count: length(evidence_chain)
        })
    }
  end
end

defmodule CNS.Config do
  @moduledoc """
  Configuration for CNS pipeline execution.

  ## Fields

  * `:max_iterations` - Maximum dialectical iterations (default: 5)
  * `:convergence_threshold` - Target confidence for convergence (default: 0.85)
  * `:coherence_threshold` - Minimum coherence score (default: 0.8)
  * `:evidence_threshold` - Minimum evidence coverage (default: 0.7)
  * `:proposer` - Proposer agent configuration
  * `:antagonist` - Antagonist agent configuration
  * `:synthesizer` - Synthesizer agent configuration
  * `:telemetry_enabled` - Enable telemetry events (default: true)
  * `:timeout` - Timeout per operation in ms (default: 30000)

  ## Examples

      iex> config = %CNS.Config{max_iterations: 3, convergence_threshold: 0.9}
      iex> config.max_iterations
      3
  """

  @type agent_config :: %{
          optional(:model) => String.t(),
          optional(:temperature) => float(),
          optional(:max_tokens) => pos_integer(),
          optional(:ensemble) => boolean(),
          optional(:models) => [String.t()],
          optional(:voting_strategy) => atom(),
          optional(:lora_adapter) => String.t()
        }

  @type t :: %__MODULE__{
          max_iterations: pos_integer(),
          convergence_threshold: float(),
          coherence_threshold: float(),
          evidence_threshold: float(),
          proposer: agent_config(),
          antagonist: agent_config(),
          synthesizer: agent_config(),
          telemetry_enabled: boolean(),
          timeout: pos_integer(),
          metadata: map()
        }

  defstruct max_iterations: 5,
            convergence_threshold: 0.85,
            coherence_threshold: 0.8,
            evidence_threshold: 0.7,
            proposer: %{model: "gpt-4", temperature: 0.7, max_tokens: 2000},
            antagonist: %{model: "gpt-4", temperature: 0.8, max_tokens: 2000},
            synthesizer: %{model: "gpt-4", temperature: 0.3, max_tokens: 3000},
            telemetry_enabled: true,
            timeout: 30_000,
            metadata: %{}

  @doc """
  Create a new Config with optional overrides.

  ## Examples

      iex> config = CNS.Config.new(max_iterations: 10)
      iex> config.max_iterations
      10

      iex> config = CNS.Config.new(proposer: %{model: "claude-3"})
      iex> config.proposer.model
      "claude-3"
  """
  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    struct(__MODULE__, opts)
  end

  @doc """
  Validate a config struct.

  ## Examples

      iex> config = %CNS.Config{}
      iex> CNS.Config.validate(config)
      {:ok, config}
  """
  @spec validate(t()) :: {:ok, t()} | {:error, [String.t()]}
  def validate(%__MODULE__{} = config) do
    errors =
      []
      |> validate_max_iterations(config)
      |> validate_threshold(config, :convergence_threshold)
      |> validate_threshold(config, :coherence_threshold)
      |> validate_threshold(config, :evidence_threshold)
      |> validate_timeout(config)

    case errors do
      [] -> {:ok, config}
      _ -> {:error, Enum.reverse(errors)}
    end
  end

  @doc """
  Merge config with another config or keyword list.

  ## Examples

      iex> base = %CNS.Config{max_iterations: 5}
      iex> merged = CNS.Config.merge(base, max_iterations: 10)
      iex> merged.max_iterations
      10
  """
  @spec merge(t(), keyword() | t()) :: t()
  def merge(%__MODULE__{} = config, opts) when is_list(opts) do
    struct(config, opts)
  end

  def merge(%__MODULE__{} = config, %__MODULE__{} = other) do
    Map.merge(config, other, fn
      _k, v1, v2 when is_map(v1) and is_map(v2) -> Map.merge(v1, v2)
      _k, _v1, v2 -> v2
    end)
  end

  @doc """
  Get default quality targets for CNS 3.0.

  Returns map with target thresholds:
  - Schema compliance >= 95%
  - Citation accuracy >= 95%
  - Mean entailment >= 0.50
  """
  @spec quality_targets() :: map()
  def quality_targets do
    %{
      schema_compliance: 0.95,
      citation_accuracy: 0.95,
      mean_entailment: 0.50,
      min_confidence: 0.85
    }
  end

  @doc """
  Convert config to JSON-serializable map.
  """
  @spec to_map(t()) :: map()
  def to_map(%__MODULE__{} = config) do
    Map.from_struct(config)
  end

  @doc """
  Create config from map.
  """
  @spec from_map(map()) :: {:ok, t()} | {:error, term()}
  def from_map(map) when is_map(map) do
    config = %__MODULE__{
      max_iterations: get_field_or(map, :max_iterations, 5),
      convergence_threshold: get_field_or(map, :convergence_threshold, 0.85),
      coherence_threshold: get_field_or(map, :coherence_threshold, 0.8),
      evidence_threshold: get_field_or(map, :evidence_threshold, 0.7),
      proposer: get_field_or(map, :proposer, %{}),
      antagonist: get_field_or(map, :antagonist, %{}),
      synthesizer: get_field_or(map, :synthesizer, %{}),
      telemetry_enabled: get_field_or(map, :telemetry_enabled, true),
      timeout: get_field_or(map, :timeout, 30_000),
      metadata: get_field_or(map, :metadata, %{})
    }

    {:ok, config}
  rescue
    e -> {:error, Exception.message(e)}
  end

  defp get_field_or(map, key, default) do
    Map.get(map, to_string(key)) || Map.get(map, key) || default
  end

  # Private validation functions

  defp validate_max_iterations(errors, %{max_iterations: n})
       when is_integer(n) and n > 0 do
    errors
  end

  defp validate_max_iterations(errors, _) do
    ["max_iterations must be a positive integer" | errors]
  end

  defp validate_threshold(errors, config, key) do
    value = Map.get(config, key)

    if is_number(value) and value >= 0.0 and value <= 1.0 do
      errors
    else
      ["#{key} must be a number between 0.0 and 1.0" | errors]
    end
  end

  defp validate_timeout(errors, %{timeout: t}) when is_integer(t) and t > 0 do
    errors
  end

  defp validate_timeout(errors, _) do
    ["timeout must be a positive integer" | errors]
  end
end

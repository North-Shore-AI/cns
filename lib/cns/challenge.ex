defmodule CNS.Challenge do
  @moduledoc """
  Challenge structure representing antagonist challenges to claims.

  Challenges identify issues with claims including contradictions,
  insufficient evidence, logical fallacies, and scope limitations.

  ## Fields

  * `:id` - Unique identifier
  * `:target_id` - ID of the SNO being challenged
  * `:challenge_type` - Type of challenge (:contradiction, :evidence_gap, :scope, :logical, :alternative)
  * `:description` - Description of the challenge
  * `:counter_evidence` - Evidence supporting the challenge
  * `:severity` - Severity level (:high, :medium, :low)
  * `:confidence` - Confidence in the challenge validity
  * `:resolution` - How the challenge was resolved (nil if pending)

  ## Examples

      iex> challenge = CNS.Challenge.new("sno-123", :contradiction, "Conflicts with X")
      iex> challenge.challenge_type
      :contradiction
  """

  alias CNS.Evidence

  @type challenge_type :: :contradiction | :evidence_gap | :scope | :logical | :alternative
  @type severity :: :high | :medium | :low
  @type resolution :: :accepted | :rejected | :partial | :pending

  @type t :: %__MODULE__{
          id: String.t(),
          target_id: String.t(),
          challenge_type: challenge_type(),
          description: String.t(),
          counter_evidence: [Evidence.t()],
          severity: severity(),
          confidence: float(),
          resolution: resolution(),
          metadata: map()
        }

  @enforce_keys [:target_id, :challenge_type, :description]
  defstruct [
    :id,
    :target_id,
    :challenge_type,
    :description,
    counter_evidence: [],
    severity: :medium,
    confidence: 0.5,
    resolution: :pending,
    metadata: %{}
  ]

  @doc """
  Create a new Challenge struct.

  ## Options

  * `:counter_evidence` - List of supporting evidence
  * `:severity` - :high, :medium, or :low
  * `:confidence` - Confidence score 0.0-1.0
  * `:id` - Custom ID

  ## Examples

      iex> challenge = CNS.Challenge.new("sno-1", :contradiction, "Contradicts evidence")
      iex> challenge.severity
      :medium

      iex> challenge = CNS.Challenge.new("sno-1", :evidence_gap, "Missing sources", severity: :high)
      iex> challenge.severity
      :high
  """
  @spec new(String.t(), challenge_type(), String.t(), keyword()) :: t()
  def new(target_id, challenge_type, description, opts \\ []) do
    %__MODULE__{
      id: Keyword.get(opts, :id, UUID.uuid4()),
      target_id: target_id,
      challenge_type: challenge_type,
      description: description,
      counter_evidence: Keyword.get(opts, :counter_evidence, []),
      severity: Keyword.get(opts, :severity, :medium),
      confidence: Keyword.get(opts, :confidence, 0.5),
      resolution: Keyword.get(opts, :resolution, :pending),
      metadata: Keyword.get(opts, :metadata, %{})
    }
  end

  @doc """
  Validate a challenge struct.

  ## Examples

      iex> challenge = CNS.Challenge.new("sno-1", :contradiction, "Test")
      iex> CNS.Challenge.validate(challenge)
      {:ok, challenge}
  """
  @spec validate(t()) :: {:ok, t()} | {:error, [String.t()]}
  def validate(%__MODULE__{} = challenge) do
    errors =
      []
      |> validate_target_id(challenge)
      |> validate_challenge_type(challenge)
      |> validate_description(challenge)
      |> validate_severity(challenge)
      |> validate_confidence(challenge)
      |> validate_resolution(challenge)

    case errors do
      [] -> {:ok, challenge}
      _ -> {:error, Enum.reverse(errors)}
    end
  end

  @doc """
  Convert challenge to JSON-serializable map.
  """
  @spec to_map(t()) :: map()
  def to_map(%__MODULE__{} = challenge) do
    %{
      id: challenge.id,
      target_id: challenge.target_id,
      challenge_type: Atom.to_string(challenge.challenge_type),
      description: challenge.description,
      counter_evidence: Enum.map(challenge.counter_evidence, &Evidence.to_map/1),
      severity: Atom.to_string(challenge.severity),
      confidence: challenge.confidence,
      resolution: Atom.to_string(challenge.resolution),
      metadata: challenge.metadata
    }
  end

  @doc """
  Create challenge from a map.
  """
  @spec from_map(map()) :: {:ok, t()} | {:error, term()}
  def from_map(map) when is_map(map) do
    challenge = %__MODULE__{
      id: get_field(map, :id) || UUID.uuid4(),
      target_id: get_field(map, :target_id),
      challenge_type: parse_atom(get_field(map, :challenge_type)),
      description: get_field(map, :description),
      counter_evidence: parse_counter_evidence(get_field(map, :counter_evidence)),
      severity: parse_atom(get_field(map, :severity) || :medium),
      confidence: parse_float(get_field(map, :confidence) || 0.5),
      resolution: parse_atom(get_field(map, :resolution) || :pending),
      metadata: get_field(map, :metadata) || %{}
    }

    {:ok, challenge}
  rescue
    e -> {:error, Exception.message(e)}
  end

  defp get_field(map, key), do: Map.get(map, to_string(key)) || Map.get(map, key)

  defp parse_counter_evidence(nil), do: []

  defp parse_counter_evidence(list) when is_list(list) do
    list
    |> Enum.map(fn e ->
      case Evidence.from_map(e) do
        {:ok, ev} -> ev
        _ -> nil
      end
    end)
    |> Enum.reject(&is_nil/1)
  end

  @doc """
  Calculate chirality score for the challenge.

  Higher scores indicate more significant challenges that need resolution.

  ## Examples

      iex> challenge = CNS.Challenge.new("sno-1", :contradiction, "Test", severity: :high, confidence: 0.9)
      iex> score = CNS.Challenge.chirality_score(challenge)
      iex> score > 0.5
      true
  """
  @spec chirality_score(t()) :: float()
  def chirality_score(%__MODULE__{} = challenge) do
    severity_weight =
      case challenge.severity do
        :high -> 1.0
        :medium -> 0.6
        :low -> 0.3
      end

    type_weight =
      case challenge.challenge_type do
        :contradiction -> 1.0
        :logical -> 0.9
        :evidence_gap -> 0.7
        :scope -> 0.5
        :alternative -> 0.4
      end

    Float.round(challenge.confidence * severity_weight * type_weight, 4)
  end

  @doc """
  Check if challenge is critical (high severity, high confidence).

  ## Examples

      iex> challenge = CNS.Challenge.new("sno-1", :contradiction, "Test", severity: :high, confidence: 0.8)
      iex> CNS.Challenge.critical?(challenge)
      true
  """
  @spec critical?(t()) :: boolean()
  def critical?(%__MODULE__{severity: :high, confidence: conf}) when conf >= 0.7, do: true
  def critical?(_), do: false

  @doc """
  Resolve a challenge with a given resolution status.

  ## Examples

      iex> challenge = CNS.Challenge.new("sno-1", :contradiction, "Test")
      iex> resolved = CNS.Challenge.resolve(challenge, :accepted)
      iex> resolved.resolution
      :accepted
  """
  @spec resolve(t(), resolution()) :: t()
  def resolve(%__MODULE__{} = challenge, resolution) do
    %{challenge | resolution: resolution}
  end

  @doc """
  Check if challenge is pending resolution.
  """
  @spec pending?(t()) :: boolean()
  def pending?(%__MODULE__{resolution: :pending}), do: true
  def pending?(_), do: false

  # Private functions

  defp validate_target_id(errors, %{target_id: id}) when is_binary(id) and byte_size(id) > 0 do
    errors
  end

  defp validate_target_id(errors, _), do: ["target_id must be a non-empty string" | errors]

  defp validate_challenge_type(errors, %{challenge_type: type})
       when type in [:contradiction, :evidence_gap, :scope, :logical, :alternative] do
    errors
  end

  defp validate_challenge_type(errors, _) do
    [
      "challenge_type must be :contradiction, :evidence_gap, :scope, :logical, or :alternative"
      | errors
    ]
  end

  defp validate_description(errors, %{description: desc})
       when is_binary(desc) and byte_size(desc) > 0 do
    errors
  end

  defp validate_description(errors, _), do: ["description must be a non-empty string" | errors]

  defp validate_severity(errors, %{severity: sev}) when sev in [:high, :medium, :low] do
    errors
  end

  defp validate_severity(errors, _), do: ["severity must be :high, :medium, or :low" | errors]

  defp validate_confidence(errors, %{confidence: c}) when is_number(c) and c >= 0.0 and c <= 1.0 do
    errors
  end

  defp validate_confidence(errors, _), do: ["confidence must be between 0.0 and 1.0" | errors]

  defp validate_resolution(errors, %{resolution: res})
       when res in [:accepted, :rejected, :partial, :pending] do
    errors
  end

  defp validate_resolution(errors, _) do
    ["resolution must be :accepted, :rejected, :partial, or :pending" | errors]
  end

  defp parse_atom(val) when is_atom(val), do: val
  defp parse_atom(val) when is_binary(val), do: String.to_existing_atom(val)

  defp parse_float(val) when is_float(val), do: val
  defp parse_float(val) when is_integer(val), do: val * 1.0
end

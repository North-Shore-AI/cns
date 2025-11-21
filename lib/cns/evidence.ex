defmodule CNS.Evidence do
  @moduledoc """
  Evidence structure for grounding claims to verifiable sources.

  Evidence records link claims to their supporting data with validity and relevance scores.

  ## Fields

  * `:id` - Unique identifier for the evidence
  * `:source` - Source reference (URL, citation, document ID)
  * `:content` - The actual evidence content/text
  * `:validity` - Score from 0.0 to 1.0 indicating source trustworthiness
  * `:relevance` - Score from 0.0 to 1.0 indicating relevance to the claim
  * `:retrieval_method` - How the evidence was obtained (:manual, :search, :citation, :inference)
  * `:timestamp` - When the evidence was retrieved/created

  ## Examples

      iex> evidence = CNS.Evidence.new("Stanford Study 2023", "Study found 20% improvement")
      iex> evidence.validity
      1.0

      iex> evidence = CNS.Evidence.new("Blog post", "Opinion", validity: 0.3)
      iex> evidence.validity
      0.3
  """

  @type t :: %__MODULE__{
          id: String.t(),
          source: String.t(),
          content: String.t(),
          validity: float(),
          relevance: float(),
          retrieval_method: atom(),
          timestamp: DateTime.t()
        }

  @enforce_keys [:source]
  defstruct [
    :id,
    :source,
    :content,
    validity: 1.0,
    relevance: 1.0,
    retrieval_method: :manual,
    timestamp: nil
  ]

  @doc """
  Create a new Evidence struct with defaults.

  ## Options

  * `:validity` - Source validity score (default: 1.0)
  * `:relevance` - Relevance score (default: 1.0)
  * `:retrieval_method` - How evidence was obtained (default: :manual)
  * `:id` - Custom ID (default: auto-generated UUID)

  ## Examples

      iex> evidence = CNS.Evidence.new("Paper XYZ", "Key finding")
      iex> evidence.source
      "Paper XYZ"

      iex> evidence = CNS.Evidence.new("URL", "Content", validity: 0.5, retrieval_method: :search)
      iex> {evidence.validity, evidence.retrieval_method}
      {0.5, :search}
  """
  @spec new(String.t(), String.t(), keyword()) :: t()
  def new(source, content \\ "", opts \\ []) do
    %__MODULE__{
      id: Keyword.get(opts, :id, generate_id()),
      source: source,
      content: content,
      validity: Keyword.get(opts, :validity, 1.0),
      relevance: Keyword.get(opts, :relevance, 1.0),
      retrieval_method: Keyword.get(opts, :retrieval_method, :manual),
      timestamp: Keyword.get(opts, :timestamp, DateTime.utc_now())
    }
  end

  @doc """
  Validate an evidence struct.

  Returns `{:ok, evidence}` if valid, `{:error, reasons}` otherwise.

  ## Examples

      iex> evidence = CNS.Evidence.new("Valid source", "Content")
      iex> CNS.Evidence.validate(evidence)
      {:ok, evidence}

      iex> evidence = %CNS.Evidence{source: "", validity: 1.5}
      iex> {:error, reasons} = CNS.Evidence.validate(evidence)
      iex> length(reasons) > 0
      true
  """
  @spec validate(t()) :: {:ok, t()} | {:error, [String.t()]}
  def validate(%__MODULE__{} = evidence) do
    errors =
      []
      |> validate_source(evidence)
      |> validate_validity(evidence)
      |> validate_relevance(evidence)
      |> validate_retrieval_method(evidence)

    case errors do
      [] -> {:ok, evidence}
      _ -> {:error, Enum.reverse(errors)}
    end
  end

  @doc """
  Convert evidence to JSON-serializable map.

  ## Examples

      iex> evidence = CNS.Evidence.new("Source", "Content")
      iex> map = CNS.Evidence.to_map(evidence)
      iex> map.source
      "Source"
  """
  @spec to_map(t()) :: map()
  def to_map(%__MODULE__{} = evidence) do
    %{
      id: evidence.id,
      source: evidence.source,
      content: evidence.content,
      validity: evidence.validity,
      relevance: evidence.relevance,
      retrieval_method: Atom.to_string(evidence.retrieval_method),
      timestamp: format_timestamp(evidence.timestamp)
    }
  end

  @doc """
  Create evidence from a map (e.g., from JSON).

  ## Examples

      iex> map = %{"source" => "Test", "content" => "Data", "validity" => 0.8}
      iex> {:ok, evidence} = CNS.Evidence.from_map(map)
      iex> evidence.source
      "Test"
  """
  @spec from_map(map()) :: {:ok, t()} | {:error, term()}
  def from_map(map) when is_map(map) do
    try do
      evidence = %__MODULE__{
        id: Map.get(map, "id") || Map.get(map, :id) || generate_id(),
        source: Map.get(map, "source") || Map.get(map, :source),
        content: Map.get(map, "content") || Map.get(map, :content) || "",
        validity: parse_float(Map.get(map, "validity") || Map.get(map, :validity) || 1.0),
        relevance: parse_float(Map.get(map, "relevance") || Map.get(map, :relevance) || 1.0),
        retrieval_method:
          parse_atom(Map.get(map, "retrieval_method") || Map.get(map, :retrieval_method) || :manual),
        timestamp: parse_timestamp(Map.get(map, "timestamp") || Map.get(map, :timestamp))
      }

      {:ok, evidence}
    rescue
      e -> {:error, Exception.message(e)}
    end
  end

  @doc """
  Calculate a combined score from validity and relevance.

  ## Examples

      iex> evidence = CNS.Evidence.new("Source", "Content", validity: 0.8, relevance: 0.6)
      iex> CNS.Evidence.score(evidence)
      0.48
  """
  @spec score(t()) :: float()
  def score(%__MODULE__{validity: validity, relevance: relevance}) do
    Float.round(validity * relevance, 4)
  end

  @doc """
  Check if evidence meets minimum quality thresholds.

  ## Examples

      iex> evidence = CNS.Evidence.new("Source", "Content", validity: 0.8, relevance: 0.7)
      iex> CNS.Evidence.meets_threshold?(evidence, 0.5)
      true

      iex> evidence = CNS.Evidence.new("Source", "Content", validity: 0.3, relevance: 0.3)
      iex> CNS.Evidence.meets_threshold?(evidence, 0.5)
      false
  """
  @spec meets_threshold?(t(), float()) :: boolean()
  def meets_threshold?(%__MODULE__{} = evidence, threshold) do
    score(evidence) >= threshold
  end

  # Private functions

  defp generate_id do
    UUID.uuid4()
  end

  defp validate_source(errors, %{source: source})
       when is_binary(source) and byte_size(source) > 0 do
    errors
  end

  defp validate_source(errors, _), do: ["source must be a non-empty string" | errors]

  defp validate_validity(errors, %{validity: v}) when is_number(v) and v >= 0.0 and v <= 1.0 do
    errors
  end

  defp validate_validity(errors, _), do: ["validity must be a number between 0.0 and 1.0" | errors]

  defp validate_relevance(errors, %{relevance: r}) when is_number(r) and r >= 0.0 and r <= 1.0 do
    errors
  end

  defp validate_relevance(errors, _),
    do: ["relevance must be a number between 0.0 and 1.0" | errors]

  defp validate_retrieval_method(errors, %{retrieval_method: method})
       when method in [:manual, :search, :citation, :inference] do
    errors
  end

  defp validate_retrieval_method(errors, _) do
    ["retrieval_method must be one of :manual, :search, :citation, :inference" | errors]
  end

  defp format_timestamp(nil), do: nil
  defp format_timestamp(%DateTime{} = dt), do: DateTime.to_iso8601(dt)

  defp parse_timestamp(nil), do: DateTime.utc_now()
  defp parse_timestamp(%DateTime{} = dt), do: dt

  defp parse_timestamp(str) when is_binary(str) do
    case DateTime.from_iso8601(str) do
      {:ok, dt, _} -> dt
      _ -> DateTime.utc_now()
    end
  end

  defp parse_float(val) when is_float(val), do: val
  defp parse_float(val) when is_integer(val), do: val * 1.0
  defp parse_float(val) when is_binary(val), do: String.to_float(val)

  defp parse_atom(val) when is_atom(val), do: val
  defp parse_atom(val) when is_binary(val), do: String.to_existing_atom(val)
end

defmodule CNS.SNO do
  @moduledoc """
  Structured Narrative Object (SNO) - the core data structure for claims in CNS.

  SNOs capture claims with their evidence, confidence scores, provenance chains,
  and synthesis history. They are the fundamental unit of knowledge in the
  dialectical reasoning process.

  ## Fields

  * `:id` - Unique identifier
  * `:claim` - The claim text
  * `:evidence` - List of supporting evidence
  * `:confidence` - Score from 0.0 to 1.0
  * `:provenance` - Record of how the claim was derived
  * `:metadata` - Additional arbitrary metadata
  * `:children` - Child SNOs (for hierarchical claims)
  * `:synthesis_history` - Record of synthesis operations

  ## Examples

      iex> sno = CNS.SNO.new("Coffee improves alertness")
      iex> sno.claim
      "Coffee improves alertness"

      iex> evidence = [CNS.Evidence.new("Study 2023", "Findings")]
      iex> sno = CNS.SNO.new("Claim", evidence: evidence, confidence: 0.9)
      iex> sno.confidence
      0.9
  """

  alias CNS.{Evidence, Provenance}

  @type t :: %__MODULE__{
          id: String.t(),
          claim: String.t(),
          evidence: [Evidence.t()],
          confidence: float(),
          provenance: Provenance.t() | nil,
          metadata: map(),
          children: [t()],
          synthesis_history: [map()]
        }

  @enforce_keys [:claim]
  defstruct [
    :id,
    :claim,
    evidence: [],
    confidence: 0.5,
    provenance: nil,
    metadata: %{},
    children: [],
    synthesis_history: []
  ]

  @doc """
  Create a new SNO with defaults.

  ## Options

  * `:evidence` - List of Evidence structs (default: [])
  * `:confidence` - Confidence score 0.0-1.0 (default: 0.5)
  * `:provenance` - Provenance struct (default: nil)
  * `:metadata` - Additional metadata map (default: %{})
  * `:id` - Custom ID (default: auto-generated UUID)

  ## Examples

      iex> sno = CNS.SNO.new("Test claim")
      iex> sno.confidence
      0.5

      iex> sno = CNS.SNO.new("High confidence", confidence: 0.95)
      iex> sno.confidence
      0.95
  """
  @spec new(String.t(), keyword()) :: t()
  def new(claim, opts \\ []) do
    %__MODULE__{
      id: Keyword.get(opts, :id, UUID.uuid4()),
      claim: claim,
      evidence: Keyword.get(opts, :evidence, []),
      confidence: Keyword.get(opts, :confidence, 0.5),
      provenance: Keyword.get(opts, :provenance),
      metadata: Keyword.get(opts, :metadata, %{}),
      children: Keyword.get(opts, :children, []),
      synthesis_history: Keyword.get(opts, :synthesis_history, [])
    }
  end

  @doc """
  Validate an SNO struct.

  Returns `{:ok, sno}` if valid, `{:error, reasons}` otherwise.

  ## Examples

      iex> sno = CNS.SNO.new("Valid claim")
      iex> CNS.SNO.validate(sno)
      {:ok, sno}

      iex> sno = %CNS.SNO{claim: "", confidence: 1.5}
      iex> {:error, reasons} = CNS.SNO.validate(sno)
      iex> length(reasons) > 0
      true
  """
  @spec validate(t()) :: {:ok, t()} | {:error, [String.t()]}
  def validate(%__MODULE__{} = sno) do
    errors =
      []
      |> validate_claim(sno)
      |> validate_confidence(sno)
      |> validate_evidence(sno)

    case errors do
      [] -> {:ok, sno}
      _ -> {:error, Enum.reverse(errors)}
    end
  end

  @doc """
  Convert SNO to JSON-serializable map.

  ## Examples

      iex> sno = CNS.SNO.new("Test claim", confidence: 0.8)
      iex> map = CNS.SNO.to_map(sno)
      iex> map.claim
      "Test claim"
  """
  @spec to_map(t()) :: map()
  def to_map(%__MODULE__{} = sno) do
    %{
      id: sno.id,
      claim: sno.claim,
      evidence: Enum.map(sno.evidence, &Evidence.to_map/1),
      confidence: sno.confidence,
      provenance: if(sno.provenance, do: Provenance.to_map(sno.provenance), else: nil),
      metadata: sno.metadata,
      children: Enum.map(sno.children, &to_map/1),
      synthesis_history: sno.synthesis_history
    }
  end

  @doc """
  Encode SNO to JSON string.

  ## Examples

      iex> sno = CNS.SNO.new("Test")
      iex> {:ok, json} = CNS.SNO.to_json(sno)
      iex> is_binary(json)
      true
  """
  @spec to_json(t()) :: {:ok, String.t()} | {:error, term()}
  def to_json(%__MODULE__{} = sno) do
    sno
    |> to_map()
    |> Jason.encode()
  end

  @doc """
  Create SNO from a map (e.g., from JSON).

  ## Examples

      iex> map = %{"claim" => "Test", "confidence" => 0.7}
      iex> {:ok, sno} = CNS.SNO.from_map(map)
      iex> sno.claim
      "Test"
  """
  @spec from_map(map()) :: {:ok, t()} | {:error, term()}
  def from_map(map) when is_map(map) do
    try do
      evidence =
        (Map.get(map, "evidence") || Map.get(map, :evidence) || [])
        |> Enum.map(fn e ->
          case Evidence.from_map(e) do
            {:ok, ev} -> ev
            _ -> nil
          end
        end)
        |> Enum.reject(&is_nil/1)

      provenance =
        case Map.get(map, "provenance") || Map.get(map, :provenance) do
          nil ->
            nil

          p ->
            case Provenance.from_map(p) do
              {:ok, prov} -> prov
              _ -> nil
            end
        end

      children =
        (Map.get(map, "children") || Map.get(map, :children) || [])
        |> Enum.map(fn c ->
          case from_map(c) do
            {:ok, child} -> child
            _ -> nil
          end
        end)
        |> Enum.reject(&is_nil/1)

      sno = %__MODULE__{
        id: Map.get(map, "id") || Map.get(map, :id) || UUID.uuid4(),
        claim: Map.get(map, "claim") || Map.get(map, :claim),
        evidence: evidence,
        confidence: parse_float(Map.get(map, "confidence") || Map.get(map, :confidence) || 0.5),
        provenance: provenance,
        metadata: Map.get(map, "metadata") || Map.get(map, :metadata) || %{},
        children: children,
        synthesis_history:
          Map.get(map, "synthesis_history") || Map.get(map, :synthesis_history) || []
      }

      {:ok, sno}
    rescue
      e -> {:error, Exception.message(e)}
    end
  end

  @doc """
  Parse SNO from JSON string.

  ## Examples

      iex> {:ok, sno} = CNS.SNO.from_json(~s({"claim": "Test", "confidence": 0.8}))
      iex> sno.claim
      "Test"
  """
  @spec from_json(String.t()) :: {:ok, t()} | {:error, term()}
  def from_json(json) when is_binary(json) do
    with {:ok, map} <- Jason.decode(json),
         {:ok, sno} <- from_map(map) do
      {:ok, sno}
    end
  end

  @doc """
  Add evidence to an SNO.

  ## Examples

      iex> sno = CNS.SNO.new("Claim")
      iex> evidence = CNS.Evidence.new("Source", "Content")
      iex> updated = CNS.SNO.add_evidence(sno, evidence)
      iex> length(updated.evidence)
      1
  """
  @spec add_evidence(t(), Evidence.t()) :: t()
  def add_evidence(%__MODULE__{} = sno, %Evidence{} = evidence) do
    %{sno | evidence: sno.evidence ++ [evidence]}
  end

  @doc """
  Update confidence score.

  ## Examples

      iex> sno = CNS.SNO.new("Claim", confidence: 0.5)
      iex> updated = CNS.SNO.update_confidence(sno, 0.8)
      iex> updated.confidence
      0.8
  """
  @spec update_confidence(t(), float()) :: t()
  def update_confidence(%__MODULE__{} = sno, confidence)
      when confidence >= 0.0 and confidence <= 1.0 do
    %{sno | confidence: confidence}
  end

  @doc """
  Calculate average evidence validity score.

  ## Examples

      iex> sno = CNS.SNO.new("Claim")
      iex> CNS.SNO.evidence_score(sno)
      0.0

      iex> e1 = CNS.Evidence.new("S1", "C1", validity: 0.8)
      iex> e2 = CNS.Evidence.new("S2", "C2", validity: 0.6)
      iex> sno = CNS.SNO.new("Claim", evidence: [e1, e2])
      iex> CNS.SNO.evidence_score(sno)
      0.7
  """
  @spec evidence_score(t()) :: float()
  def evidence_score(%__MODULE__{evidence: []}) do
    0.0
  end

  def evidence_score(%__MODULE__{evidence: evidence}) do
    scores = Enum.map(evidence, & &1.validity)
    Float.round(Enum.sum(scores) / length(scores), 4)
  end

  @doc """
  Calculate combined quality score (confidence * evidence score).

  ## Examples

      iex> e = CNS.Evidence.new("S", "C", validity: 0.8)
      iex> sno = CNS.SNO.new("Claim", evidence: [e], confidence: 0.9)
      iex> CNS.SNO.quality_score(sno)
      0.72
  """
  @spec quality_score(t()) :: float()
  def quality_score(%__MODULE__{evidence: []} = sno) do
    Float.round(sno.confidence * 0.5, 4)
  end

  def quality_score(%__MODULE__{} = sno) do
    Float.round(sno.confidence * evidence_score(sno), 4)
  end

  @doc """
  Check if SNO meets quality threshold.

  ## Examples

      iex> sno = CNS.SNO.new("Claim", confidence: 0.9)
      iex> CNS.SNO.meets_threshold?(sno, 0.4)
      true
  """
  @spec meets_threshold?(t(), float()) :: boolean()
  def meets_threshold?(%__MODULE__{} = sno, threshold) do
    quality_score(sno) >= threshold
  end

  @doc """
  Get word count of claim.
  """
  @spec word_count(t()) :: non_neg_integer()
  def word_count(%__MODULE__{claim: claim}) do
    claim
    |> String.split(~r/\s+/, trim: true)
    |> length()
  end

  # Private functions

  defp validate_claim(errors, %{claim: claim}) when is_binary(claim) and byte_size(claim) > 0 do
    errors
  end

  defp validate_claim(errors, _), do: ["claim must be a non-empty string" | errors]

  defp validate_confidence(errors, %{confidence: c}) when is_number(c) and c >= 0.0 and c <= 1.0 do
    errors
  end

  defp validate_confidence(errors, _),
    do: ["confidence must be a number between 0.0 and 1.0" | errors]

  defp validate_evidence(errors, %{evidence: evidence}) when is_list(evidence) do
    if Enum.all?(evidence, &match?(%Evidence{}, &1)) do
      errors
    else
      ["all evidence items must be Evidence structs" | errors]
    end
  end

  defp validate_evidence(errors, _), do: ["evidence must be a list" | errors]

  defp parse_float(val) when is_float(val), do: val
  defp parse_float(val) when is_integer(val), do: val * 1.0
  defp parse_float(val) when is_binary(val), do: String.to_float(val)
end

defmodule CNS.Provenance do
  @moduledoc """
  Provenance tracking for claims - records how a claim was derived.

  Provenance enables full traceability of the dialectical process, tracking
  the origin, parent claims, and transformations applied.

  ## Fields

  * `:origin` - Source agent (:proposer, :antagonist, :synthesizer, :external)
  * `:parent_ids` - IDs of parent claims this was derived from
  * `:transformation` - Description of the transformation applied
  * `:model_id` - Model used for generation
  * `:timestamp` - When the claim was created
  * `:iteration` - Pipeline iteration number

  ## Examples

      iex> prov = CNS.Provenance.new(:proposer)
      iex> prov.origin
      :proposer

      iex> prov = CNS.Provenance.new(:synthesizer, parent_ids: ["id1", "id2"])
      iex> length(prov.parent_ids)
      2
  """

  @type origin :: :proposer | :antagonist | :synthesizer | :external
  @type t :: %__MODULE__{
          origin: origin(),
          parent_ids: [String.t()],
          transformation: String.t(),
          model_id: String.t() | nil,
          timestamp: DateTime.t(),
          iteration: non_neg_integer()
        }

  defstruct [
    :origin,
    parent_ids: [],
    transformation: "",
    model_id: nil,
    timestamp: nil,
    iteration: 0
  ]

  @doc """
  Create a new Provenance struct.

  ## Options

  * `:parent_ids` - List of parent claim IDs
  * `:transformation` - Description of transformation
  * `:model_id` - Model identifier
  * `:iteration` - Pipeline iteration number

  ## Examples

      iex> prov = CNS.Provenance.new(:proposer, transformation: "initial extraction")
      iex> prov.transformation
      "initial extraction"
  """
  @spec new(origin(), keyword()) :: t()
  def new(origin, opts \\ []) do
    %__MODULE__{
      origin: origin,
      parent_ids: Keyword.get(opts, :parent_ids, []),
      transformation: Keyword.get(opts, :transformation, ""),
      model_id: Keyword.get(opts, :model_id),
      timestamp: Keyword.get(opts, :timestamp, DateTime.utc_now()),
      iteration: Keyword.get(opts, :iteration, 0)
    }
  end

  @doc """
  Validate a provenance struct.

  ## Examples

      iex> prov = CNS.Provenance.new(:proposer)
      iex> CNS.Provenance.validate(prov)
      {:ok, prov}

      iex> prov = %CNS.Provenance{origin: :invalid}
      iex> {:error, _} = CNS.Provenance.validate(prov)
  """
  @spec validate(t()) :: {:ok, t()} | {:error, [String.t()]}
  def validate(%__MODULE__{} = prov) do
    errors =
      []
      |> validate_origin(prov)
      |> validate_parent_ids(prov)
      |> validate_iteration(prov)

    case errors do
      [] -> {:ok, prov}
      _ -> {:error, Enum.reverse(errors)}
    end
  end

  @doc """
  Convert provenance to JSON-serializable map.
  """
  @spec to_map(t()) :: map()
  def to_map(%__MODULE__{} = prov) do
    %{
      origin: Atom.to_string(prov.origin),
      parent_ids: prov.parent_ids,
      transformation: prov.transformation,
      model_id: prov.model_id,
      timestamp: format_timestamp(prov.timestamp),
      iteration: prov.iteration
    }
  end

  @doc """
  Create provenance from a map.
  """
  @spec from_map(map()) :: {:ok, t()} | {:error, term()}
  def from_map(map) when is_map(map) do
    try do
      prov = %__MODULE__{
        origin: parse_origin(Map.get(map, "origin") || Map.get(map, :origin)),
        parent_ids: Map.get(map, "parent_ids") || Map.get(map, :parent_ids) || [],
        transformation: Map.get(map, "transformation") || Map.get(map, :transformation) || "",
        model_id: Map.get(map, "model_id") || Map.get(map, :model_id),
        timestamp: parse_timestamp(Map.get(map, "timestamp") || Map.get(map, :timestamp)),
        iteration: Map.get(map, "iteration") || Map.get(map, :iteration) || 0
      }

      {:ok, prov}
    rescue
      e -> {:error, Exception.message(e)}
    end
  end

  @doc """
  Check if provenance indicates a synthesis operation.

  ## Examples

      iex> prov = CNS.Provenance.new(:synthesizer)
      iex> CNS.Provenance.is_synthesis?(prov)
      true

      iex> prov = CNS.Provenance.new(:proposer)
      iex> CNS.Provenance.is_synthesis?(prov)
      false
  """
  @spec is_synthesis?(t()) :: boolean()
  def is_synthesis?(%__MODULE__{origin: :synthesizer}), do: true
  def is_synthesis?(_), do: false

  @doc """
  Get the depth of derivation (number of parents in chain).
  """
  @spec depth(t()) :: non_neg_integer()
  def depth(%__MODULE__{parent_ids: parent_ids}), do: length(parent_ids)

  # Private functions

  defp validate_origin(errors, %{origin: origin})
       when origin in [:proposer, :antagonist, :synthesizer, :external] do
    errors
  end

  defp validate_origin(errors, _) do
    ["origin must be :proposer, :antagonist, :synthesizer, or :external" | errors]
  end

  defp validate_parent_ids(errors, %{parent_ids: ids}) when is_list(ids) do
    if Enum.all?(ids, &is_binary/1) do
      errors
    else
      ["all parent_ids must be strings" | errors]
    end
  end

  defp validate_parent_ids(errors, _), do: ["parent_ids must be a list" | errors]

  defp validate_iteration(errors, %{iteration: i}) when is_integer(i) and i >= 0 do
    errors
  end

  defp validate_iteration(errors, _), do: ["iteration must be a non-negative integer" | errors]

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

  defp parse_origin(origin) when is_atom(origin), do: origin
  defp parse_origin("proposer"), do: :proposer
  defp parse_origin("antagonist"), do: :antagonist
  defp parse_origin("synthesizer"), do: :synthesizer
  defp parse_origin("external"), do: :external
end

defmodule CNS.Pipeline.Schema do
  @moduledoc """
  Data schemas for the CNS training data pipeline.
  """

  defmodule TrainingExample do
    @moduledoc "Training example with prompt-completion pair"

    @type t :: %__MODULE__{
            prompt: String.t(),
            completion: String.t(),
            metadata: map()
          }

    @enforce_keys [:prompt, :completion, :metadata]
    defstruct [:prompt, :completion, :metadata]

    @doc """
    Convert training example to JSON string.
    """
    @spec to_json(t()) :: String.t()
    def to_json(%__MODULE__{} = example) do
      Jason.encode!(%{
        prompt: example.prompt,
        completion: example.completion,
        metadata: example.metadata
      })
    end

    @doc """
    Parse training example from JSON string.
    """
    @spec from_json(String.t()) :: {:ok, t()} | {:error, term()}
    def from_json(json) do
      case Jason.decode(json) do
        {:ok, data} ->
          example = %__MODULE__{
            prompt: data["prompt"],
            completion: data["completion"],
            metadata: data["metadata"] || %{}
          }

          {:ok, example}

        {:error, reason} ->
          {:error, reason}
      end
    end
  end

  defmodule ClaimEntry do
    @moduledoc "Intermediate claim entry during conversion"

    @type t :: %__MODULE__{
            id: String.t(),
            text: String.t(),
            evidence_ids: [String.t()],
            label: String.t() | nil
          }

    @enforce_keys [:id, :text]
    defstruct [:id, :text, evidence_ids: [], label: nil]
  end

  defmodule Lineage do
    @moduledoc "Data lineage tracking for reproducibility"

    @type t :: %__MODULE__{
            source_file: String.t(),
            timestamp: DateTime.t(),
            transformations: [String.t()],
            hash: String.t()
          }

    @enforce_keys [:source_file, :timestamp, :transformations, :hash]
    defstruct [:source_file, :timestamp, :transformations, :hash]

    @doc """
    Add a transformation to the lineage history.
    """
    @spec add_transformation(t(), String.t()) :: t()
    def add_transformation(%__MODULE__{} = lineage, transformation) do
      %{lineage | transformations: lineage.transformations ++ [transformation]}
    end

    @doc """
    Create new lineage for a source file.
    """
    @spec new(String.t()) :: t()
    def new(source_file) do
      %__MODULE__{
        source_file: source_file,
        timestamp: DateTime.utc_now(),
        transformations: [],
        hash: compute_hash(source_file)
      }
    end

    defp compute_hash(file_path) do
      if File.exists?(file_path) do
        file_path
        |> File.read!()
        |> then(&:crypto.hash(:sha256, &1))
        |> Base.encode16(case: :lower)
        |> String.slice(0, 16)
      else
        ""
      end
    end
  end
end

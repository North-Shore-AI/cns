defmodule CNS.Validation.Citation do
  @moduledoc """
  Citation validation for CNS claims.

  Validates that document references in claims exist in the corpus
  and detects potential hallucination (fabricated citations).
  """

  @doc """
  Extract document IDs from text containing CLAIM format.

  ## Examples

      iex> text = "CLAIM[c1] (Document 12345): Some claim"
      iex> CNS.Validation.Citation.extract_document_ids(text)
      ["12345"]
  """
  @spec extract_document_ids(String.t()) :: [String.t()]
  def extract_document_ids(text) do
    ~r/\(Document\s+(\d+)\)/i
    |> Regex.scan(text)
    |> Enum.map(fn [_, id] -> id end)
    |> Enum.uniq()
  end

  @doc """
  Validate that all cited documents exist in the corpus.

  ## Parameters
    - claim_text: Text containing CLAIM format with document references
    - corpus: MapSet of valid document IDs

  ## Returns
    Map with :valid, :cited, :missing fields

  ## Examples

      iex> corpus = MapSet.new(["12345"])
      iex> result = CNS.Validation.Citation.validate("CLAIM[c1] (Document 12345): Test", corpus)
      iex> result.valid
      true
  """
  @spec validate(String.t(), MapSet.t()) :: map()
  def validate(claim_text, corpus) do
    cited = extract_document_ids(claim_text)

    missing =
      cited
      |> Enum.reject(&MapSet.member?(corpus, &1))

    %{
      valid: Enum.empty?(missing),
      cited: cited,
      missing: missing,
      total_cited: length(cited),
      total_valid: length(cited) - length(missing)
    }
  end

  @doc """
  Check for potential hallucination (fabricated citations).

  ## Parameters
    - claim_text: Text containing CLAIM format
    - corpus: MapSet of valid document IDs

  ## Returns
    Map with :hallucination_detected, :fabricated_ids fields

  ## Examples

      iex> corpus = MapSet.new(["12345"])
      iex> result = CNS.Validation.Citation.check_hallucination("CLAIM[c1] (Document 99999): Test", corpus)
      iex> result.hallucination_detected
      true
  """
  @spec check_hallucination(String.t(), MapSet.t()) :: map()
  def check_hallucination(claim_text, corpus) do
    validation = validate(claim_text, corpus)

    %{
      hallucination_detected: not Enum.empty?(validation.missing),
      fabricated_ids: validation.missing,
      confidence: calculate_confidence(validation)
    }
  end

  @doc """
  Validate a batch of claims against corpus.

  ## Returns
    Map with aggregate statistics
  """
  @spec validate_batch([String.t()], MapSet.t()) :: map()
  def validate_batch(claims, corpus) do
    results = Enum.map(claims, &validate(&1, corpus))

    total = length(results)
    valid_count = Enum.count(results, & &1.valid)

    %{
      total: total,
      valid: valid_count,
      invalid: total - valid_count,
      accuracy: if(total > 0, do: valid_count / total, else: 0.0),
      results: results
    }
  end

  # Private functions

  defp calculate_confidence(validation) do
    if validation.total_cited == 0 do
      # No citations = no hallucination risk
      1.0
    else
      validation.total_valid / validation.total_cited
    end
  end
end

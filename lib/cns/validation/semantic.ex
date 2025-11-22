defmodule CNS.Validation.Semantic do
  @moduledoc """
  4-stage semantic validation pipeline for CNS claim extraction.

  Stages:
  1. Citation Accuracy - Hard gate for document ID verification
  2. Entailment Scoring - NLI-based evidence entailment
  3. Semantic Similarity - Text similarity scoring
  4. Paraphrase Tolerance - Accept valid rephrasings
  """

  defmodule ValidationResult do
    @moduledoc "Result of semantic validation pipeline"

    @type t :: %__MODULE__{
            citation_valid: boolean(),
            cited_ids: MapSet.t(String.t()),
            missing_ids: MapSet.t(String.t()),
            entailment_score: float(),
            entailment_pass: boolean(),
            semantic_similarity: float(),
            similarity_pass: boolean(),
            paraphrase_accepted: boolean(),
            overall_pass: boolean(),
            schema_valid: boolean(),
            schema_errors: [String.t()]
          }

    @enforce_keys [
      :citation_valid,
      :cited_ids,
      :missing_ids,
      :entailment_score,
      :entailment_pass,
      :semantic_similarity,
      :similarity_pass,
      :paraphrase_accepted,
      :overall_pass,
      :schema_valid,
      :schema_errors
    ]
    defstruct @enforce_keys
  end

  defmodule Config do
    @moduledoc "Configuration for semantic validation thresholds"

    @type t :: %__MODULE__{
            entailment_threshold: float(),
            similarity_threshold: float()
          }

    defstruct entailment_threshold: 0.75,
              similarity_threshold: 0.7
  end

  @doc """
  Extract document IDs from text using various patterns.

  ## Examples

      iex> CNS.Validation.Semantic.extract_document_ids("Document 123 and [DocID: abc]")
      MapSet.new(["123", "abc"])
  """
  @spec extract_document_ids(String.t()) :: MapSet.t(String.t())
  def extract_document_ids(text) do
    patterns = [
      # Document 12345
      ~r/Document\s+(\d+)/i,
      # (Document 12345)
      ~r/\(Document\s+(\d+)\)/i,
      # [DocID: abc123]
      ~r/\[DocID:\s*([^\]]+)\]/i,
      # [ref:xyz789]
      ~r/\[ref:([^\]]+)\]/i,
      # [12345:0]
      ~r/\[(\d+):\d+\]/
    ]

    patterns
    |> Enum.flat_map(fn pattern ->
      Regex.scan(pattern, text)
      |> Enum.map(fn
        [_, id] -> String.trim(id)
        _ -> nil
      end)
    end)
    |> Enum.reject(&is_nil/1)
    |> MapSet.new()
  end

  @doc """
  Validate citations in text against corpus and gold evidence.

  Returns {valid?, cited_ids, missing_ids}.

  ## Examples

      iex> text = "Document 123"
      iex> corpus = %{"123" => %{}}
      iex> gold = MapSet.new(["123"])
      iex> CNS.Validation.Semantic.validate_citations(text, corpus, gold)
      {true, MapSet.new(["123"]), MapSet.new()}
  """
  @spec validate_citations(String.t(), map(), MapSet.t(String.t())) ::
          {boolean(), MapSet.t(String.t()), MapSet.t(String.t())}
  def validate_citations(text, corpus, gold_ids) do
    cited_ids = extract_document_ids(text)

    # Filter to only IDs that exist in corpus
    valid_ids = MapSet.filter(cited_ids, &Map.has_key?(corpus, &1))
    missing_ids = MapSet.difference(gold_ids, valid_ids)

    citation_valid = MapSet.size(missing_ids) == 0

    {citation_valid, valid_ids, missing_ids}
  end

  @doc """
  Compute text similarity using word overlap (Jaccard-like).

  This is a simple similarity metric. For production use,
  integrate with embedding-based similarity.

  ## Examples

      iex> CNS.Validation.Semantic.compute_similarity("hello world", "hello world")
      1.0
  """
  @spec compute_similarity(String.t(), String.t()) :: float()
  def compute_similarity(text_a, text_b) do
    words_a = tokenize(text_a)
    words_b = tokenize(text_b)

    if MapSet.size(words_a) == 0 and MapSet.size(words_b) == 0 do
      0.0
    else
      intersection = MapSet.intersection(words_a, words_b) |> MapSet.size()
      union = MapSet.union(words_a, words_b) |> MapSet.size()

      if union == 0, do: 0.0, else: intersection / union
    end
  end

  defp tokenize(text) do
    text
    |> String.downcase()
    |> String.replace(~r/[^\w\s]/, "")
    |> String.split(~r/\s+/, trim: true)
    |> MapSet.new()
  end

  @doc """
  Validate a claim through the 4-stage pipeline.

  ## Parameters
    - config: Validation thresholds
    - generated_claim: The claim text to validate
    - gold_claim: Expected claim text
    - generated_full_output: Full model output (for citation extraction)
    - corpus: Map of doc_id => document data
    - gold_evidence_ids: Set of expected evidence document IDs

  ## Returns
    ValidationResult with stage-by-stage results
  """
  @spec validate_claim(
          Config.t(),
          String.t(),
          String.t(),
          String.t(),
          map(),
          MapSet.t(String.t())
        ) :: ValidationResult.t()
  def validate_claim(
        config,
        generated_claim,
        gold_claim,
        generated_full_output,
        corpus,
        gold_evidence_ids
      ) do
    # Stage 1: Citation Accuracy
    {citation_valid, cited_ids, missing_ids} =
      validate_citations(generated_full_output, corpus, gold_evidence_ids)

    # Short circuit if citations fail (hard gate)
    if not citation_valid do
      failed_result(citation_valid, cited_ids, missing_ids)
    else
      # Stage 2: Entailment (placeholder - returns high score for now)
      # In production, integrate with NLI model
      entailment_score = compute_entailment_placeholder(generated_claim, corpus, cited_ids)
      entailment_pass = entailment_score >= config.entailment_threshold

      # Stage 3: Semantic Similarity
      similarity = compute_similarity(generated_claim, gold_claim)
      similarity_pass = similarity >= config.similarity_threshold

      # Stage 4: Paraphrase Tolerance
      paraphrase_accepted = entailment_pass or similarity_pass

      # Overall
      overall_pass = citation_valid and entailment_pass and paraphrase_accepted

      %ValidationResult{
        citation_valid: citation_valid,
        cited_ids: cited_ids,
        missing_ids: missing_ids,
        entailment_score: entailment_score,
        entailment_pass: entailment_pass,
        semantic_similarity: similarity,
        similarity_pass: similarity_pass,
        paraphrase_accepted: paraphrase_accepted,
        overall_pass: overall_pass,
        schema_valid: true,
        schema_errors: []
      }
    end
  end

  @doc """
  Create a failed validation result.

  Used when citation validation fails (hard gate).
  """
  @spec failed_result(boolean(), MapSet.t(String.t()), MapSet.t(String.t())) :: ValidationResult.t()
  def failed_result(citation_valid, cited_ids, missing_ids) do
    %ValidationResult{
      citation_valid: citation_valid,
      cited_ids: cited_ids,
      missing_ids: missing_ids,
      entailment_score: 0.0,
      entailment_pass: false,
      semantic_similarity: 0.0,
      similarity_pass: false,
      paraphrase_accepted: false,
      overall_pass: false,
      schema_valid: true,
      schema_errors: []
    }
  end

  # Placeholder for entailment scoring
  # In production, integrate with Bumblebee NLI model
  defp compute_entailment_placeholder(claim, corpus, cited_ids) do
    evidence_text = get_evidence_text(cited_ids, corpus)

    if String.length(evidence_text) > 0 and String.length(claim) > 0 do
      # Simple heuristic: word overlap as proxy for entailment
      compute_similarity(claim, evidence_text)
    else
      0.0
    end
  end

  defp get_evidence_text(doc_ids, corpus) do
    doc_ids
    |> Enum.map(&Map.get(corpus, &1, %{}))
    |> Enum.map(&Map.get(&1, "text", ""))
    |> Enum.join(" ")
  end
end

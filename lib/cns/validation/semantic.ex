defmodule CNS.Validation.Semantic do
  @moduledoc """
  4-stage semantic validation pipeline for CNS claim extraction.

  Stages:
  1. Citation Accuracy - Hard gate for document ID verification
  2. Entailment Scoring - NLI-based evidence entailment (DeBERTa-v3)
  3. Semantic Similarity - Embedding-based text similarity (MiniLM)
  4. Paraphrase Tolerance - Accept valid rephrasings

  ## Model Integration

  When Bumblebee models are available (via `CNS.Validation.ModelLoader`),
  this module uses real transformer-based inference:

  - **Entailment**: DeBERTa-v3-base NLI model for entailment/neutral/contradiction
  - **Similarity**: all-MiniLM-L6-v2 for cosine similarity of sentence embeddings

  When models are not available, it falls back to word-overlap heuristics.

  ## Usage

      config = %CNS.Validation.Semantic.Config{}
      result = CNS.Validation.Semantic.validate_claim(
        config, generated_claim, gold_claim, full_output, corpus, gold_ids
      )
  """

  require Logger
  alias CNS.Validation.ModelLoader

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
  Compute semantic similarity between two texts.

  When embedding models are available, uses cosine similarity of
  sentence embeddings from all-MiniLM-L6-v2.

  Falls back to word overlap (Jaccard) when models aren't loaded.

  ## Examples

      iex> CNS.Validation.Semantic.compute_similarity("hello world", "hello world")
      1.0
  """
  @spec compute_similarity(String.t(), String.t()) :: float()
  def compute_similarity(text_a, text_b) do
    case compute_embedding_similarity(text_a, text_b) do
      {:ok, score} ->
        score

      {:error, _reason} ->
        # Fallback to word overlap
        compute_word_overlap_similarity(text_a, text_b)
    end
  end

  @doc """
  Compute embedding-based similarity using MiniLM model.

  Returns {:ok, score} or {:error, reason}.
  """
  @spec compute_embedding_similarity(String.t(), String.t()) :: {:ok, float()} | {:error, term()}
  def compute_embedding_similarity(text_a, text_b) do
    if String.length(text_a) == 0 or String.length(text_b) == 0 do
      {:ok, 0.0}
    else
      case ModelLoader.get_embedding_model() do
        {:ok, serving} ->
          # Get embeddings for both texts
          %{embedding: embedding_a} = Nx.Serving.run(serving, text_a)
          %{embedding: embedding_b} = Nx.Serving.run(serving, text_b)

          # Compute cosine similarity
          similarity = cosine_similarity(embedding_a, embedding_b)
          {:ok, max(0.0, min(1.0, similarity))}

        {:error, reason} ->
          {:error, reason}
      end
    end
  rescue
    e ->
      {:error, Exception.message(e)}
  end

  defp cosine_similarity(a, b) do
    # Flatten to 1D tensors
    a_flat = Nx.flatten(a)
    b_flat = Nx.flatten(b)

    dot = Nx.dot(a_flat, b_flat) |> Nx.to_number()
    norm_a = Nx.LinAlg.norm(a_flat) |> Nx.to_number()
    norm_b = Nx.LinAlg.norm(b_flat) |> Nx.to_number()

    if norm_a == 0 or norm_b == 0 do
      0.0
    else
      dot / (norm_a * norm_b)
    end
  end

  defp compute_word_overlap_similarity(text_a, text_b) do
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
    if citation_valid do
      # Stage 2: Entailment (NLI model or fallback)
      entailment_score = compute_entailment(generated_claim, corpus, cited_ids)
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
    else
      failed_result(citation_valid, cited_ids, missing_ids)
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

  @doc """
  Compute entailment score between evidence and claim.

  When NLI model is available, uses DeBERTa-v3-base to classify
  whether evidence entails the claim.

  Falls back to word overlap heuristic when model isn't loaded.

  Returns a score from 0.0 to 1.0 where higher means stronger entailment.
  """
  @spec compute_entailment(String.t(), map(), MapSet.t(String.t())) :: float()
  def compute_entailment(claim, corpus, cited_ids) do
    evidence_text = get_evidence_text(cited_ids, corpus)

    if String.length(evidence_text) == 0 or String.length(claim) == 0 do
      0.0
    else
      case compute_nli_entailment(evidence_text, claim) do
        {:ok, score} ->
          score

        {:error, _reason} ->
          # Fallback to word overlap
          compute_word_overlap_similarity(claim, evidence_text)
      end
    end
  end

  @doc """
  Compute NLI-based entailment using DeBERTa model.

  Args:
    - premise: The evidence text
    - hypothesis: The claim to verify

  Returns {:ok, entailment_probability} or {:error, reason}.

  The model outputs probabilities for [contradiction, neutral, entailment].
  We return the entailment probability.
  """
  @spec compute_nli_entailment(String.t(), String.t()) :: {:ok, float()} | {:error, term()}
  def compute_nli_entailment(premise, hypothesis) do
    case ModelLoader.get_nli_model() do
      {:ok, serving} ->
        # NLI model expects a pair of (premise, hypothesis)
        result = Nx.Serving.run(serving, {premise, hypothesis})

        # Extract entailment probability
        entailment_score = extract_entailment_score(result)
        {:ok, entailment_score}

      {:error, reason} ->
        {:error, reason}
    end
  rescue
    e ->
      {:error, Exception.message(e)}
  end

  defp extract_entailment_score(%{predictions: predictions}) do
    # Find the entailment label and its score
    entailment_pred =
      Enum.find(predictions, fn pred ->
        label = String.downcase(pred.label)
        String.contains?(label, "entailment")
      end)

    case entailment_pred do
      nil -> 0.0
      pred -> pred.score
    end
  end

  defp extract_entailment_score(_), do: 0.0

  defp get_evidence_text(doc_ids, corpus) do
    doc_ids
    |> Enum.map(&Map.get(corpus, &1, %{}))
    |> Enum.flat_map(fn doc ->
      # Support both "text" and "abstract" fields
      text = Map.get(doc, "text", "")
      abstract = Map.get(doc, "abstract", "")
      title = Map.get(doc, "title", "")

      abstract_text =
        case abstract do
          list when is_list(list) -> Enum.join(list, " ")
          str when is_binary(str) -> str
          _ -> ""
        end

      [title, abstract_text, text]
      |> Enum.reject(&(&1 == ""))
    end)
    |> Enum.join(" ")
  end
end

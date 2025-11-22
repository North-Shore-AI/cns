defmodule CNS.Pipeline.Converters do
  @moduledoc """
  Dataset conversion utilities for claim extraction training data.
  """

  alias CNS.Pipeline.Schema.TrainingExample

  @type corpus_map :: %{String.t() => %{sentences: [String.t()]}}

  @doc """
  Build extraction prompt with passage.
  """
  @spec build_prompt(String.t()) :: String.t()
  def build_prompt(passage) do
    """
    Extract all claims and relations from the following passage.
    Output format: CLAIM[c1]: main claim, CLAIM[c2..n]: evidence, RELATION: cx label cy

    Passage:
    #{passage}
    """
  end

  @doc """
  Build completion from claim text and evidence list.

  Evidence format: [{text, label, doc_ref}, ...]
  """
  @spec build_completion(String.t(), [{String.t(), String.t(), String.t()}]) :: String.t()
  def build_completion(claim_text, evidence) do
    claims = ["CLAIM[c1]: #{claim_text}"]

    {evidence_claims, relations} =
      evidence
      |> Enum.with_index(2)
      |> Enum.map(fn {{text, label, doc_ref}, idx} ->
        claim_line = "CLAIM[c#{idx}]: #{text} #{doc_ref}"
        relation = "RELATION: c#{idx} #{label} c1"
        {claim_line, relation}
      end)
      |> Enum.unzip()

    (claims ++ evidence_claims ++ relations) |> Enum.join("\n")
  end

  @doc """
  Normalize evidence label to lowercase standard form.
  """
  @spec normalize_label(String.t()) :: String.t()
  def normalize_label(label) do
    case String.upcase(label) do
      "SUPPORTS" -> "supports"
      "SUPPORT" -> "supports"
      "REFUTES" -> "refutes"
      "CONTRADICT" -> "refutes"
      other -> String.downcase(other)
    end
  end

  @doc """
  Parse a SciFact entry into a training example.
  """
  @spec parse_scifact_entry(map(), corpus_map()) :: TrainingExample.t()
  def parse_scifact_entry(entry, corpus) do
    claim_text = entry["claim"]
    evidence = gather_evidence(entry, corpus)
    passage = build_passage(entry, corpus)

    completion = build_completion(claim_text, evidence)
    prompt = build_prompt(passage)

    %TrainingExample{
      prompt: prompt,
      completion: completion,
      metadata: %{
        source: "scifact",
        claim_id: entry["id"]
      }
    }
  end

  @doc """
  Gather evidence sentences with labels from entry.

  Returns list of {text, normalized_label, doc_ref} tuples.
  """
  @spec gather_evidence(map(), corpus_map()) :: [{String.t(), String.t(), String.t()}]
  def gather_evidence(entry, corpus) do
    entry
    |> Map.get("evidence", %{})
    |> Enum.flat_map(fn {doc_id, evidence_sets} ->
      Enum.flat_map(evidence_sets, fn evidence_set ->
        label = Map.get(evidence_set, "label", "supports")
        sent_ids = Map.get(evidence_set, "sentences", [])

        Enum.map(sent_ids, fn sent_idx ->
          text = get_sentence(corpus, doc_id, sent_idx)
          {text, normalize_label(label), "[#{doc_id}:#{sent_idx}]"}
        end)
      end)
    end)
    |> Enum.reject(fn {text, _, _} -> text == "" end)
  end

  @doc """
  Check if entry has evidence.
  """
  @spec has_evidence?(map()) :: boolean()
  def has_evidence?(entry) do
    map_size(Map.get(entry, "evidence", %{})) > 0
  end

  # Private functions

  defp build_passage(entry, corpus) do
    entry
    |> Map.get("evidence", %{})
    |> Enum.flat_map(fn {doc_id, _} ->
      case Map.get(corpus, doc_id) do
        nil ->
          []

        %{sentences: sents} ->
          sents
          |> Enum.with_index()
          |> Enum.map(fn {sent, idx} -> "[#{doc_id}:#{idx}] #{sent}" end)
      end
    end)
    |> Enum.join("\n")
  end

  defp get_sentence(corpus, doc_id, sent_idx) do
    corpus
    |> Map.get(doc_id, %{sentences: []})
    |> Map.get(:sentences, [])
    |> Enum.at(sent_idx, "")
  end
end

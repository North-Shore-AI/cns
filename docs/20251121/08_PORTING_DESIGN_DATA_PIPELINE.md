# CNS Data Pipeline Porting Design: Python to Elixir

**Version:** 1.0.0
**Date:** 2025-11-21
**Status:** Design Document
**Author:** Claude Code

---

## Executive Summary

This document provides a comprehensive design for porting the CNS data pipeline from Python to Elixir. The pipeline handles scientific claim extraction dataset processing, validation, and training data preparation. The Elixir implementation will leverage OTP concurrency, Explorer dataframes, and integration with the existing crucible_datasets infrastructure.

---

## Table of Contents

1. [Dataset Converters](#1-dataset-converters)
2. [Dataset Validation](#2-dataset-validation)
3. [Lineage Tracking](#3-lineage-tracking)
4. [Data Schemas](#4-data-schemas)
5. [Integration Strategy](#5-integration-strategy)
6. [Effort Estimation](#6-effort-estimation)

---

## 1. Dataset Converters

### 1.1 Python Implementation Summary

The Python pipeline includes three main converters:

| Script | Source Format | Purpose |
|--------|--------------|---------|
| `convert_scifact.py` | SciFact JSONL | Scientific fact-checking claims |
| `convert_fever.py` | FEVER JSONL + Wiki TSV/JSONL | Wikipedia-based fact verification |
| `csv_to_claim_jsonl.py` | Manual annotation CSV | Human-annotated training data |

**Key Transformations:**
- Claims → `CLAIM[c1]: text` format with sequential IDs
- Evidence → `CLAIM[c2..n]: sentence [DocID]`
- Labels → `RELATION: cx supports/refutes cy`
- Output: `{prompt, completion, metadata}` JSONL

### 1.2 Elixir Module Design

```elixir
defmodule CNS.Pipeline.Converters do
  @moduledoc """
  Dataset conversion modules for claim extraction training data.
  """
end
```

#### 1.2.1 SciFact Converter

```elixir
defmodule CNS.Pipeline.Converters.SciFact do
  @moduledoc """
  Converts SciFact dataset to claim-extractor training format.

  Input:
  - claims_dev.jsonl or claims_train.jsonl
  - corpus.jsonl with abstracts

  Output:
  - JSONL with {prompt, completion, metadata}
  """

  alias CNS.Pipeline.Schema.TrainingExample
  alias CNS.Pipeline.Schema.ClaimEntry

  @type corpus_map :: %{String.t() => %{sentences: [String.t()]}}

  # Public API

  @doc """
  Convert SciFact files to training examples stream.
  """
  @spec convert(Path.t(), Path.t(), keyword()) :: Enumerable.t(TrainingExample.t())
  def convert(claims_path, corpus_path, opts \\ []) do
    corpus = load_corpus(corpus_path)

    claims_path
    |> File.stream!()
    |> Stream.map(&Jason.decode!/1)
    |> Stream.filter(&has_evidence?/1)
    |> Stream.map(&(build_example(&1, corpus, opts)))
  end

  @doc """
  Write converted examples to JSONL file.
  """
  @spec to_jsonl(Enumerable.t(), Path.t()) :: :ok | {:error, term()}
  def to_jsonl(examples, output_path) do
    examples
    |> Stream.map(&TrainingExample.to_json/1)
    |> Stream.intersperse("\n")
    |> Stream.into(File.stream!(output_path))
    |> Stream.run()
  end

  # Private functions

  @spec load_corpus(Path.t()) :: corpus_map()
  defp load_corpus(path) do
    path
    |> File.stream!()
    |> Stream.map(&Jason.decode!/1)
    |> Enum.into(%{}, fn entry ->
      doc_id = extract_doc_id(entry)
      {doc_id, %{sentences: Map.get(entry, "abstract", [])}}
    end)
  end

  @spec extract_doc_id(map()) :: String.t()
  defp extract_doc_id(entry) do
    entry["doc_id"] || entry["docid"] || entry["id"] || raise "No doc_id found"
  end

  @spec normalize_label(String.t()) :: String.t()
  defp normalize_label(label) do
    case String.upcase(label) do
      "SUPPORTS" -> "supports"
      "SUPPORT" -> "supports"
      "REFUTES" -> "refutes"
      "CONTRADICT" -> "refutes"
      other -> String.downcase(other)
    end
  end

  @spec build_passage(map(), corpus_map()) :: String.t()
  defp build_passage(claim_entry, corpus) do
    claim_entry
    |> Map.get("evidence", %{})
    |> Enum.flat_map(fn {doc_id, _} ->
      case Map.get(corpus, doc_id) do
        nil -> []
        %{sentences: sents} ->
          sents
          |> Enum.with_index()
          |> Enum.map(fn {sent, idx} -> "[#{doc_id}:#{idx}] #{sent}" end)
      end
    end)
    |> Enum.join("\n")
  end

  @spec gather_evidence(map(), corpus_map()) :: [{String.t(), String.t(), String.t()}]
  defp gather_evidence(claim_entry, corpus) do
    claim_entry
    |> Map.get("evidence", %{})
    |> Enum.flat_map(fn {doc_id, evidence_sets} ->
      Enum.flat_map(evidence_sets, fn %{"label" => label, "sentences" => sent_ids} ->
        Enum.map(sent_ids, fn sent_idx ->
          text = get_sentence(corpus, doc_id, sent_idx)
          {text, normalize_label(label), "[#{doc_id}:#{sent_idx}]"}
        end)
      end)
    end)
  end

  @spec get_sentence(corpus_map(), String.t(), integer()) :: String.t()
  defp get_sentence(corpus, doc_id, sent_idx) do
    corpus
    |> Map.get(doc_id, %{sentences: []})
    |> Map.get(:sentences, [])
    |> Enum.at(sent_idx, "")
  end

  @spec has_evidence?(map()) :: boolean()
  defp has_evidence?(entry), do: map_size(Map.get(entry, "evidence", %{})) > 0

  @spec build_example(map(), corpus_map(), keyword()) :: TrainingExample.t()
  defp build_example(claim_entry, corpus, _opts) do
    claim_text = claim_entry["claim"]
    evidence = gather_evidence(claim_entry, corpus)
    passage = build_passage(claim_entry, corpus)

    completion = build_completion(claim_text, evidence)
    prompt = build_prompt(passage)

    %TrainingExample{
      prompt: prompt,
      completion: completion,
      metadata: %{
        source: "scifact",
        claim_id: claim_entry["id"]
      }
    }
  end

  @spec build_prompt(String.t()) :: String.t()
  defp build_prompt(passage) do
    """
    Extract all claims and relations from the following passage.
    Output format: CLAIM[c1]: main claim, CLAIM[c2..n]: evidence, RELATION: cx label cy

    Passage:
    #{passage}
    """
  end

  @spec build_completion(String.t(), [{String.t(), String.t(), String.t()}]) :: String.t()
  defp build_completion(claim_text, evidence) do
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
end
```

#### 1.2.2 FEVER Converter

```elixir
defmodule CNS.Pipeline.Converters.FEVER do
  @moduledoc """
  Converts FEVER dataset to claim-extractor training format.

  Handles large Wikipedia dump files with streaming.
  """

  alias CNS.Pipeline.Schema.TrainingExample

  @type wiki_map :: %{{String.t(), integer()} => String.t()}

  # Public API

  @spec convert(Path.t(), Path.t(), keyword()) :: Enumerable.t(TrainingExample.t())
  def convert(claims_path, wiki_dir, opts \\ []) do
    include_nei = Keyword.get(opts, :include_nei, false)
    wiki_map = load_wiki_sentences(wiki_dir)

    claims_path
    |> File.stream!()
    |> Stream.map(&Jason.decode!/1)
    |> Stream.filter(&(include_nei or &1["label"] != "NOT ENOUGH INFO"))
    |> Stream.filter(&has_valid_evidence?/1)
    |> Stream.map(&(build_example(&1, wiki_map)))
  end

  @spec to_jsonl(Enumerable.t(), Path.t()) :: :ok
  def to_jsonl(examples, output_path) do
    CNS.Pipeline.Converters.SciFact.to_jsonl(examples, output_path)
  end

  # Private functions

  @spec load_wiki_sentences(Path.t()) :: wiki_map()
  defp load_wiki_sentences(wiki_dir) do
    wiki_dir
    |> File.ls!()
    |> Enum.filter(&String.ends_with?(&1, [".jsonl", ".tsv"]))
    |> Task.async_stream(fn file ->
      path = Path.join(wiki_dir, file)
      parse_wiki_file(path)
    end, max_concurrency: System.schedulers_online())
    |> Enum.reduce(%{}, fn {:ok, map}, acc -> Map.merge(acc, map) end)
  end

  @spec parse_wiki_file(Path.t()) :: wiki_map()
  defp parse_wiki_file(path) do
    cond do
      String.ends_with?(path, ".jsonl") -> parse_wiki_jsonl(path)
      String.ends_with?(path, ".tsv") -> parse_wiki_tsv(path)
      true -> %{}
    end
  end

  @spec parse_wiki_jsonl(Path.t()) :: wiki_map()
  defp parse_wiki_jsonl(path) do
    path
    |> File.stream!()
    |> Stream.map(&Jason.decode!/1)
    |> Enum.reduce(%{}, fn entry, acc ->
      page_id = entry["id"]
      entry["lines"]
      |> String.split("\n")
      |> Enum.with_index()
      |> Enum.reduce(acc, fn {line, idx}, inner_acc ->
        case String.split(line, "\t", parts: 2) do
          [_num, text] -> Map.put(inner_acc, {page_id, idx}, text)
          _ -> inner_acc
        end
      end)
    end)
  end

  @spec parse_wiki_tsv(Path.t()) :: wiki_map()
  defp parse_wiki_tsv(path) do
    path
    |> File.stream!()
    |> NimbleCSV.RFC4180.parse_stream(separator: "\t")
    |> Enum.reduce(%{}, fn [page_id, sent_idx | rest], acc ->
      text = Enum.at(rest, 0, "")
      Map.put(acc, {page_id, String.to_integer(sent_idx)}, text)
    end)
  end

  @spec has_valid_evidence?(map()) :: boolean()
  defp has_valid_evidence?(entry) do
    case entry["evidence"] do
      nil -> false
      [] -> false
      [[]] -> false
      _ -> true
    end
  end

  @spec build_example(map(), wiki_map()) :: TrainingExample.t()
  defp build_example(entry, wiki_map) do
    claim_text = entry["claim"]
    label = normalize_label(entry["label"])

    evidence_texts =
      entry["evidence"]
      |> List.first([])
      |> Enum.map(fn [_, _, page, sent_idx] ->
        Map.get(wiki_map, {page, sent_idx}, "")
      end)
      |> Enum.reject(&(&1 == ""))

    completion = build_completion(claim_text, evidence_texts, label)

    %TrainingExample{
      prompt: build_prompt(claim_text),
      completion: completion,
      metadata: %{
        source: "fever",
        id: entry["id"],
        label: entry["label"]
      }
    }
  end

  @spec normalize_label(String.t()) :: String.t()
  defp normalize_label("SUPPORTS"), do: "supports"
  defp normalize_label("REFUTES"), do: "refutes"
  defp normalize_label(other), do: String.downcase(other)

  @spec build_prompt(String.t()) :: String.t()
  defp build_prompt(claim) do
    """
    Extract supporting or refuting evidence for the following claim.

    Claim: #{claim}
    """
  end

  @spec build_completion(String.t(), [String.t()], String.t()) :: String.t()
  defp build_completion(claim_text, evidence_texts, label) do
    claims = ["CLAIM[c1]: #{claim_text}"]

    {evidence_claims, relations} =
      evidence_texts
      |> Enum.with_index(2)
      |> Enum.map(fn {text, idx} ->
        {"CLAIM[c#{idx}]: #{text}", "RELATION: c#{idx} #{label} c1"}
      end)
      |> Enum.unzip()

    (claims ++ evidence_claims ++ relations) |> Enum.join("\n")
  end
end
```

#### 1.2.3 CSV to JSONL Converter

```elixir
defmodule CNS.Pipeline.Converters.CSV do
  @moduledoc """
  Converts manually annotated CSV files to JSONL training format.
  """

  alias CNS.Pipeline.Schema.TrainingExample

  @spec convert(Path.t(), keyword()) :: Enumerable.t(TrainingExample.t())
  def convert(csv_path, opts \\ []) do
    csv_path
    |> File.stream!()
    |> NimbleCSV.RFC4180.parse_stream()
    |> Stream.drop(1)  # Skip header
    |> Stream.map(&parse_row(&1, opts))
  end

  @spec to_jsonl(Enumerable.t(), Path.t()) :: :ok
  def to_jsonl(examples, output_path) do
    CNS.Pipeline.Converters.SciFact.to_jsonl(examples, output_path)
  end

  @spec parse_row([String.t()], keyword()) :: TrainingExample.t()
  defp parse_row([passage, claims_raw, relations_raw | _rest], opts) do
    source = Keyword.get(opts, :source, "manual")

    claims = parse_claims(claims_raw)
    relations = parse_relations(relations_raw)

    completion = Enum.join(claims ++ relations, "\n")

    %TrainingExample{
      prompt: build_prompt(passage),
      completion: completion,
      metadata: %{source: source}
    }
  end

  @spec parse_claims(String.t()) :: [String.t()]
  defp parse_claims(raw) do
    raw
    |> String.split("|")
    |> Enum.map(&String.trim/1)
    |> Enum.reject(&(&1 == ""))
  end

  @spec parse_relations(String.t()) :: [String.t()]
  defp parse_relations(raw) do
    raw
    |> String.split(";")
    |> Enum.map(&String.trim/1)
    |> Enum.reject(&(&1 == ""))
    |> Enum.map(&format_relation/1)
  end

  @spec format_relation(String.t()) :: String.t()
  defp format_relation(rel) do
    case Regex.run(~r/^(c\d+)\s+(supports|refutes)\s+(c\d+)$/i, rel) do
      [_, src, label, dst] -> "RELATION: #{src} #{String.downcase(label)} #{dst}"
      _ -> "RELATION: #{rel}"
    end
  end

  @spec build_prompt(String.t()) :: String.t()
  defp build_prompt(passage) do
    """
    Extract all claims and relations from the following passage.

    #{passage}
    """
  end
end
```

### 1.3 Public API Summary

| Module | Function | Description |
|--------|----------|-------------|
| `Converters.SciFact` | `convert/3` | Stream SciFact to training examples |
| `Converters.SciFact` | `to_jsonl/2` | Write examples to JSONL |
| `Converters.FEVER` | `convert/3` | Stream FEVER to training examples |
| `Converters.FEVER` | `to_jsonl/2` | Write examples to JSONL |
| `Converters.CSV` | `convert/2` | Stream CSV to training examples |
| `Converters.CSV` | `to_jsonl/2` | Write examples to JSONL |

### 1.4 Dependencies

- `jason` - JSON encoding/decoding
- `nimble_csv` - CSV parsing
- `explorer` (optional) - DataFrame operations for large datasets

### 1.5 Estimated Effort

| Component | Effort | Notes |
|-----------|--------|-------|
| SciFact Converter | 3 days | Core converter, well-documented format |
| FEVER Converter | 4 days | Complex wiki loading, multiple formats |
| CSV Converter | 1 day | Simple transformation |
| Tests | 2 days | Unit tests, integration tests |
| **Total** | **10 days** | |

---

## 2. Dataset Validation

### 2.1 Python Implementation Summary

`validate_dataset.py` provides comprehensive validation:

- **Schema validation**: prompt/completion non-empty strings
- **Claim parsing**: Sequential IDs (c1, c2, c3...)
- **Exact matching**: Gold claim text verification
- **Embedding matching**: Semantic similarity via sentence-transformers
- **Relation validation**: References to valid claim IDs
- **Quality scoring**: Error accumulation and reporting

Key class: `EmbeddingMatcher` using sentence-transformers for semantic similarity.

### 2.2 Elixir Module Design

```elixir
defmodule CNS.Pipeline.Validation do
  @moduledoc """
  Dataset validation for claim-extractor JSONL files.

  Validates schema conformance, claim structure, evidence matching,
  and provides quality metrics.
  """
end
```

#### 2.2.1 Core Validator

```elixir
defmodule CNS.Pipeline.Validation.Validator do
  @moduledoc """
  Core validation engine for training datasets.
  """

  alias CNS.Pipeline.Schema.{TrainingExample, ValidationError, ValidationResult}
  alias CNS.Pipeline.Validation.{ClaimValidator, EvidenceValidator, RelationValidator}

  @type validation_opts :: [
    claims_path: Path.t() | nil,
    corpus_path: Path.t() | nil,
    embedding_mode: boolean(),
    max_examples: integer() | nil,
    similarity_threshold: float()
  ]

  # Public API

  @doc """
  Validate a JSONL dataset file.

  Returns a ValidationResult with errors, warnings, and metrics.
  """
  @spec validate(Path.t(), validation_opts()) :: ValidationResult.t()
  def validate(dataset_path, opts \\ []) do
    context = build_context(opts)

    {errors, stats} =
      dataset_path
      |> stream_dataset(opts[:max_examples])
      |> Stream.with_index(1)
      |> Enum.reduce({[], %{}}, fn {row, idx}, {errors, stats} ->
        case validate_row(idx, row, context) do
          {:ok, row_stats} ->
            {errors, merge_stats(stats, row_stats)}
          {:error, row_errors, row_stats} ->
            {errors ++ row_errors, merge_stats(stats, row_stats)}
        end
      end)

    %ValidationResult{
      valid?: errors == [],
      errors: errors,
      stats: stats,
      total_examples: stats[:total] || 0
    }
  end

  @doc """
  Validate and optionally clean a dataset, outputting valid rows.
  """
  @spec validate_and_clean(Path.t(), Path.t(), validation_opts()) :: ValidationResult.t()
  def validate_and_clean(input_path, output_path, opts \\ []) do
    context = build_context(opts)

    {valid_rows, errors, stats} =
      input_path
      |> stream_dataset(opts[:max_examples])
      |> Stream.with_index(1)
      |> Enum.reduce({[], [], %{}}, fn {row, idx}, {valid, errs, stats} ->
        case validate_row(idx, row, context) do
          {:ok, row_stats} ->
            {[row | valid], errs, merge_stats(stats, row_stats)}
          {:error, row_errors, row_stats} ->
            {valid, errs ++ row_errors, merge_stats(stats, row_stats)}
        end
      end)

    # Write valid rows to output
    valid_rows
    |> Enum.reverse()
    |> Enum.each(fn row ->
      json = Jason.encode!(row)
      File.write!(output_path, json <> "\n", [:append])
    end)

    %ValidationResult{
      valid?: errors == [],
      errors: errors,
      stats: Map.put(stats, :cleaned_count, length(valid_rows)),
      total_examples: stats[:total] || 0
    }
  end

  # Private functions

  @spec stream_dataset(Path.t(), integer() | nil) :: Enumerable.t()
  defp stream_dataset(path, nil) do
    path
    |> File.stream!()
    |> Stream.map(&Jason.decode!/1)
  end
  defp stream_dataset(path, max) do
    path
    |> File.stream!()
    |> Stream.take(max)
    |> Stream.map(&Jason.decode!/1)
  end

  @spec build_context(validation_opts()) :: map()
  defp build_context(opts) do
    claim_map = if opts[:claims_path], do: load_claim_map(opts[:claims_path]), else: %{}
    corpus_map = if opts[:corpus_path], do: load_corpus_map(opts[:corpus_path]), else: %{}

    embedding_matcher =
      if opts[:embedding_mode] do
        CNS.Pipeline.Validation.EmbeddingMatcher.new(opts[:similarity_threshold] || 0.85)
      else
        nil
      end

    %{
      claim_map: claim_map,
      corpus_map: corpus_map,
      embedding_matcher: embedding_matcher,
      evidence_map: build_evidence_map(claim_map, corpus_map)
    }
  end

  @spec validate_row(integer(), map(), map()) ::
    {:ok, map()} | {:error, [ValidationError.t()], map()}
  defp validate_row(idx, row, context) do
    errors = []
    stats = %{total: 1}

    # Schema validation
    errors = errors ++ validate_schema(idx, row)

    # Parse completion
    case CNS.Pipeline.Schema.ClaimParser.parse(row["completion"]) do
      {:ok, claims, relations} ->
        # Claim validation
        claim_errors = ClaimValidator.validate(idx, claims, context)

        # Evidence validation
        evidence_errors = EvidenceValidator.validate(idx, claims, context)

        # Relation validation
        relation_errors = RelationValidator.validate(idx, relations, claims)

        all_errors = errors ++ claim_errors ++ evidence_errors ++ relation_errors

        if all_errors == [] do
          {:ok, stats}
        else
          {:error, all_errors, stats}
        end

      {:error, parse_error} ->
        {:error, [%ValidationError{line: idx, message: parse_error}], stats}
    end
  end

  @spec validate_schema(integer(), map()) :: [ValidationError.t()]
  defp validate_schema(idx, row) do
    errors = []

    errors = if !is_binary(row["prompt"]) or row["prompt"] == "" do
      [%ValidationError{line: idx, message: "prompt must be non-empty string"} | errors]
    else
      errors
    end

    errors = if !is_binary(row["completion"]) or row["completion"] == "" do
      [%ValidationError{line: idx, message: "completion must be non-empty string"} | errors]
    else
      errors
    end

    errors
  end

  @spec load_claim_map(Path.t()) :: map()
  defp load_claim_map(path) do
    path
    |> File.stream!()
    |> Stream.map(&Jason.decode!/1)
    |> Enum.into(%{}, fn entry -> {entry["id"], entry} end)
  end

  @spec load_corpus_map(Path.t()) :: map()
  defp load_corpus_map(path) do
    path
    |> File.stream!()
    |> Stream.map(&Jason.decode!/1)
    |> Enum.into(%{}, fn entry ->
      doc_id = entry["doc_id"] || entry["id"]
      {doc_id, %{sentences: entry["abstract"] || []}}
    end)
  end

  @spec build_evidence_map(map(), map()) :: map()
  defp build_evidence_map(claim_map, corpus_map) do
    claim_map
    |> Enum.reduce(%{}, fn {claim_id, entry}, acc ->
      evidence =
        entry
        |> Map.get("evidence", %{})
        |> Enum.flat_map(fn {doc_id, evidence_sets} ->
          Enum.flat_map(evidence_sets, fn %{"sentences" => sent_ids} ->
            Enum.map(sent_ids, fn sent_idx ->
              get_sentence(corpus_map, doc_id, sent_idx)
            end)
          end)
        end)

      Map.put(acc, claim_id, evidence)
    end)
  end

  @spec get_sentence(map(), String.t(), integer()) :: String.t()
  defp get_sentence(corpus, doc_id, sent_idx) do
    corpus
    |> Map.get(doc_id, %{sentences: []})
    |> Map.get(:sentences, [])
    |> Enum.at(sent_idx, "")
  end

  @spec merge_stats(map(), map()) :: map()
  defp merge_stats(acc, new) do
    Map.merge(acc, new, fn _k, v1, v2 -> v1 + v2 end)
  end
end
```

#### 2.2.2 Claim Validator

```elixir
defmodule CNS.Pipeline.Validation.ClaimValidator do
  @moduledoc """
  Validates claim structure and sequential IDs.
  """

  alias CNS.Pipeline.Schema.ValidationError

  @spec validate(integer(), [map()], map()) :: [ValidationError.t()]
  def validate(idx, claims, context) do
    errors = []

    # Check sequential IDs
    errors = errors ++ validate_sequential_ids(idx, claims)

    # Check c1 matches gold claim if available
    errors = errors ++ validate_gold_claim(idx, claims, context)

    errors
  end

  @spec validate_sequential_ids(integer(), [map()]) :: [ValidationError.t()]
  defp validate_sequential_ids(idx, claims) do
    expected_ids = 1..length(claims) |> Enum.map(&"c#{&1}")
    actual_ids = Enum.map(claims, & &1.id)

    if actual_ids != expected_ids do
      [%ValidationError{
        line: idx,
        message: "Non-sequential claim IDs: expected #{inspect(expected_ids)}, got #{inspect(actual_ids)}"
      }]
    else
      []
    end
  end

  @spec validate_gold_claim(integer(), [map()], map()) :: [ValidationError.t()]
  defp validate_gold_claim(_idx, [], _context), do: []
  defp validate_gold_claim(idx, [c1 | _], %{claim_map: claim_map}) when map_size(claim_map) == 0 do
    []
  end
  defp validate_gold_claim(idx, [c1 | _], %{claim_map: claim_map} = context) do
    # Find matching gold claim (would need claim_id from metadata)
    # Simplified: just check c1 exists
    []
  end
end
```

#### 2.2.3 Embedding Matcher

```elixir
defmodule CNS.Pipeline.Validation.EmbeddingMatcher do
  @moduledoc """
  Semantic similarity matching using embeddings.

  Uses Bumblebee for sentence-transformers models.
  """

  defstruct [:model, :tokenizer, :threshold]

  @type t :: %__MODULE__{
    model: Nx.Serving.t(),
    tokenizer: any(),
    threshold: float()
  }

  @doc """
  Create a new embedding matcher with the given similarity threshold.
  """
  @spec new(float()) :: t()
  def new(threshold \\ 0.85) do
    # Load sentence-transformers model via Bumblebee
    {:ok, model_info} = Bumblebee.load_model({:hf, "sentence-transformers/all-MiniLM-L6-v2"})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "sentence-transformers/all-MiniLM-L6-v2"})

    serving = Bumblebee.Text.text_embedding(model_info, tokenizer)

    %__MODULE__{
      model: serving,
      tokenizer: tokenizer,
      threshold: threshold
    }
  end

  @doc """
  Check if two texts are semantically similar.
  """
  @spec similar?(t(), String.t(), String.t()) :: boolean()
  def similar?(%__MODULE__{} = matcher, text1, text2) do
    similarity(matcher, text1, text2) >= matcher.threshold
  end

  @doc """
  Compute cosine similarity between two texts.
  """
  @spec similarity(t(), String.t(), String.t()) :: float()
  def similarity(%__MODULE__{model: serving}, text1, text2) do
    %{embedding: emb1} = Nx.Serving.run(serving, text1)
    %{embedding: emb2} = Nx.Serving.run(serving, text2)

    cosine_similarity(emb1, emb2)
  end

  @doc """
  Find best matching text from candidates.
  """
  @spec find_best_match(t(), String.t(), [String.t()]) :: {String.t(), float()} | nil
  def find_best_match(%__MODULE__{} = matcher, query, candidates) do
    candidates
    |> Enum.map(fn cand -> {cand, similarity(matcher, query, cand)} end)
    |> Enum.max_by(fn {_, sim} -> sim end, fn -> nil end)
  end

  @spec cosine_similarity(Nx.Tensor.t(), Nx.Tensor.t()) :: float()
  defp cosine_similarity(a, b) do
    dot = Nx.dot(a, b) |> Nx.to_number()
    norm_a = Nx.LinAlg.norm(a) |> Nx.to_number()
    norm_b = Nx.LinAlg.norm(b) |> Nx.to_number()

    if norm_a > 0 and norm_b > 0 do
      dot / (norm_a * norm_b)
    else
      0.0
    end
  end
end
```

#### 2.2.4 Quality Scorer

```elixir
defmodule CNS.Pipeline.Validation.QualityScorer do
  @moduledoc """
  Compute quality scores for validated datasets.
  """

  @type quality_report :: %{
    schema_errors: integer(),
    claim_errors: integer(),
    relation_errors: integer(),
    evidence_match_rate: float(),
    overall_score: float()
  }

  @spec score(CNS.Pipeline.Schema.ValidationResult.t()) :: quality_report()
  def score(%{errors: errors, stats: stats}) do
    total = stats[:total] || 1

    schema_errors = count_by_type(errors, :schema)
    claim_errors = count_by_type(errors, :claim)
    relation_errors = count_by_type(errors, :relation)

    error_rate = length(errors) / total

    %{
      schema_errors: schema_errors,
      claim_errors: claim_errors,
      relation_errors: relation_errors,
      evidence_match_rate: stats[:evidence_matches] || 0 / max(stats[:evidence_total] || 1, 1),
      overall_score: 1.0 - error_rate
    }
  end

  @spec count_by_type([map()], atom()) :: integer()
  defp count_by_type(errors, type) do
    Enum.count(errors, &(&1.type == type))
  end
end
```

### 2.3 Public API Summary

| Module | Function | Description |
|--------|----------|-------------|
| `Validation.Validator` | `validate/2` | Validate dataset, return errors/stats |
| `Validation.Validator` | `validate_and_clean/3` | Validate and output clean dataset |
| `Validation.EmbeddingMatcher` | `new/1` | Create embedding matcher |
| `Validation.EmbeddingMatcher` | `similar?/3` | Check semantic similarity |
| `Validation.EmbeddingMatcher` | `similarity/3` | Compute similarity score |
| `Validation.QualityScorer` | `score/1` | Generate quality report |

### 2.4 Dependencies

- `jason` - JSON parsing
- `nx` - Numerical computing
- `bumblebee` - Pre-trained models (sentence-transformers)
- `exla` or `torchx` - Nx backend for GPU acceleration

### 2.5 Estimated Effort

| Component | Effort | Notes |
|-----------|--------|-------|
| Core Validator | 3 days | Main validation logic |
| Claim/Relation Validators | 2 days | Parsing and validation |
| Embedding Matcher | 4 days | Bumblebee integration, optimization |
| Quality Scorer | 1 day | Metrics computation |
| Tests | 3 days | Property-based testing for edge cases |
| **Total** | **13 days** | |

---

## 3. Lineage Tracking

### 3.1 Python Implementation Summary

`record_lineage.py` provides simple artifact tracking:

- SHA-256 hash computation for each file
- File size tracking
- JSON output format
- CLI interface for batch processing

`train_claim_extractor.py` extends this with:
- Git commit tracking
- Timestamp recording
- Config hash for reproducibility
- Provenance logs saved alongside artifacts

### 3.2 Elixir Module Design

```elixir
defmodule CNS.Pipeline.Lineage do
  @moduledoc """
  Data lineage and artifact tracking for reproducibility.

  Tracks file hashes, sizes, timestamps, and git state for
  complete provenance of ML artifacts.
  """
end
```

#### 3.2.1 Core Lineage Tracker

```elixir
defmodule CNS.Pipeline.Lineage.Tracker do
  @moduledoc """
  Core lineage tracking with ETS-based storage.
  """

  use GenServer

  alias CNS.Pipeline.Lineage.{Record, Artifact}

  @table_name :cns_lineage

  # Client API

  @doc """
  Start the lineage tracker.
  """
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Record lineage for a file.
  """
  @spec record(Path.t()) :: {:ok, Record.t()} | {:error, term()}
  def record(path) do
    GenServer.call(__MODULE__, {:record, path})
  end

  @doc """
  Record lineage for multiple files.
  """
  @spec record_batch([Path.t()]) :: {:ok, [Record.t()]} | {:error, term()}
  def record_batch(paths) do
    GenServer.call(__MODULE__, {:record_batch, paths}, :infinity)
  end

  @doc """
  Get lineage record for a file.
  """
  @spec get(Path.t()) :: Record.t() | nil
  def get(path) do
    case :ets.lookup(@table_name, path) do
      [{^path, record}] -> record
      [] -> nil
    end
  end

  @doc """
  Get all lineage records.
  """
  @spec all() :: [Record.t()]
  def all do
    :ets.tab2list(@table_name) |> Enum.map(fn {_, record} -> record end)
  end

  @doc """
  Export lineage to JSON file.
  """
  @spec export(Path.t()) :: :ok
  def export(output_path) do
    records = all()

    json =
      records
      |> Enum.into(%{}, fn r -> {r.path, Record.to_map(r)} end)
      |> Jason.encode!(pretty: true)

    File.write!(output_path, json)
  end

  @doc """
  Create a provenance record for an artifact.
  """
  @spec create_provenance(Path.t(), keyword()) :: Artifact.t()
  def create_provenance(artifact_path, opts \\ []) do
    input_files = Keyword.get(opts, :inputs, [])
    config = Keyword.get(opts, :config, %{})

    %Artifact{
      path: artifact_path,
      sha256: compute_sha256(artifact_path),
      bytes: file_size(artifact_path),
      created_at: DateTime.utc_now(),
      git_commit: git_commit(),
      git_dirty?: git_dirty?(),
      inputs: Enum.map(input_files, &get/1),
      config_hash: hash_config(config)
    }
  end

  # Server callbacks

  @impl true
  def init(_opts) do
    table = :ets.new(@table_name, [:named_table, :set, :public, read_concurrency: true])
    {:ok, %{table: table}}
  end

  @impl true
  def handle_call({:record, path}, _from, state) do
    result = do_record(path)
    {:reply, result, state}
  end

  @impl true
  def handle_call({:record_batch, paths}, _from, state) do
    results =
      paths
      |> Task.async_stream(&do_record/1, max_concurrency: System.schedulers_online() * 2)
      |> Enum.map(fn {:ok, result} -> result end)

    {:reply, {:ok, results}, state}
  end

  # Private functions

  @spec do_record(Path.t()) :: {:ok, Record.t()} | {:error, term()}
  defp do_record(path) do
    with {:ok, stat} <- File.stat(path),
         {:ok, hash} <- {:ok, compute_sha256(path)} do
      record = %Record{
        path: path,
        sha256: hash,
        bytes: stat.size,
        recorded_at: DateTime.utc_now()
      }

      :ets.insert(@table_name, {path, record})
      {:ok, record}
    else
      {:error, reason} -> {:error, reason}
    end
  end

  @spec compute_sha256(Path.t()) :: String.t()
  defp compute_sha256(path) do
    File.stream!(path, [], 65_536)
    |> Enum.reduce(:crypto.hash_init(:sha256), &:crypto.hash_update(&2, &1))
    |> :crypto.hash_final()
    |> Base.encode16(case: :lower)
  end

  @spec file_size(Path.t()) :: integer()
  defp file_size(path) do
    case File.stat(path) do
      {:ok, %{size: size}} -> size
      _ -> 0
    end
  end

  @spec git_commit() :: String.t() | nil
  defp git_commit do
    case System.cmd("git", ["rev-parse", "HEAD"], stderr_to_stdout: true) do
      {commit, 0} -> String.trim(commit)
      _ -> nil
    end
  end

  @spec git_dirty?() :: boolean()
  defp git_dirty? do
    case System.cmd("git", ["status", "--porcelain"], stderr_to_stdout: true) do
      {"", 0} -> false
      {_, 0} -> true
      _ -> false
    end
  end

  @spec hash_config(map()) :: String.t()
  defp hash_config(config) do
    config
    |> Jason.encode!()
    |> then(&:crypto.hash(:sha256, &1))
    |> Base.encode16(case: :lower)
    |> String.slice(0, 16)
  end
end
```

#### 3.2.2 Lineage Records

```elixir
defmodule CNS.Pipeline.Lineage.Record do
  @moduledoc """
  A single lineage record for a file.
  """

  defstruct [:path, :sha256, :bytes, :recorded_at]

  @type t :: %__MODULE__{
    path: String.t(),
    sha256: String.t(),
    bytes: integer(),
    recorded_at: DateTime.t()
  }

  @spec to_map(t()) :: map()
  def to_map(%__MODULE__{} = r) do
    %{
      sha256: r.sha256,
      bytes: r.bytes,
      recorded_at: DateTime.to_iso8601(r.recorded_at)
    }
  end
end

defmodule CNS.Pipeline.Lineage.Artifact do
  @moduledoc """
  A provenance record for a generated artifact.
  """

  defstruct [
    :path,
    :sha256,
    :bytes,
    :created_at,
    :git_commit,
    :git_dirty?,
    :inputs,
    :config_hash
  ]

  @type t :: %__MODULE__{
    path: String.t(),
    sha256: String.t(),
    bytes: integer(),
    created_at: DateTime.t(),
    git_commit: String.t() | nil,
    git_dirty?: boolean(),
    inputs: [CNS.Pipeline.Lineage.Record.t()],
    config_hash: String.t()
  }

  @spec to_json(t()) :: String.t()
  def to_json(%__MODULE__{} = artifact) do
    %{
      path: artifact.path,
      sha256: artifact.sha256,
      bytes: artifact.bytes,
      created_at: DateTime.to_iso8601(artifact.created_at),
      git_commit: artifact.git_commit,
      git_dirty: artifact.git_dirty?,
      inputs: Enum.map(artifact.inputs, &CNS.Pipeline.Lineage.Record.to_map/1),
      config_hash: artifact.config_hash
    }
    |> Jason.encode!(pretty: true)
  end
end
```

#### 3.2.3 Mnesia-based Persistent Storage (Optional)

```elixir
defmodule CNS.Pipeline.Lineage.Store do
  @moduledoc """
  Persistent lineage storage using Mnesia.

  Use for long-term artifact tracking across experiments.
  """

  alias CNS.Pipeline.Lineage.{Record, Artifact}

  @record_table :lineage_records
  @artifact_table :lineage_artifacts

  @doc """
  Initialize Mnesia tables.
  """
  @spec init() :: :ok | {:error, term()}
  def init do
    :mnesia.create_schema([node()])
    :mnesia.start()

    :mnesia.create_table(@record_table, [
      attributes: [:path, :sha256, :bytes, :recorded_at],
      disc_copies: [node()],
      type: :set
    ])

    :mnesia.create_table(@artifact_table, [
      attributes: [:id, :path, :sha256, :bytes, :created_at, :git_commit, :git_dirty, :inputs, :config_hash],
      disc_copies: [node()],
      type: :set
    ])

    :ok
  end

  @doc """
  Store a lineage record.
  """
  @spec store_record(Record.t()) :: :ok | {:error, term()}
  def store_record(%Record{} = record) do
    :mnesia.transaction(fn ->
      :mnesia.write({@record_table, record.path, record.sha256, record.bytes, record.recorded_at})
    end)
    |> case do
      {:atomic, _} -> :ok
      {:aborted, reason} -> {:error, reason}
    end
  end

  @doc """
  Store an artifact provenance record.
  """
  @spec store_artifact(Artifact.t()) :: :ok | {:error, term()}
  def store_artifact(%Artifact{} = artifact) do
    id = :crypto.strong_rand_bytes(16) |> Base.encode16(case: :lower)

    :mnesia.transaction(fn ->
      :mnesia.write({
        @artifact_table,
        id,
        artifact.path,
        artifact.sha256,
        artifact.bytes,
        artifact.created_at,
        artifact.git_commit,
        artifact.git_dirty?,
        artifact.inputs,
        artifact.config_hash
      })
    end)
    |> case do
      {:atomic, _} -> :ok
      {:aborted, reason} -> {:error, reason}
    end
  end

  @doc """
  Query artifacts by input file hash.
  """
  @spec find_by_input(String.t()) :: [Artifact.t()]
  def find_by_input(input_sha256) do
    :mnesia.transaction(fn ->
      :mnesia.match_object({@artifact_table, :_, :_, :_, :_, :_, :_, :_, :_, :_})
    end)
    |> case do
      {:atomic, results} ->
        results
        |> Enum.filter(fn {_, _, _, _, _, _, _, _, inputs, _} ->
          Enum.any?(inputs, &(&1.sha256 == input_sha256))
        end)
        |> Enum.map(&tuple_to_artifact/1)
      _ -> []
    end
  end

  @spec tuple_to_artifact(tuple()) :: Artifact.t()
  defp tuple_to_artifact({_, _id, path, sha256, bytes, created_at, git_commit, git_dirty, inputs, config_hash}) do
    %Artifact{
      path: path,
      sha256: sha256,
      bytes: bytes,
      created_at: created_at,
      git_commit: git_commit,
      git_dirty?: git_dirty,
      inputs: inputs,
      config_hash: config_hash
    }
  end
end
```

### 3.3 Public API Summary

| Module | Function | Description |
|--------|----------|-------------|
| `Lineage.Tracker` | `start_link/1` | Start tracker GenServer |
| `Lineage.Tracker` | `record/1` | Record single file lineage |
| `Lineage.Tracker` | `record_batch/1` | Record multiple files |
| `Lineage.Tracker` | `get/1` | Get record for path |
| `Lineage.Tracker` | `export/1` | Export to JSON |
| `Lineage.Tracker` | `create_provenance/2` | Create artifact provenance |
| `Lineage.Store` | `init/0` | Initialize Mnesia tables |
| `Lineage.Store` | `store_record/1` | Persist record |
| `Lineage.Store` | `store_artifact/1` | Persist artifact |

### 3.4 Dependencies

- `:crypto` (OTP) - SHA-256 hashing
- `jason` - JSON encoding
- `:mnesia` (OTP) - Persistent storage (optional)

### 3.5 Estimated Effort

| Component | Effort | Notes |
|-----------|--------|-------|
| Core Tracker | 2 days | ETS-based, file hashing |
| Provenance Records | 1 day | Struct definitions |
| Mnesia Store | 2 days | Optional persistent storage |
| Tests | 1 day | File-based tests |
| **Total** | **6 days** | |

---

## 4. Data Schemas

### 4.1 Python Implementation Summary

The Python implementation defines formats implicitly through:

- JSONL structure: `{prompt, completion, metadata}`
- Claim format: `CLAIM[c1]: text`
- Relation format: `RELATION: src label dst`
- Regex patterns for parsing

### 4.2 Elixir Module Design

#### 4.2.1 Core Schemas with Ecto Changesets

```elixir
defmodule CNS.Pipeline.Schema do
  @moduledoc """
  Core data schemas for CNS pipeline with validation.
  """
end

defmodule CNS.Pipeline.Schema.ClaimEntry do
  @moduledoc """
  A single claim with ID and text.
  """

  use Ecto.Schema
  import Ecto.Changeset

  @primary_key false
  embedded_schema do
    field :id, :string
    field :text, :string
    field :doc_ref, :string  # Optional document reference
  end

  @type t :: %__MODULE__{
    id: String.t(),
    text: String.t(),
    doc_ref: String.t() | nil
  }

  @spec changeset(t() | %{}, map()) :: Ecto.Changeset.t()
  def changeset(claim, attrs) do
    claim
    |> cast(attrs, [:id, :text, :doc_ref])
    |> validate_required([:id, :text])
    |> validate_format(:id, ~r/^c\d+$/, message: "must be in format c1, c2, etc.")
    |> validate_length(:text, min: 1)
  end

  @spec new(map()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new(attrs) do
    %__MODULE__{}
    |> changeset(attrs)
    |> apply_action(:insert)
  end

  @spec to_line(t()) :: String.t()
  def to_line(%__MODULE__{} = claim) do
    base = "CLAIM[#{claim.id}]: #{claim.text}"
    if claim.doc_ref, do: "#{base} #{claim.doc_ref}", else: base
  end
end

defmodule CNS.Pipeline.Schema.Relation do
  @moduledoc """
  A relation between claims.
  """

  use Ecto.Schema
  import Ecto.Changeset

  @primary_key false
  embedded_schema do
    field :source, :string
    field :label, :string
    field :target, :string
  end

  @type t :: %__MODULE__{
    source: String.t(),
    label: String.t(),
    target: String.t()
  }

  @valid_labels ["supports", "refutes"]

  @spec changeset(t() | %{}, map()) :: Ecto.Changeset.t()
  def changeset(relation, attrs) do
    relation
    |> cast(attrs, [:source, :label, :target])
    |> validate_required([:source, :label, :target])
    |> validate_format(:source, ~r/^c\d+$/)
    |> validate_format(:target, ~r/^c\d+$/)
    |> validate_inclusion(:label, @valid_labels)
  end

  @spec new(map()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new(attrs) do
    %__MODULE__{}
    |> changeset(attrs)
    |> apply_action(:insert)
  end

  @spec to_line(t()) :: String.t()
  def to_line(%__MODULE__{} = rel) do
    "RELATION: #{rel.source} #{rel.label} #{rel.target}"
  end
end

defmodule CNS.Pipeline.Schema.TrainingExample do
  @moduledoc """
  A training example with prompt, completion, and metadata.
  """

  use Ecto.Schema
  import Ecto.Changeset

  @primary_key false
  embedded_schema do
    field :prompt, :string
    field :completion, :string
    field :metadata, :map, default: %{}
  end

  @type t :: %__MODULE__{
    prompt: String.t(),
    completion: String.t(),
    metadata: map()
  }

  @spec changeset(t() | %{}, map()) :: Ecto.Changeset.t()
  def changeset(example, attrs) do
    example
    |> cast(attrs, [:prompt, :completion, :metadata])
    |> validate_required([:prompt, :completion])
    |> validate_length(:prompt, min: 1)
    |> validate_length(:completion, min: 1)
  end

  @spec new(map()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def new(attrs) do
    %__MODULE__{}
    |> changeset(attrs)
    |> apply_action(:insert)
  end

  @spec to_json(t()) :: String.t()
  def to_json(%__MODULE__{} = example) do
    %{
      prompt: example.prompt,
      completion: example.completion,
      metadata: example.metadata
    }
    |> Jason.encode!()
  end

  @spec from_json(String.t()) :: {:ok, t()} | {:error, term()}
  def from_json(json) do
    with {:ok, map} <- Jason.decode(json) do
      new(map)
    end
  end
end

defmodule CNS.Pipeline.Schema.ValidationError do
  @moduledoc """
  A validation error with line number and message.
  """

  defstruct [:line, :message, :type]

  @type t :: %__MODULE__{
    line: integer(),
    message: String.t(),
    type: atom()
  }
end

defmodule CNS.Pipeline.Schema.ValidationResult do
  @moduledoc """
  Result of dataset validation.
  """

  defstruct [:valid?, :errors, :stats, :total_examples]

  @type t :: %__MODULE__{
    valid?: boolean(),
    errors: [CNS.Pipeline.Schema.ValidationError.t()],
    stats: map(),
    total_examples: integer()
  }
end
```

#### 4.2.2 Claim Parser

```elixir
defmodule CNS.Pipeline.Schema.ClaimParser do
  @moduledoc """
  Parse CLAIM/RELATION format text into structured data.
  """

  alias CNS.Pipeline.Schema.{ClaimEntry, Relation}

  @claim_regex ~r/^CLAIM\s*\[?(c\d+)\]?\s*:\s*(.+?)(?:\s+\[([^\]]+)\])?\s*$/i
  @relation_regex ~r/^RELATION\s*:\s*(c\d+)\s+(supports|refutes)\s+(c\d+)\s*$/i

  @doc """
  Parse completion text into claims and relations.
  """
  @spec parse(String.t()) :: {:ok, [ClaimEntry.t()], [Relation.t()]} | {:error, String.t()}
  def parse(text) do
    lines = String.split(text, "\n", trim: true)

    {claims, relations, errors} =
      Enum.reduce(lines, {[], [], []}, fn line, {claims, rels, errs} ->
        cond do
          String.match?(line, @claim_regex) ->
            case parse_claim_line(line) do
              {:ok, claim} -> {[claim | claims], rels, errs}
              {:error, msg} -> {claims, rels, [msg | errs]}
            end

          String.match?(line, @relation_regex) ->
            case parse_relation_line(line) do
              {:ok, rel} -> {claims, [rel | rels], errs}
              {:error, msg} -> {claims, rels, [msg | errs]}
            end

          String.trim(line) == "" ->
            {claims, rels, errs}

          true ->
            {claims, rels, ["Unrecognized line: #{line}" | errs]}
        end
      end)

    if errors == [] do
      {:ok, Enum.reverse(claims), Enum.reverse(relations)}
    else
      {:error, Enum.join(errors, "; ")}
    end
  end

  @doc """
  Parse a single CLAIM line.
  """
  @spec parse_claim_line(String.t()) :: {:ok, ClaimEntry.t()} | {:error, String.t()}
  def parse_claim_line(line) do
    case Regex.run(@claim_regex, line) do
      [_, id, text, doc_ref] ->
        ClaimEntry.new(%{id: String.downcase(id), text: String.trim(text), doc_ref: doc_ref})

      [_, id, text] ->
        ClaimEntry.new(%{id: String.downcase(id), text: String.trim(text)})

      nil ->
        {:error, "Invalid claim line: #{line}"}
    end
  end

  @doc """
  Parse a single RELATION line.
  """
  @spec parse_relation_line(String.t()) :: {:ok, Relation.t()} | {:error, String.t()}
  def parse_relation_line(line) do
    case Regex.run(@relation_regex, line) do
      [_, src, label, dst] ->
        Relation.new(%{
          source: String.downcase(src),
          label: String.downcase(label),
          target: String.downcase(dst)
        })

      nil ->
        {:error, "Invalid relation line: #{line}"}
    end
  end

  @doc """
  Render claims and relations back to text format.
  """
  @spec render([ClaimEntry.t()], [Relation.t()]) :: String.t()
  def render(claims, relations) do
    claim_lines = Enum.map(claims, &ClaimEntry.to_line/1)
    relation_lines = Enum.map(relations, &Relation.to_line/1)

    (claim_lines ++ relation_lines) |> Enum.join("\n")
  end

  @doc """
  Enforce that c1 matches the given gold text.
  """
  @spec enforce_c1([ClaimEntry.t()], String.t()) :: [ClaimEntry.t()]
  def enforce_c1([], gold_text) do
    [%ClaimEntry{id: "c1", text: gold_text}]
  end
  def enforce_c1([first | rest], gold_text) do
    if first.id == "c1" do
      [%{first | text: gold_text} | rest]
    else
      [%ClaimEntry{id: "c1", text: gold_text}, first | rest]
    end
  end
end
```

### 4.3 Public API Summary

| Module | Function | Description |
|--------|----------|-------------|
| `Schema.ClaimEntry` | `new/1` | Create validated claim |
| `Schema.ClaimEntry` | `to_line/1` | Render to text format |
| `Schema.Relation` | `new/1` | Create validated relation |
| `Schema.Relation` | `to_line/1` | Render to text format |
| `Schema.TrainingExample` | `new/1` | Create validated example |
| `Schema.TrainingExample` | `to_json/1` | Serialize to JSON |
| `Schema.TrainingExample` | `from_json/1` | Parse from JSON |
| `Schema.ClaimParser` | `parse/1` | Parse completion text |
| `Schema.ClaimParser` | `render/2` | Render to text |
| `Schema.ClaimParser` | `enforce_c1/2` | Enforce gold claim |

### 4.4 Dependencies

- `ecto` - Schema and changeset validation
- `jason` - JSON encoding/decoding

### 4.5 Estimated Effort

| Component | Effort | Notes |
|-----------|--------|-------|
| Ecto Schemas | 2 days | ClaimEntry, Relation, TrainingExample |
| ClaimParser | 2 days | Regex parsing, rendering |
| Validation Changesets | 1 day | Custom validators |
| Tests | 2 days | Property-based for parser |
| **Total** | **7 days** | |

---

## 5. Integration Strategy

### 5.1 Integration with crucible_datasets

The CNS data pipeline should integrate with the existing crucible_datasets infrastructure:

```elixir
defmodule CrucibleDatasets.Sources.CNS do
  @moduledoc """
  CNS claim extraction datasets for crucible_datasets.
  """

  @behaviour CrucibleDatasets.Source

  alias CNS.Pipeline.Converters.{SciFact, FEVER}
  alias CNS.Pipeline.Schema.TrainingExample

  @impl true
  def load(dataset_name, opts \\ []) do
    case dataset_name do
      "scifact" -> load_scifact(opts)
      "fever" -> load_fever(opts)
      _ -> {:error, :unknown_dataset}
    end
  end

  @impl true
  def info(dataset_name) do
    case dataset_name do
      "scifact" -> %{
        name: "SciFact",
        description: "Scientific fact-checking claims with evidence",
        size: ~5_000,
        format: :jsonl,
        tasks: [:claim_extraction, :fact_checking]
      }
      "fever" -> %{
        name: "FEVER",
        description: "Fact Extraction and VERification",
        size: ~185_000,
        format: :jsonl,
        tasks: [:claim_extraction, :fact_checking]
      }
    end
  end

  defp load_scifact(opts) do
    claims_path = opts[:claims_path] || default_scifact_claims()
    corpus_path = opts[:corpus_path] || default_scifact_corpus()

    SciFact.convert(claims_path, corpus_path, opts)
  end

  defp load_fever(opts) do
    claims_path = opts[:claims_path] || default_fever_claims()
    wiki_dir = opts[:wiki_dir] || default_fever_wiki()

    FEVER.convert(claims_path, wiki_dir, opts)
  end
end
```

### 5.2 Module Organization

```
lib/
  cns/
    pipeline/
      converters/
        scifact.ex
        fever.ex
        csv.ex
      validation/
        validator.ex
        claim_validator.ex
        evidence_validator.ex
        relation_validator.ex
        embedding_matcher.ex
        quality_scorer.ex
      lineage/
        tracker.ex
        record.ex
        artifact.ex
        store.ex
      schema/
        claim_entry.ex
        relation.ex
        training_example.ex
        validation_error.ex
        validation_result.ex
        claim_parser.ex
```

### 5.3 Application Supervision

```elixir
defmodule CNS.Pipeline.Application do
  use Application

  @impl true
  def start(_type, _args) do
    children = [
      CNS.Pipeline.Lineage.Tracker
    ]

    opts = [strategy: :one_for_one, name: CNS.Pipeline.Supervisor]
    Supervisor.start_link(children, opts)
  end
end
```

---

## 6. Effort Estimation

### 6.1 Summary by Component

| Component | Effort (days) |
|-----------|---------------|
| Dataset Converters | 10 |
| Dataset Validation | 13 |
| Lineage Tracking | 6 |
| Data Schemas | 7 |
| Integration | 3 |
| Documentation | 2 |
| **Total** | **41 days** |

### 6.2 Priority Order

1. **Phase 1 - Foundation (Week 1-2)**: Data Schemas, ClaimParser
2. **Phase 2 - Core Pipeline (Week 3-4)**: SciFact Converter, Validation
3. **Phase 3 - Extended (Week 5-6)**: FEVER/CSV Converters, Lineage
4. **Phase 4 - Integration (Week 7)**: crucible_datasets integration, Embedding Matcher

### 6.3 Risk Factors

| Risk | Mitigation |
|------|------------|
| Bumblebee model loading | Pre-download models, cache aggressively |
| Large FEVER wiki files | Stream processing, memory-mapped files |
| Regex edge cases | Extensive property-based testing |
| Mnesia schema changes | Migration utilities |

### 6.4 Testing Strategy

- Unit tests for all public functions
- Property-based tests for parsers (StreamData)
- Integration tests with sample datasets
- Performance benchmarks for large files

---

## Appendix A: Complete Dependencies

```elixir
# mix.exs
defp deps do
  [
    # JSON
    {:jason, "~> 1.4"},

    # CSV
    {:nimble_csv, "~> 1.2"},

    # Validation
    {:ecto, "~> 3.11"},

    # ML/Embeddings
    {:nx, "~> 0.7"},
    {:exla, "~> 0.7"},  # or {:torchx, "~> 0.7"}
    {:bumblebee, "~> 0.5"},

    # DataFrames (optional)
    {:explorer, "~> 0.8"},

    # YAML config
    {:yaml_elixir, "~> 2.9"},

    # Testing
    {:stream_data, "~> 0.6", only: [:test, :dev]}
  ]
end
```

---

## Appendix B: Example Usage

```elixir
# Convert SciFact dataset
alias CNS.Pipeline.Converters.SciFact
alias CNS.Pipeline.Validation.Validator
alias CNS.Pipeline.Lineage.Tracker

# 1. Convert
SciFact.convert("claims_train.jsonl", "corpus.jsonl")
|> SciFact.to_jsonl("scifact_train.jsonl")

# 2. Record lineage
Tracker.record("scifact_train.jsonl")

# 3. Validate
result = Validator.validate("scifact_train.jsonl",
  claims_path: "claims_train.jsonl",
  corpus_path: "corpus.jsonl",
  embedding_mode: true
)

IO.puts("Valid: #{result.valid?}")
IO.puts("Errors: #{length(result.errors)}")
IO.puts("Total examples: #{result.total_examples}")

# 4. Export lineage
Tracker.export("lineage.json")
```

---

**Document Version:** 1.0.0
**Last Updated:** 2025-11-21

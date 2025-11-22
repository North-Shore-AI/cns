# Data Pipeline Guide

## Overview

The `CNS.Pipeline` modules handle dataset conversion, training data generation, and lineage tracking for reproducible ML workflows.

## Modules

- `CNS.Pipeline.Schema` - Data structures for training examples
- `CNS.Pipeline.Converters` - Dataset format converters

## Training Example Format

### The TrainingExample Struct

```elixir
%CNS.Pipeline.Schema.TrainingExample{
  prompt: String.t(),      # Input prompt for the model
  completion: String.t(),  # Expected output
  metadata: map()          # Source tracking, claim IDs, etc.
}
```

### Creating Training Examples

```elixir
alias CNS.Pipeline.Schema.TrainingExample

example = %TrainingExample{
  prompt: "Extract claims from: Evidence shows...",
  completion: "CLAIM[c1]: Main claim\nRELATION: c2 supports c1",
  metadata: %{source: "scifact", claim_id: "12345"}
}
```

### Serialization

```elixir
# To JSON string
json = TrainingExample.to_json(example)
# {"prompt":"...","completion":"...","metadata":{...}}

# From JSON string
{:ok, example} = TrainingExample.from_json(json)
```

## Dataset Conversion

### Converting SciFact Data

```elixir
alias CNS.Pipeline.Converters

# SciFact entry format
entry = %{
  "id" => 1,
  "claim" => "Continuous positive airway pressure improves obstructive sleep apnea",
  "evidence" => %{
    "12345" => [
      %{"label" => "SUPPORTS", "sentences" => [0, 1]}
    ]
  }
}

# Corpus with evidence documents
corpus = %{
  "12345" => %{
    sentences: [
      "CPAP therapy reduced apnea-hypopnea index by 90%.",
      "Patient outcomes improved significantly."
    ]
  }
}

# Convert to training example
example = Converters.parse_scifact_entry(entry, corpus)
```

### Output Format

Generated completions follow this structure:

```
CLAIM[c1]: Main hypothesis
CLAIM[c2]: Supporting evidence [DocID:SentIdx]
CLAIM[c3]: More evidence [DocID:SentIdx]
RELATION: c2 supports c1
RELATION: c3 supports c1
```

## Converter Functions

### Building Prompts

```elixir
passage = "Evidence sentence 1. Evidence sentence 2."
prompt = Converters.build_prompt(passage)

# Output:
# Extract all claims and relations from the following passage.
# Output format: CLAIM[c1]: main claim, CLAIM[c2..n]: evidence, RELATION: cx label cy
#
# Passage:
# Evidence sentence 1. Evidence sentence 2.
```

### Building Completions

```elixir
claim_text = "Main hypothesis"
evidence = [
  {"Supporting evidence text", "supports", "[123:0]"},
  {"Counter evidence text", "refutes", "[456:1]"}
]

completion = Converters.build_completion(claim_text, evidence)

# Output:
# CLAIM[c1]: Main hypothesis
# CLAIM[c2]: Supporting evidence text [123:0]
# CLAIM[c3]: Counter evidence text [456:1]
# RELATION: c2 supports c1
# RELATION: c3 refutes c1
```

### Label Normalization

```elixir
Converters.normalize_label("SUPPORTS")    # "supports"
Converters.normalize_label("SUPPORT")     # "supports"
Converters.normalize_label("REFUTES")     # "refutes"
Converters.normalize_label("CONTRADICT")  # "refutes"
Converters.normalize_label("NEUTRAL")     # "neutral"
```

### Evidence Gathering

```elixir
entry = %{
  "evidence" => %{
    "doc1" => [
      %{"label" => "SUPPORTS", "sentences" => [0, 2]}
    ]
  }
}

corpus = %{
  "doc1" => %{sentences: ["Sent 0", "Sent 1", "Sent 2"]}
}

evidence = Converters.gather_evidence(entry, corpus)
# [
#   {"Sent 0", "supports", "[doc1:0]"},
#   {"Sent 2", "supports", "[doc1:2]"}
# ]
```

### Checking for Evidence

```elixir
Converters.has_evidence?(%{"evidence" => %{"123" => [%{}]}})
# true

Converters.has_evidence?(%{"evidence" => %{}})
# false

Converters.has_evidence?(%{})
# false
```

## Lineage Tracking

### The Lineage Struct

```elixir
%CNS.Pipeline.Schema.Lineage{
  source_file: String.t(),         # Original file path
  timestamp: DateTime.t(),         # Creation time
  transformations: [String.t()],   # Applied transformations
  hash: String.t()                 # File content hash
}
```

### Creating Lineage Records

```elixir
alias CNS.Pipeline.Schema.Lineage

lineage = Lineage.new("data/scifact/claims.jsonl")
# %Lineage{
#   source_file: "data/scifact/claims.jsonl",
#   timestamp: ~U[2025-11-21 ...],
#   transformations: [],
#   hash: "a1b2c3d4e5f6..."
# }
```

### Tracking Transformations

```elixir
lineage = Lineage.new("input.jsonl")
  |> Lineage.add_transformation("load")
  |> Lineage.add_transformation("filter_empty")
  |> Lineage.add_transformation("convert_format")

lineage.transformations
# ["load", "filter_empty", "convert_format"]
```

## Complete Conversion Pipeline

```elixir
defmodule MyApp.DatasetConverter do
  alias CNS.Pipeline.{Converters, Schema}
  alias Schema.{TrainingExample, Lineage}

  def convert_scifact(claims_path, corpus_path, output_path) do
    # Load corpus
    corpus = load_corpus(corpus_path)

    # Create lineage
    lineage = Lineage.new(claims_path)
      |> Lineage.add_transformation("load_claims")

    # Stream convert
    claims_path
    |> File.stream!()
    |> Stream.map(&Jason.decode!/1)
    |> Stream.filter(&Converters.has_evidence?/1)
    |> Stream.map(&Converters.parse_scifact_entry(&1, corpus))
    |> Stream.map(&TrainingExample.to_json/1)
    |> Stream.intersperse("\n")
    |> Stream.into(File.stream!(output_path))
    |> Stream.run()

    # Return lineage
    lineage
    |> Lineage.add_transformation("filter_empty")
    |> Lineage.add_transformation("convert_to_jsonl")
  end

  defp load_corpus(path) do
    path
    |> File.stream!()
    |> Stream.map(&Jason.decode!/1)
    |> Enum.into(%{}, fn entry ->
      id = entry["doc_id"] || entry["id"]
      {to_string(id), %{sentences: entry["abstract"] || []}}
    end)
  end
end
```

## Working with JSONL Files

### Writing JSONL

```elixir
examples = [example1, example2, example3]

File.open!("output.jsonl", [:write], fn file ->
  Enum.each(examples, fn example ->
    IO.puts(file, TrainingExample.to_json(example))
  end)
end)
```

### Reading JSONL

```elixir
examples =
  "input.jsonl"
  |> File.stream!()
  |> Stream.map(&String.trim/1)
  |> Stream.reject(&(&1 == ""))
  |> Stream.map(fn line ->
    {:ok, example} = TrainingExample.from_json(line)
    example
  end)
  |> Enum.to_list()
```

## ClaimEntry Struct

Intermediate representation during conversion:

```elixir
%CNS.Pipeline.Schema.ClaimEntry{
  id: "c1",
  text: "Main claim text",
  evidence_ids: ["123", "456"],
  label: "SUPPORTS"
}
```

## Best Practices

1. **Stream large files** - Use `File.stream!` for memory efficiency
2. **Track lineage** - Record all transformations for reproducibility
3. **Validate early** - Filter invalid entries before conversion
4. **Hash source files** - Detect when source data changes
5. **Use consistent IDs** - Normalize document and claim IDs

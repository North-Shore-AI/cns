# API Reference

Complete reference for all CNS public modules and functions.

---

## CNS.Schema.Parser

Parse structured claims and relations from LLM output.

### Types

```elixir
@type relation :: {String.t(), String.t(), String.t()}
```

### Functions

#### `parse_claims/1`

```elixir
@spec parse_claims(String.t() | [String.t()]) :: %{String.t() => Claim.t()}
```

Parse all CLAIM lines from text. Returns map of `claim_id => Claim`.

#### `parse_relation/1`

```elixir
@spec parse_relation(String.t()) :: relation() | nil
```

Parse a single RELATION line. Returns `{source, label, target}` or `nil`.

#### `parse_relations/1`

```elixir
@spec parse_relations(String.t() | [String.t()]) :: [relation()]
```

Parse all relations from text.

#### `parse/1`

```elixir
@spec parse(String.t()) :: {%{String.t() => Claim.t()}, [relation()]}
```

Parse complete output into claims and relations.

---

## CNS.Logic.Betti

Compute Betti numbers and topology metrics.

### Types

```elixir
@type relation :: {String.t(), String.t(), String.t()}
```

### Functions

#### `compute_graph_stats/2`

```elixir
@spec compute_graph_stats([String.t()], [relation()]) :: GraphStats.t()
```

Compute graph topology statistics including Betti numbers.

#### `polarity_conflict?/2`

```elixir
@spec polarity_conflict?([relation()], String.t()) :: boolean()
```

Detect if a claim has conflicting polarity. Default target: `"c1"`.

#### `find_cycles/1`

```elixir
@spec find_cycles(Graph.t()) :: [[String.t()]]
```

Find all cycles in the reasoning graph. Limited to 100 cycles.

---

## CNS.Metrics.Chirality

Compute chirality scores using Fisher-Rao distance.

### Functions

#### `build_fisher_rao_stats/2`

```elixir
@spec build_fisher_rao_stats(Nx.Tensor.t() | [[number()]], float()) :: FisherRaoStats.t()
```

Build statistics from embedding vectors. Default epsilon: `1.0e-6`.

#### `fisher_rao_distance/3`

```elixir
@spec fisher_rao_distance(Nx.Tensor.t(), Nx.Tensor.t(), FisherRaoStats.t()) :: float()
```

Compute weighted distance using diagonal FIM approximation.

#### `compute_chirality_score/3`

```elixir
@spec compute_chirality_score(float(), float(), boolean()) :: float()
```

Compute composite chirality score from distance, overlap, and conflict.

#### `compare/6`

```elixir
@spec compare(module(), FisherRaoStats.t(), String.t(), String.t(), float(), boolean()) :: ChiralityResult.t()
```

Compare thesis and antithesis to compute chirality score.

---

## CNS.Validation.Semantic

4-stage semantic validation pipeline.

### Functions

#### `extract_document_ids/1`

```elixir
@spec extract_document_ids(String.t()) :: MapSet.t(String.t())
```

Extract document IDs from text using various patterns.

#### `validate_citations/3`

```elixir
@spec validate_citations(String.t(), map(), MapSet.t(String.t())) ::
  {boolean(), MapSet.t(String.t()), MapSet.t(String.t())}
```

Validate citations against corpus and gold evidence. Returns `{valid?, cited, missing}`.

#### `compute_similarity/2`

```elixir
@spec compute_similarity(String.t(), String.t()) :: float()
```

Compute text similarity using word overlap.

#### `validate_claim/6`

```elixir
@spec validate_claim(Config.t(), String.t(), String.t(), String.t(), map(), MapSet.t(String.t())) :: ValidationResult.t()
```

Validate a claim through the 4-stage pipeline.

#### `failed_result/3`

```elixir
@spec failed_result(boolean(), MapSet.t(String.t()), MapSet.t(String.t())) :: ValidationResult.t()
```

Create a failed validation result for early termination.

---

## CNS.Pipeline.Schema

Data structures for training pipelines.

### TrainingExample

#### `to_json/1`

```elixir
@spec to_json(TrainingExample.t()) :: String.t()
```

Convert training example to JSON string.

#### `from_json/1`

```elixir
@spec from_json(String.t()) :: {:ok, TrainingExample.t()} | {:error, term()}
```

Parse training example from JSON string.

### Lineage

#### `new/1`

```elixir
@spec new(String.t()) :: Lineage.t()
```

Create new lineage for a source file with hash.

#### `add_transformation/2`

```elixir
@spec add_transformation(Lineage.t(), String.t()) :: Lineage.t()
```

Add a transformation to the lineage history.

---

## CNS.Pipeline.Converters

Dataset format converters.

### Functions

#### `build_prompt/1`

```elixir
@spec build_prompt(String.t()) :: String.t()
```

Build extraction prompt with passage.

#### `build_completion/2`

```elixir
@spec build_completion(String.t(), [{String.t(), String.t(), String.t()}]) :: String.t()
```

Build completion from claim text and evidence list.

#### `normalize_label/1`

```elixir
@spec normalize_label(String.t()) :: String.t()
```

Normalize evidence label to lowercase standard form.

#### `parse_scifact_entry/2`

```elixir
@spec parse_scifact_entry(map(), corpus_map()) :: TrainingExample.t()
```

Parse a SciFact entry into a training example.

#### `gather_evidence/2`

```elixir
@spec gather_evidence(map(), corpus_map()) :: [{String.t(), String.t(), String.t()}]
```

Gather evidence sentences with labels from entry.

#### `has_evidence?/1`

```elixir
@spec has_evidence?(map()) :: boolean()
```

Check if entry has evidence.

---

## CNS.Training.Evaluation

Evaluation harness for model assessment.

### Functions

#### `compute_metrics/2`

```elixir
@spec compute_metrics([String.t()], [String.t()]) :: Metrics.t()
```

Compute basic metrics from predictions and gold labels.

#### `evaluate_claims/3`

```elixir
@spec evaluate_claims([String.t()], [String.t()], EvalConfig.t()) :: EvalResult.t()
```

Evaluate claim extraction predictions against gold standard.

#### `compute_f1/2`

```elixir
@spec compute_f1(float(), float()) :: float()
```

Compute F1 score from precision and recall.

#### `extract_claims_from_output/1`

```elixir
@spec extract_claims_from_output(String.t()) :: [String.t()]
```

Extract claim texts from formatted output.

#### `extract_relations_from_output/1`

```elixir
@spec extract_relations_from_output(String.t()) :: [{String.t(), String.t(), String.t()}]
```

Extract relations from formatted output.

#### `evaluate_detailed/2`

```elixir
@spec evaluate_detailed([String.t()], [String.t()]) :: map()
```

Evaluate with detailed metrics for claims and relations separately.

---

## Structs Reference

### CNS.Schema.Parser.Claim

```elixir
%Claim{
  identifier: String.t(),
  text: String.t(),
  document_id: String.t() | nil
}
```

### CNS.Logic.Betti.GraphStats

```elixir
%GraphStats{
  nodes: non_neg_integer(),
  edges: non_neg_integer(),
  components: non_neg_integer(),
  beta1: non_neg_integer(),
  cycles: [[String.t()]],
  polarity_conflict: boolean()
}
```

### CNS.Metrics.Chirality.FisherRaoStats

```elixir
%FisherRaoStats{
  mean: Nx.Tensor.t(),
  inv_var: Nx.Tensor.t()
}
```

### CNS.Metrics.Chirality.ChiralityResult

```elixir
%ChiralityResult{
  fisher_rao_distance: float(),
  evidence_overlap: float(),
  polarity_conflict: boolean(),
  chirality_score: float()
}
```

### CNS.Validation.Semantic.ValidationResult

```elixir
%ValidationResult{
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
```

### CNS.Validation.Semantic.Config

```elixir
%Config{
  entailment_threshold: float(),  # default: 0.75
  similarity_threshold: float()   # default: 0.7
}
```

### CNS.Pipeline.Schema.TrainingExample

```elixir
%TrainingExample{
  prompt: String.t(),
  completion: String.t(),
  metadata: map()
}
```

### CNS.Pipeline.Schema.Lineage

```elixir
%Lineage{
  source_file: String.t(),
  timestamp: DateTime.t(),
  transformations: [String.t()],
  hash: String.t()
}
```

### CNS.Training.Evaluation.Metrics

```elixir
%Metrics{
  precision: float(),
  recall: float(),
  f1: float(),
  accuracy: float()
}
```

### CNS.Training.Evaluation.EvalConfig

```elixir
%EvalConfig{
  batch_size: pos_integer(),  # default: 32
  max_samples: pos_integer() | nil
}
```

### CNS.Training.Evaluation.EvalResult

```elixir
%EvalResult{
  metrics: Metrics.t(),
  num_samples: non_neg_integer(),
  errors: [String.t()]
}
```

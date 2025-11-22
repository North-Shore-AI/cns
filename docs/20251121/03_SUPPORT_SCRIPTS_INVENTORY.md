# CNS Support Models Scripts Inventory

**Location:** `tinkerer/cns-support-models/scripts/`
**Date:** 2025-11-21
**Purpose:** Data pipeline scripts for claim extraction model training

---

## Overview

The `cns-support-models` repository contains Python scripts for processing scientific fact-checking datasets (SciFact, FEVER) into training formats for claim extraction models. These scripts form a complete ML data pipeline from raw datasets to trained LoRA adapters.

---

## Script Analysis

### 1. claim_schema.py

**Purpose:** Core schema definitions and parsing utilities for CLAIM/RELATION format

**Input/Output Formats:**
- **Input:** Raw text lines containing CLAIM[id]: text and RELATION: src label dst
- **Output:** Structured dictionaries with claim IDs and text

**Key Functions:**
- `parse_claim_lines(lines)` - Parse CLAIM[c#]: text lines into list of dicts
- `render_claim_lines(claims)` - Convert structured claims back to text
- `enforce_c1(claims, gold_text)` - Force CLAIM[c1] to match canonical text
- `parse_relation_line(line)` - Extract (src, label, dst) tuple from RELATION line
- `claims_to_dict(claims)` - Convert list to {id: text} mapping
- `replace_claim_text(claims, claim_id, new_text)` - Update claim text in place

**Data Transformations:**
- Regex-based parsing: `CLAIM_LINE_RE`, `RELATION_LINE_RE`
- Handles case variations and optional Document citations
- Normalizes claim IDs to lowercase

**Dependencies:**
- `re` (standard library)

**Elixir Equivalent Needs:**
- Regex parsing module for CLAIM/RELATION format
- Data structure for claims: `%{id: string, text: string}`
- Schema enforcement functions

---

### 2. csv_to_claim_jsonl.py

**Purpose:** Convert manually annotated CSV files to JSONL training format

**Input/Output Formats:**
- **Input:** CSV with columns: `passage/text`, `claims`, `relations`
  - Claims format: `CLAIM[c1]: ... | CLAIM[c2]: ...`
  - Relations format: `c2 supports c1; c3 refutes c1`
- **Output:** JSONL with `{prompt, completion, metadata}`

**Key Functions:**
- `parse_list(field, sep)` - Split and strip field by separator
- `main()` - CLI entry point with argparse

**Data Transformations:**
- CSV row -> prompt/completion pair
- Pipe-separated claims -> individual CLAIM lines
- Semicolon-separated relations -> RELATION: lines
- Metadata extraction (source, annotator)

**Dependencies:**
- `argparse`, `csv`, `json`, `pathlib` (standard library)

**Elixir Equivalent Needs:**
- CSV parsing with `NimbleCSV`
- JSONL streaming writer
- String splitting/trimming utilities

---

### 3. validate_dataset.py

**Purpose:** Validate claim-extractor JSONL datasets before training

**Input/Output Formats:**
- **Input:**
  - JSONL dataset file
  - Optional: claims JSONL, corpus JSONL for evidence validation
- **Output:** Validation errors or success message; optionally cleaned JSONL

**Key Functions:**
- `load_corpus_map(path)` - Load corpus.jsonl into {doc_id: sentences} mapping
- `load_claim_metadata(claims_path, corpus_map)` - Build claim_map and evidence_map
- `iter_dataset(path, max_examples)` - Iterate JSONL with optional limit
- `EmbeddingMatcher` class - Semantic similarity matching with sentence-transformers
- `validate_row(idx, row, claim_map, evidence_map, ...)` - Validate single row

**Data Transformations:**
- Schema validation: prompt/completion must be non-empty strings
- CLAIM line parsing and sequential ID validation
- Evidence matching: exact or embedding-based
- Gold claim text verification
- Relation reference validation

**Dependencies:**
- `argparse`, `json`, `math`, `sys`, `pathlib` (standard library)
- `sentence-transformers` (optional, for embedding mode)
- `claim_schema` (local)

**Elixir Equivalent Needs:**
- Streaming JSONL validation
- Embedding-based matching (Nx + SentenceTransformers via Bumblebee)
- Error accumulation and reporting
- Data quality metrics

---

### 4. convert_scifact.py

**Purpose:** Convert SciFact dataset to claim-extractor training format

**Input/Output Formats:**
- **Input:**
  - `claims_dev.jsonl` or `claims_train.jsonl`
  - `corpus.jsonl` with abstracts
- **Output:** JSONL with `{prompt, completion, metadata}`

**Key Functions:**
- `load_jsonl(path)` - Load all JSONL lines
- `normalize_label(label)` - Map SUPPORTS/REFUTES variants to canonical form
- `build_passage(documents)` - Build passage text with document IDs
- `gather_evidence(claim_entry)` - Normalize evidence format from SciFact
- `build_claim_completion(claim_text, evidence_texts)` - Generate CLAIM/RELATION output

**Data Transformations:**
- SciFact claims -> CLAIM[c1] main hypothesis
- Evidence sentences -> CLAIM[c2..n] with document citations
- Label normalization: SUPPORTS -> supports, REFUTES/CONTRADICT -> refutes
- Document ID extraction from multiple field names (doc_id, docid, id)
- Prompt construction with task instructions

**Dependencies:**
- `argparse`, `json`, `pathlib` (standard library)

**Elixir Equivalent Needs:**
- SciFact format parser
- Evidence gatherer with flexible schema handling
- Label normalization
- Prompt template generation

---

### 5. convert_fever.py

**Purpose:** Convert FEVER dataset to claim-extractor training format

**Input/Output Formats:**
- **Input:**
  - `fever.train.jsonl` - Claims with evidence annotations
  - Wiki pages directory (JSONL or TSV shards)
- **Output:** JSONL with `{prompt, completion, metadata}`

**Key Functions:**
- `load_wiki_sentences(wiki_dir)` - Build (page_id, sent_idx) -> text mapping
- `iter_claims(claims_path)` - Iterate FEVER claims
- `build_completion(claim, evidence_texts, label)` - Generate formatted output

**Data Transformations:**
- Wikipedia sentence lookup from multiple format types (JSONL, TSV)
- FEVER evidence sets: list of [_, _, page, sent_idx] tuples
- NOT ENOUGH INFO filtering (configurable)
- Label mapping for relations

**Dependencies:**
- `argparse`, `json`, `logging`, `pathlib`, `collections` (standard library)

**Elixir Equivalent Needs:**
- Multi-format file parser (JSONL/TSV)
- Large file streaming for wiki sentences
- Evidence set extraction
- Configurable label filtering

---

### 6. train_claim_extractor.py

**Purpose:** Train LoRA adapter for claim extraction using Tinker API

**Input/Output Formats:**
- **Input:**
  - YAML config file (data paths, model, optimization params)
  - JSONL training data
- **Output:**
  - Saved adapter weights
  - Provenance JSON logs
  - Periodic checkpoints

**Key Functions:**
- `sha256_file(path)` - Compute file hash for lineage
- `git_commit()` - Get current git commit
- `load_config(path)` - Parse YAML config
- `load_examples(path, limit)` - Load training examples
- `_build_claim_labels(completion_text, tokenizer)` - Map tokens to claim IDs
- `build_datum(example, tokenizer, citation_penalty_multiplier)` - Create Tinker Datum with weighted loss

**Data Transformations:**
- JSONL -> Example dataclass
- Token-level claim labeling for weighted loss
- Weight calculation: CLAIM_C1_WEIGHT=5.0, CLAIM_EVIDENCE_WEIGHT=2.0
- Citation validation penalty: CITATION_VALIDITY_WEIGHT=5.0
- Batching and shuffling

**Dependencies:**
- `argparse`, `datetime`, `hashlib`, `json`, `math`, `os`, `random`, `subprocess`, `pathlib` (standard library)
- `yaml` (PyYAML)
- `tinker` (Tinker API client)
- `claim_schema` (local)
- `citation_validation` (from thinker/)

**Elixir Equivalent Needs:**
- YAML config parsing
- Token-level loss weighting
- Training loop with batching
- Checkpoint management
- Provenance/lineage tracking
- GPU-accelerated training via Nx/Axon

---

### 7. eval_claim_extractor.py

**Purpose:** Sample from trained adapter and display/save results

**Input/Output Formats:**
- **Input:**
  - Optional prompt file (or default test prompt)
  - Adapter name and base model
- **Output:**
  - Console output of prompt/completion
  - Optional JSON file with results

**Key Functions:**
- `load_prompt(path)` - Load prompt from file or use default
- `main()` - Sample from adapter with force-c1 options

**Data Transformations:**
- Prompt tokenization
- Completion decoding
- CLAIM[c1] enforcement via enforce_c1()
- Output format specification appended to prompt

**Dependencies:**
- `argparse`, `json`, `pathlib` (standard library)
- `tinker` (Tinker API client)
- `claim_schema` (local)

**Elixir Equivalent Needs:**
- LLM inference client
- Response parsing and schema enforcement
- JSON output serialization

---

### 8. eval_scifact_dev.py

**Purpose:** Structured evaluation against SciFact dev set with metrics

**Input/Output Formats:**
- **Input:**
  - Training config YAML
  - SciFact corpus and claims
- **Output:**
  - Console metrics summary
  - Optional JSONL with detailed predictions

**Key Functions:**
- `load_corpus(path)` - Load corpus into {doc_id: entry} map
- `extract_passage(claim_entry, corpus)` - Build passage from cited docs
- `cleanup_completion(raw_text)` - Best-effort schema enforcement
- `parse_completion(text)` - Extract claims and relations
- `extract_gold_evidence(claim_entry, corpus)` - Get gold evidence with labels
- `fuzzy_similarity(a, b)` - SequenceMatcher-based text similarity
- `_normalize_schema_line(line, auto_idx)` - Normalize various claim formats

**Data Transformations:**
- Multiple claim format normalization (CLAIM[c#], CLAIM c#, C#:, numbered lists)
- Gold claim matching (exact and fuzzy)
- Evidence semantic matching
- Relation accuracy computation
- Schema cleanup with constrained decoding

**Metrics Computed:**
- claim_match_rate (exact)
- fuzzy_hits (similarity >= threshold)
- semantic_rate (evidence alignment)
- relation_accuracy (label correctness)
- Schema cleanup and fallback rates
- Timeout failures

**Dependencies:**
- `argparse`, `json`, `os`, `re`, `pathlib`, `difflib`, `concurrent.futures` (standard library)
- `yaml` (PyYAML)
- `tinker` (Tinker API client)
- `claim_schema` (local)

**Elixir Equivalent Needs:**
- Evaluation pipeline
- Multiple similarity metrics
- Schema normalization/constrained decoding
- Metrics aggregation
- Async sampling with timeout handling

---

### 9. record_lineage.py

**Purpose:** Record SHA-256 hashes for dataset artifact tracking

**Input/Output Formats:**
- **Input:** List of file paths
- **Output:** JSON with {path: {sha256, bytes}}

**Key Functions:**
- `sha256(path)` - Compute SHA-256 hash
- `main()` - CLI with file list and output path

**Data Transformations:**
- File -> SHA-256 hash string
- File -> size in bytes
- Multiple files -> single lineage JSON

**Dependencies:**
- `argparse`, `hashlib`, `json`, `pathlib` (standard library)

**Elixir Equivalent Needs:**
- :crypto.hash for SHA-256
- File.stat for size
- JSON encoding

---

## Test Suite Analysis

### conftest.py
- Provides pytest fixtures for SciFact sample files
- Session-scoped dataset generation via converter script

### test_claim_schema.py
- Tests claim parsing with various formats
- Tests relation parsing and regex validation
- Tests enforce_c1 insertion and update

### test_data_quality.py
- Tests c1 matches source claim
- Tests relation references are valid
- Tests sequential claim IDs
- Tests embedding mode semantic matching

### test_scifact_conversion.py
- Tests label normalization
- Tests CLI generates expected completion format
- Verifies document citations in output

---

## Data Pipeline Flow

```
Raw Data Sources
    |
    +-- SciFact (claims_*.jsonl + corpus.jsonl)
    |       |
    |       v
    |   convert_scifact.py
    |
    +-- FEVER (fever.*.jsonl + wiki-pages/)
    |       |
    |       v
    |   convert_fever.py
    |
    +-- Manual Annotations (CSV)
            |
            v
        csv_to_claim_jsonl.py
            |
            v
    Processed JSONL (prompt/completion/metadata)
            |
            v
    validate_dataset.py (quality check)
            |
            v
    train_claim_extractor.py (LoRA training)
            |
            v
    Trained Adapter + Provenance Logs
            |
            v
    eval_*.py (evaluation)
            |
            v
    record_lineage.py (artifact tracking)
```

---

## Elixir Port Priority Matrix

| Script | Priority | Complexity | Notes |
|--------|----------|------------|-------|
| claim_schema.py | **High** | Low | Core parsing - needed by all others |
| validate_dataset.py | **High** | Medium | Data quality is critical |
| convert_scifact.py | **High** | Medium | Primary dataset converter |
| convert_fever.py | Medium | Medium | Secondary dataset |
| csv_to_claim_jsonl.py | Medium | Low | Manual data ingestion |
| record_lineage.py | Medium | Low | Simple hashing utility |
| train_claim_extractor.py | Low | High | Requires Tinker API equivalent (Nx/Axon) |
| eval_claim_extractor.py | Low | Medium | Requires inference client |
| eval_scifact_dev.py | Low | High | Complex evaluation metrics |

---

## Key Data Structures for Elixir

### Claim Entry
```elixir
%ClaimEntry{
  id: String.t(),
  text: String.t()
}
```

### Training Example
```elixir
%TrainingExample{
  prompt: String.t(),
  completion: String.t(),
  metadata: map()
}
```

### Validation Error
```elixir
%ValidationError{
  line: integer(),
  message: String.t()
}
```

### Lineage Record
```elixir
%LineageRecord{
  path: String.t(),
  sha256: String.t(),
  bytes: integer()
}
```

---

## Dependencies Summary

### Python Standard Library
- argparse, csv, json, pathlib, re, hashlib, os, sys, math, logging
- difflib, datetime, random, subprocess, collections, concurrent.futures

### External Python
- `yaml` (PyYAML) - Config parsing
- `tinker` - LLM training/inference API
- `sentence-transformers` (optional) - Embedding similarity

### Elixir Equivalents
- YAML: `:yaml_elixir`
- JSON: `Jason`
- CSV: `NimbleCSV`
- Regex: Built-in `Regex` module
- Hashing: `:crypto`
- ML/Training: `Nx`, `Axon`, `Bumblebee`

---

## Summary

The CNS support models scripts implement a complete ML data pipeline for claim extraction model training. The pipeline handles:

1. **Data Ingestion** - Multiple source formats (SciFact, FEVER, CSV)
2. **Schema Standardization** - CLAIM/RELATION format with sequential IDs
3. **Validation** - Schema conformance, evidence matching, quality metrics
4. **Training** - LoRA fine-tuning with weighted loss for claim emphasis
5. **Evaluation** - Structured metrics including fuzzy matching and semantic alignment
6. **Lineage** - Artifact tracking with SHA-256 hashes

For Elixir porting, prioritize:
1. Core schema parsing (claim_schema.py)
2. Validation infrastructure (validate_dataset.py)
3. SciFact converter (convert_scifact.py)

These provide the foundation for data quality and format standardization that other components depend on.

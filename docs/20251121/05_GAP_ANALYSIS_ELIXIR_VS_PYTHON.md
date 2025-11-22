# Gap Analysis: Elixir CNS vs Python Thinker

**Generated:** 2025-11-21
**Purpose:** Identify implementation gaps between Elixir CNS and Python Thinker for porting prioritization

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Fully Implemented in Elixir](#1-fully-implemented-in-elixir)
3. [Partially Implemented](#2-partially-implemented)
4. [Not Implemented](#3-not-implemented)
5. [Elixir-Only Features](#4-elixir-only-features)
6. [Detailed Gap Analysis by Module](#detailed-gap-analysis-by-module)
7. [Porting Recommendations](#porting-recommendations)

---

## Executive Summary

| Category | Count | Notes |
|----------|-------|-------|
| Fully Implemented | 8 | Core graph/topology, basic metrics, data structures |
| Partially Implemented | 5 | Missing ML models, advanced algorithms |
| Not Implemented | 9 | Semantic validation, training backends, CLI |
| Elixir-Only | 6 | Critics system, OTP patterns, Crucible contracts |

**Critical Gaps:**
- No 4-stage semantic validation (NLI + embedding models)
- No Fisher-Rao chirality with embedding support
- No PEFT/LoRA training backend integration
- No claim/relation parsing for evaluation

---

## 1. Fully Implemented in Elixir

### 1.1 Graph Topology (betti.py → topology.ex)

| Feature | Python | Elixir | Status |
|---------|--------|--------|--------|
| Betti number computation | `compute_graph_stats()` | `betti_numbers/1` | **Complete** |
| Cycle detection | `nx.simple_cycles` | `detect_cycles/1` | **Complete** |
| Connected components | Via NetworkX | `connectivity/1` | **Complete** |
| DAG validation | Implicit | `is_dag?/1` | **Complete** |
| Topological sort | Not implemented | `topological_sort/1` | **Enhanced** |

**Notes:** Elixir implementation is more comprehensive with additional functions like `find_roots/1`, `find_leaves/1`, `all_paths/3`, and `depth/1`.

### 1.2 Basic Metrics (metrics.ex)

| Feature | Python | Elixir | Status |
|---------|--------|--------|--------|
| Schema compliance | `schema_compliance_rate` | `schema_compliance/1` | **Complete** |
| Citation accuracy | `citation_accuracy_rate` | `citation_accuracy/1` | **Complete** |
| Pass rate | Various | `pass_rate/2` | **Complete** |
| Convergence delta | Implicit | `convergence_delta/2` | **Complete** |
| Quality scoring | Multiple metrics | `quality_score/1` | **Complete** |

### 1.3 Data Structures

| Python Dataclass | Elixir Struct | Status |
|-----------------|---------------|--------|
| N/A | `CNS.SNO` | **Elixir-only** (more comprehensive) |
| N/A | `CNS.Evidence` | **Elixir-only** |
| N/A | `CNS.Challenge` | **Elixir-only** |
| N/A | `CNS.Provenance` | **Elixir-only** |
| `Claim` | Embedded in SNO | **Complete** |

### 1.4 Graph Building and Traversal

| Feature | Elixir Module | Status |
|---------|--------------|--------|
| Graph construction | `CNS.Graph.Builder` | **Complete** |
| Path finding | `CNS.Graph.Traversal` | **Complete** |
| Visualization (DOT/Mermaid) | `CNS.Graph.Visualization` | **Complete** |

---

## 2. Partially Implemented

### 2.1 Chirality Metrics (chirality.py → metrics.ex)

| Python Feature | Python Function | Elixir Equivalent | What's Missing |
|----------------|-----------------|-------------------|----------------|
| Fisher-Rao distance | `fisher_rao_distance()` | `fisher_rao_distance/2` | **Missing:** Inverse variance computation, embedding support |
| Fisher-Rao stats | `build_fisher_rao_stats()` | None | **Not implemented** |
| Chirality score | `ChiralityAnalyzer.compare()` | `chirality/1` | **Missing:** Embedding-based semantic distance, composite scoring formula |
| Evidence overlap | `evidence_overlap` param | Implicit | **Missing:** Explicit calculation |

**Porting Complexity:** **High**

**What's Missing:**
- `FisherRaoStats` dataclass with mean/inv_var
- Embedder integration (SentenceTransformer equivalent)
- Normalized chirality score formula:
  ```python
  chirality_score = min(1.0, norm_distance * 0.6 + overlap_factor * 0.2 + conflict_penalty)
  ```

**Dependencies Needed:**
- Nx.Serving for embeddings or external API
- Numerical computing for variance calculations

---

### 2.2 Antagonist (antagonist.py → antagonist.ex)

| Python Feature | Elixir Equivalent | What's Missing |
|----------------|-------------------|----------------|
| `AntagonistConfig` | `Config.t()` | **Missing:** `chirality_threshold`, `high_chirality_threshold`, `entailment_threshold`, `evidence_overlap_threshold` |
| `AntagonistRunner.run()` | `process/2` | **Missing:** Artifact file I/O, severity escalation logic |
| Issue types | `challenge_type` | **Missing:** `CITATION_INVALID`, `POLARITY_CONTRADICTION`, `WEAK_ENTAILMENT` as distinct types |
| Severity levels | `severity` | **Partial:** Has high/medium/low but missing escalation rules |
| Flag telemetry | Not implemented | **Missing:** Complete telemetry output format |

**Porting Complexity:** **Medium**

**What's Missing:**
- File-based artifact processing
- Severity escalation logic:
  - CITATION_INVALID -> HIGH
  - POLARITY_CONFLICT -> HIGH
  - High chirality + evidence overlap -> HIGH
  - POLARITY_CONTRADICTION -> MEDIUM
  - WEAK_ENTAILMENT -> MEDIUM
- Summary output with `flag_rate`, `severity_breakdown`, `issue_breakdown`

---

### 2.3 Citation Validation (citation_validation.py)

| Python Feature | Elixir Equivalent | What's Missing |
|----------------|-------------------|----------------|
| `extract_document_ids()` | Implicit in proposer | **Missing:** Standalone function |
| `validate_citations()` | `CNS.Critics.Grounding` | **Partial:** Basic validation exists |
| `CitationValidationResult` | None | **Missing:** Distinct result type with `hallucination_count` |
| `compute_citation_penalty()` | None | **Not implemented** |
| `citation_validation_stats()` | None | **Not implemented** |

**Porting Complexity:** **Low**

**What's Missing:**
- Explicit `CitationValidationResult` struct
- Penalty computation for training
- Batch validation with statistics

---

### 2.4 Claim Schema Parsing (claim_schema.py)

| Python Feature | Elixir Equivalent | What's Missing |
|----------------|-------------------|----------------|
| `CLAIM_LINE_RE` | None | **Not implemented** |
| `RELATION_LINE_RE` | None | **Not implemented** |
| `parse_claim_lines()` | None | **Not implemented** |
| `parse_relation_line()` | None | **Not implemented** |

**Porting Complexity:** **Low**

**What's Missing:**
- Regex patterns for CLAIM[id] and RELATION parsing
- Functions to extract structured data from LLM completions

---

### 2.5 Configuration (config.py → config.ex)

| Python Config | Elixir Equivalent | What's Missing |
|---------------|-------------------|----------------|
| `PipelineConfig` | `CNS.Config` | **Partial:** Missing many fields |
| `EvaluationConfig` | None | **Not implemented** |
| `LocalTrainingConfig` | `lora_config` (in Training) | **Partial:** Missing backend selection |
| `DatasetValidationConfig` | None | **Not implemented** |
| `TestSuiteConfig` | None | **Not implemented** |
| `SchemaField` | None | **Not implemented** |

**Porting Complexity:** **Medium**

**What's Missing:**
- `EvaluationConfig` with checkpoint_dir, max_samples, backend selection
- `DatasetValidationConfig` with evidence_mode, embedding_model, similarity_threshold
- YAML config loading (`load_pipeline_config`)

---

## 3. Not Implemented

### 3.1 Semantic Validation (semantic_validation.py)

**Python Feature:** 4-stage semantic validation pipeline

| Component | Description | Dependencies |
|-----------|-------------|--------------|
| Stage 1: Citation Accuracy | Extract and verify document IDs | Regex |
| Stage 2: Entailment | DeBERTa-v3-NLI scoring | `transformers`, `torch` |
| Stage 3: Semantic Similarity | MiniLM embedding cosine | `sentence-transformers` |
| Stage 4: Paraphrase Tolerance | Accept valid rephrasings | Combined |

**Elixir Equivalent:** None (Critics system is different approach)

**Porting Complexity:** **High**

**Dependencies Needed:**
- Bumblebee for DeBERTa-v3-NLI
- Nx.Serving for sentence embeddings
- Or: Python interop via Port

**Data Structure to Port:**
```python
@dataclass
class ValidationResult:
    citation_valid: bool
    cited_ids: Set[str]
    missing_ids: Set[str]
    entailment_score: float
    entailment_pass: bool
    semantic_similarity: float
    similarity_pass: bool
    paraphrase_accepted: bool
    overall_pass: bool
    schema_valid: bool
    schema_errors: List[str]
```

---

### 3.2 Evaluation Harness (evaluation.py)

**Python Feature:** Complete evaluation pipeline with metrics computation

| Component | Description |
|-----------|-------------|
| `Evaluator` class | Orchestrates evaluation |
| `evaluate_semantic_match()` | Semantic matching |
| `build_evaluation_series()` | Time series for dashboards |
| Backend support | hf_peft, tinker |

**Elixir Equivalent:** None

**Porting Complexity:** **High**

**Dependencies Needed:**
- Model loading (PEFT, base models)
- Inference execution
- Metrics aggregation

**Key Metrics to Implement:**
- `schema_compliance_rate`
- `citation_accuracy_rate`
- `mean_entailment_score`
- `entailment_pass_rate`
- `mean_semantic_similarity`
- `semantic_similarity_rate`
- `paraphrase_acceptance_rate`
- `overall_pass_rate`
- `mean_beta1`, `mean_chirality_score`

---

### 3.3 Training Backends (training.py)

**Python Features:**

| Component | Description |
|-----------|-------------|
| `LocalPEFTTrainer` | HuggingFace PEFT/LoRA training |
| `TinkerScriptTrainer` | Tinker API integration |
| `CitationAwareTrainer` | Custom loss with citation penalty |
| `CitationAwareDataCollator` | Data collation with validation |

**Elixir Equivalent:** `CNS.Training` (stubbed - returns `:tinkex_not_available`)

**Porting Complexity:** **Very High**

**Dependencies Needed:**
- Axon for native training (limited LoRA support)
- Or: Tinkex integration (preferred)
- Or: Python subprocess for PEFT

**Key Features Missing:**
- 4-bit quantization (BitsAndBytes)
- LoRA config application
- Citation-aware loss computation
- Checkpoint saving/loading with PEFT

---

### 3.4 Metrics Emitter (metrics/emitter.py)

**Python Feature:** Structured metrics persistence for dashboards

| Component | Description |
|-----------|-------------|
| `MetricsEmitter` | Write metrics to disk |
| Atomic writes | .tmp → rename pattern |
| Index maintenance | `dashboard_data/index.json` |
| Deduplication | By run_id |

**Elixir Equivalent:** None (could use `crucible_telemetry`)

**Porting Complexity:** **Low**

**What's Missing:**
- File-based metrics persistence
- Dashboard index maintenance
- Run tracking

---

### 3.5 Pipeline Orchestration (pipeline.py)

**Python Feature:** High-level workflow orchestration

| Component | Elixir Equivalent | Status |
|-----------|-------------------|--------|
| `ThinkerPipeline` | `CNS.Pipeline` | **Different purpose** |
| `PipelineState` | None | **Not implemented** |
| `validate()` | None | **Not implemented** |
| `train()` | Stubbed | **Not implemented** |
| `evaluate()` | None | **Not implemented** |
| `run()` | `CNS.Pipeline.run/2` | **Different scope** |

**Porting Complexity:** **Medium**

**Note:** Elixir `CNS.Pipeline` is for dialectical reasoning (Proposer → Antagonist → Synthesizer), not for training orchestration.

---

### 3.6 Data Utilities (data.py)

**Python Feature:** Dataset download and preparation

| Component | Description |
|-----------|-------------|
| SciFact download | Scientific fact verification |
| FEVER download | Fact extraction dataset |
| Format conversion | To JSONL training format |

**Elixir Equivalent:** None (could leverage `crucible_datasets`)

**Porting Complexity:** **Low-Medium**

---

### 3.7 Validation Utilities (validation.py)

**Python Features:**

| Component | Description |
|-----------|-------------|
| `TestSuiteRunner` | Pytest execution |
| `DatasetValidator` | Schema validation |
| `DatasetValidationResult` | Validation output |

**Elixir Equivalent:** None

**Porting Complexity:** **Low**

---

### 3.8 CLI Interface (cli.py)

**Python Feature:** Command-line interface

| Command | Description |
|---------|-------------|
| `info` | Environment details |
| `manifest` | Adapter manifest |
| `validate` | Run tests + validation |
| `train` | Training execution |
| `eval` | Checkpoint evaluation |
| `run` | Full pipeline |
| `antagonist` | Quality flagging |
| `data setup` | Dataset preparation |

**Elixir Equivalent:** None

**Porting Complexity:** **Medium**

**Options:**
- Mix tasks
- Burrito for CLI binary
- Escripts

---

### 3.9 Polarity Conflict Detection

**Python Feature:** In `betti.py`

```python
def _determine_polarity_conflict(relations: Sequence[Relation], target: str = "c1") -> bool
```

**Elixir Equivalent:** Not explicitly implemented

**Porting Complexity:** **Low**

**What's Missing:**
- Check if same claim receives both "supports" and "refutes" edges

---

## 4. Elixir-Only Features

### 4.1 Critics System

Complete multi-critic evaluation framework not in Python:

| Critic | Weight | Purpose |
|--------|--------|---------|
| `Grounding` | 0.4 | Factual accuracy, evidence quality |
| `Logic` | 0.3 | Logical consistency, circular reasoning |
| `Novelty` | 0.15 | Originality, information density |
| `Causal` | 0.1 | Causal validity |
| `Bias` | 0.05 | Fairness, loaded language |

**Python Equivalent:** Partial overlap with `semantic_validation.py` and `antagonist.py`

### 4.2 Three-Agent Architecture

| Agent | Purpose |
|-------|---------|
| `CNS.Proposer` | Initial claim generation |
| `CNS.Antagonist` | Challenge generation |
| `CNS.Synthesizer` | Conflict resolution |

**Python Equivalent:** `antagonist.py` is post-hoc flagging, not dialectical synthesis

### 4.3 Crucible Framework Contracts

Behaviour contracts for integration:

- `CrucibleFramework.Datasets`
- `CrucibleFramework.Sampling`
- `CrucibleFramework.Ensemble.ML`
- `CrucibleFramework.Lora`

**Python Equivalent:** Direct library imports (transformers, peft)

### 4.4 OTP Patterns

- GenServer-based critics
- Supervision trees
- Async pipeline execution (`run_async/2`)

### 4.5 Comprehensive Data Structures

| Struct | Features |
|--------|----------|
| `CNS.SNO` | Children, synthesis_history, quality_score |
| `CNS.Provenance` | Origin tracking, transformation history |
| `CNS.Challenge` | Resolution status, chirality_score |

### 4.6 Graph Visualization

- DOT export
- Mermaid diagrams
- Text representation

---

## Detailed Gap Analysis by Module

### betti.py vs topology.ex

| Feature | betti.py | topology.ex | Gap |
|---------|----------|-------------|-----|
| `GraphStats` dataclass | Yes | No (returns maps) | Minor - use maps |
| Polarity conflict | Yes | No | **Needs implementation** |
| Betti numbers | β₁ only | b0 and b1 | **Elixir is better** |
| Cycle detection | nx.simple_cycles | Custom DFS | Equivalent |
| Normalization | `_normalize_claim_id` | No | Minor |

**Action Items:**
1. Add `polarity_conflict?/2` function to topology.ex
2. Consider adding `GraphStats` struct for consistency

---

### chirality.py vs metrics.ex

| Feature | chirality.py | metrics.ex | Gap |
|---------|-------------|------------|-----|
| `FisherRaoStats` | Yes | No | **Critical gap** |
| `ChiralityResult` | Yes | No | **Critical gap** |
| `ChiralityAnalyzer` | Class with embedder | No | **Critical gap** |
| `build_fisher_rao_stats` | Yes | No | **Critical gap** |
| `fisher_rao_distance` | Mahalanobis-style | Simple L2 | **Needs ML integration** |
| Chirality formula | Composite (0.6/0.2/0.25) | Simple | **Needs update** |

**Action Items:**
1. Create `CNS.Chirality` module
2. Add `FisherRaoStats` and `ChiralityResult` structs
3. Implement embedding-based distance (requires Nx.Serving)
4. Port composite chirality score formula

---

### antagonist.py vs antagonist.ex

| Feature | antagonist.py | antagonist.ex | Gap |
|---------|--------------|---------------|-----|
| Config thresholds | chirality, entailment, overlap | None | **Needs config extension** |
| Artifact file I/O | Yes | No | **Needs implementation** |
| Issue types | 4 specific types | Generic challenge_type | **Needs mapping** |
| Severity escalation | Rule-based | Simple | **Needs rules** |
| Summary output | Detailed stats | Basic | **Needs enhancement** |

**Action Items:**
1. Add threshold fields to Config
2. Implement file-based artifact processing
3. Add severity escalation rules
4. Enhance summary output format

---

### semantic_validation.py vs Critics System

| Feature | semantic_validation.py | Critics | Gap |
|---------|----------------------|---------|-----|
| Citation validation | Stage 1 | Grounding critic | Partial |
| NLI entailment | Stage 2 (DeBERTa) | None | **Critical gap** |
| Semantic similarity | Stage 3 (MiniLM) | None | **Critical gap** |
| Paraphrase tolerance | Stage 4 | None | **Not implemented** |
| 4-stage pipeline | Yes | No | **Architectural difference** |

**Action Items:**
1. Create `CNS.SemanticValidator` module
2. Integrate Bumblebee for NLI model
3. Add sentence embedding similarity
4. Implement 4-stage pipeline pattern

---

## Porting Recommendations

### Phase 1: Low-Hanging Fruit (Low Complexity)

1. **Claim Schema Parsing** - Pure regex, easy port
   - Create `CNS.ClaimParser` module
   - Port `CLAIM_LINE_RE` and `RELATION_LINE_RE`
   - Add `parse_claim_lines/1` and `parse_relation_line/1`

2. **Citation Validation Enhancement** - Extend existing
   - Add `CitationValidationResult` struct
   - Implement `compute_citation_penalty/2`
   - Add batch validation with stats

3. **Polarity Conflict Detection** - Simple graph check
   - Add to topology.ex
   - Check for supports+refutes on same target

4. **Metrics Emitter** - File I/O only
   - Create `CNS.MetricsEmitter` module
   - Atomic writes with rename
   - Index maintenance

### Phase 2: Core Algorithms (Medium Complexity)

5. **Antagonist Enhancement**
   - Add threshold configs
   - Implement severity escalation rules
   - Add artifact file processing

6. **Configuration Extension**
   - Port `EvaluationConfig`
   - Port `DatasetValidationConfig`
   - Add YAML loading support

7. **Dataset Utilities**
   - Integrate with `crucible_datasets`
   - Add SciFact/FEVER support

### Phase 3: ML Integration (High Complexity)

8. **Chirality with Embeddings**
   - Create `CNS.Chirality` module
   - Integrate Nx.Serving for embeddings
   - Implement Fisher-Rao with inverse variance

9. **Semantic Validation Pipeline**
   - Create `CNS.SemanticValidator`
   - Integrate Bumblebee for DeBERTa-NLI
   - Implement 4-stage validation

10. **Evaluation Harness**
    - Port `Evaluator` class
    - Add backend support
    - Implement metrics aggregation

### Phase 4: Training Integration (Very High Complexity)

11. **Training Backends**
    - Complete Tinkex integration
    - Or: Python interop for PEFT
    - Citation-aware training loss

12. **CLI Interface**
    - Mix tasks for commands
    - Or: Burrito for standalone binary

---

## Dependency Matrix

| Gap | Nx | Bumblebee | Tinkex | Python Interop |
|-----|----|-----------|---------|--------------|
| Chirality Embeddings | Yes | Optional | No | Alternative |
| NLI Entailment | Yes | Yes | No | Alternative |
| Semantic Similarity | Yes | Yes | No | Alternative |
| LoRA Training | No | No | Yes | Alternative |
| PEFT Training | No | No | No | Required |

---

## Summary

**Total Gaps Identified:** 23 features across 9 modules

**Critical Priorities:**
1. Chirality with embeddings (core metric)
2. Semantic validation 4-stage pipeline (evaluation)
3. Claim schema parsing (data processing)

**Quick Wins:**
1. Citation validation enhancement
2. Polarity conflict detection
3. Metrics emitter

**Architectural Decision Needed:**
- Native Elixir ML (Nx/Bumblebee) vs Python interop for NLI/embeddings

---

*End of Gap Analysis Report*

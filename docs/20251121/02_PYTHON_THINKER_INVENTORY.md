# Python Thinker Project - Complete Module Inventory

**Date:** 2025-11-21
**Source:** `/home/home/p/g/North-Shore-AI/tinkerer/thinker`
**Version:** 0.1.0

This document provides a comprehensive inventory of all Python modules in the Thinker project, intended as the source of truth for porting to Elixir/CNS.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Module Inventory](#module-inventory)
   - [logic/betti.py](#logicbettipy)
   - [metrics/chirality.py](#metricschiralitypy)
   - [metrics/emitter.py](#metricsemitterpy)
   - [antagonist.py](#antagonistpy)
   - [semantic_validation.py](#semantic_validationpy)
   - [citation_validation.py](#citation_validationpy)
   - [evaluation.py](#evaluationpy)
   - [training.py](#trainingpy)
   - [data.py](#datapy)
   - [cli.py](#clipy)
   - [config.py](#configpy)
   - [pipeline.py](#pipelinepy)
   - [claim_schema.py](#claim_schemapy)
   - [validation.py](#validationpy)
3. [External Dependencies Summary](#external-dependencies-summary)
4. [Data Flow Architecture](#data-flow-architecture)

---

## Project Overview

Thinker is a TDD-first training pipeline orchestrator for CNS (Cognitive Neural Systems) support models. It provides:

- **4-stage semantic validation** for claim extraction
- **Chirality metrics** using Fisher-Rao distance
- **Graph topology analysis** (Betti numbers)
- **Antagonist agent** for quality flagging
- **Training backends** (HuggingFace PEFT, Tinker)
- **CLI interface** for pipeline orchestration

---

## Module Inventory

---

### logic/betti.py

**Purpose:** Compute Betti numbers and cycle diagnostics for CLAIM/RELATION graphs to detect logical inconsistencies.

#### Data Structures

```python
Relation = Tuple[str, str, str]  # (source_id, label, target_id)

@dataclass
class GraphStats:
    nodes: int
    edges: int
    components: int
    beta1: int  # First Betti number (cycle count)
    cycles: List[List[str]]
    polarity_conflict: bool
```

#### Public Functions

```python
def compute_graph_stats(claim_ids: Iterable[str], relations: Sequence[Relation]) -> GraphStats
```

Builds a reasoning graph and computes topology metrics:
- Number of nodes, edges, connected components
- Beta1 (first Betti number) = edges - nodes + components
- Detects cycles using `nx.simple_cycles`
- Detects polarity conflicts (same claim has both supports and refutes)

#### Internal Functions

```python
def _normalize_claim_id(identifier: str) -> str
def _determine_polarity_conflict(relations: Sequence[Relation], target: str = "c1") -> bool
```

#### External Dependencies

- `networkx` (nx) - Graph construction and cycle detection

#### Key Algorithms

- **Polarity Conflict Detection:** Checks if a claim receives both "supports" and "refutes" edges
- **Betti Number Computation:** β₁ = E - V + C (edges - vertices + components)
- **Cycle Detection:** Uses NetworkX `simple_cycles` on directed graph

---

### metrics/chirality.py

**Purpose:** Compute chirality scores using Fisher-Rao distance approximation for measuring semantic divergence between thesis/antithesis pairs.

#### Data Structures

```python
@dataclass
class FisherRaoStats:
    mean: np.ndarray
    inv_var: np.ndarray  # Inverse variance for diagonal approximation

@dataclass
class ChiralityResult:
    fisher_rao_distance: float
    evidence_overlap: float
    polarity_conflict: bool
    chirality_score: float  # Normalized composite score [0, 1]
```

#### Public Functions

```python
def build_fisher_rao_stats(vectors: Sequence[Sequence[float]] | np.ndarray, epsilon: float = 1e-6) -> FisherRaoStats
def fisher_rao_distance(vec_a: np.ndarray, vec_b: np.ndarray, stats: FisherRaoStats) -> float
```

#### Classes

```python
class ChiralityAnalyzer:
    def __init__(self, embedder, stats: FisherRaoStats)
    def compare(
        self,
        thesis: str,
        antithesis: str,
        *,
        evidence_overlap: float,
        polarity_conflict: bool,
    ) -> ChiralityResult
```

#### External Dependencies

- `numpy` (np) - Numerical computations
- Requires an `embedder` object with `.encode([text], convert_to_numpy=True)` method (e.g., SentenceTransformer)

#### Key Algorithms

- **Fisher-Rao Distance:** Mahalanobis-style distance using diagonal Fisher information
  ```
  distance = sqrt((diff * inv_var) . diff)
  ```
- **Chirality Score Computation:**
  ```python
  norm_distance = distance / (distance + 1.0)  # Normalize to [0, 1]
  overlap_factor = 1.0 - clamp(evidence_overlap, 0, 1)
  conflict_penalty = 0.25 if polarity_conflict else 0.0
  chirality_score = min(1.0, norm_distance * 0.6 + overlap_factor * 0.2 + conflict_penalty)
  ```

---

### metrics/emitter.py

**Purpose:** Write structured stage metrics to disk for pipeline tracking and dashboards.

#### Data Structures

```python
@dataclass
class MetricsEmitter:
    project_root: Path | None = None
    artifacts_dir: str = "artifacts"
    dashboard_dir: str = "dashboard_data"
```

#### Public Functions

```python
def emit(stage: str, payload: Dict[str, Any], run_id: Optional[str] = None, *, project_root: Path | None = None) -> Dict[str, Any]
```

#### Class Methods

```python
class MetricsEmitter:
    def emit(self, stage: str, payload: Dict[str, Any], run_id: Optional[str] = None) -> Dict[str, Any]
```

Returns:
```python
{
    "manifest_path": Path,
    "index_entry": {
        "stage": str,
        "run_id": str,
        "timestamp": str,  # ISO 8601 UTC
        "manifest_path": str,
    }
}
```

#### External Dependencies

- `json` - JSON serialization
- `pathlib` - File path handling
- `datetime` - Timestamps

#### Key Features

- Atomic writes (write to .tmp then rename)
- Auto-creates directories
- Maintains a sorted index (`dashboard_data/index.json`)
- Deduplicates entries by run_id

---

### antagonist.py

**Purpose:** Inspect evaluation artifacts and emit quality flags based on chirality, entailment, and citation validation thresholds.

#### Data Structures

```python
@dataclass
class AntagonistConfig:
    input_path: Path
    output_path: Path
    chirality_threshold: float = 0.55
    high_chirality_threshold: float = 0.65
    entailment_threshold: float = 0.5
    evidence_overlap_threshold: float = 0.2
```

#### Classes

```python
class AntagonistRunner:
    def __init__(self, config: AntagonistConfig)
    def run(self) -> Dict[str, Any]
```

#### Output Format

Each flag contains:
```python
{
    "claim_id": str,
    "severity": "LOW" | "MEDIUM" | "HIGH",
    "timestamp": str,
    "issues": [
        {
            "issue_type": str,  # CITATION_INVALID, POLARITY_CONTRADICTION, POLARITY_CONFLICT, WEAK_ENTAILMENT
            "details": dict,
        }
    ],
    "metrics": {
        "chirality_score": float,
        "fisher_rao_distance": float,
        "evidence_overlap": float,
        "polarity_conflict": bool,
        "entailment_score": float,
        "citation_valid": bool,
        "beta1": int,
    }
}
```

#### Summary Output

```python
{
    "input": str,
    "output": str,
    "total_records": int,
    "flagged_records": int,
    "flag_rate": float,
    "severity_breakdown": {"LOW": int, "MEDIUM": int, "HIGH": int},
    "issue_breakdown": {"CITATION_INVALID": int, ...},
    "flag_telemetry": [...]
}
```

#### External Dependencies

- `json` - JSON I/O
- `pathlib` - File paths
- `datetime` - Timestamps

#### Key Logic

- **Severity Escalation:**
  - CITATION_INVALID -> HIGH
  - POLARITY_CONFLICT -> HIGH
  - High chirality + evidence overlap -> HIGH
  - POLARITY_CONTRADICTION -> MEDIUM
  - WEAK_ENTAILMENT -> MEDIUM

---

### semantic_validation.py

**Purpose:** 4-stage semantic validation pipeline for CNS claim extraction, replacing exact-match evaluation.

#### Data Structures

```python
@dataclass
class ValidationResult:
    # Stage 1: Citation Accuracy
    citation_valid: bool
    cited_ids: Set[str]
    missing_ids: Set[str]

    # Stage 2: Entailment
    entailment_score: float
    entailment_pass: bool  # >= 0.75 threshold

    # Stage 3: Semantic Similarity
    semantic_similarity: float
    similarity_pass: bool  # >= 0.7 threshold

    # Stage 4: Paraphrase Tolerance
    paraphrase_accepted: bool

    # Overall
    overall_pass: bool

    # Schema compliance
    schema_valid: bool
    schema_errors: List[str]
```

#### Classes

```python
class SemanticValidator:
    ENTAILMENT_THRESHOLD = 0.75
    SIMILARITY_THRESHOLD = 0.7

    def __init__(self, device: Optional[str] = None)
    def validate_claim(
        self,
        generated_claim: str,
        gold_claim: str,
        generated_full_output: str,
        evidence_corpus: Dict[str, Dict],
        gold_evidence_ids: Set[str],
    ) -> ValidationResult
```

#### Public Functions

```python
def validate_batch(
    validator: SemanticValidator,
    predictions: List[Dict],
    gold_data: List[Dict],
    corpus: Dict[str, Dict],
) -> List[ValidationResult]
```

#### External Dependencies

- `torch` - PyTorch for model inference
- `sentence_transformers` - SentenceTransformer, util for similarity
- `transformers` - AutoModelForSequenceClassification, AutoTokenizer

#### Models Used

- **NLI Model:** `cross-encoder/nli-deberta-v3-large` (entailment)
- **Embedding Model:** `all-MiniLM-L6-v2` (semantic similarity)

#### Key Algorithms

- **4-Stage Validation Pipeline:**
  1. Citation Accuracy (hard gate) - Extract and verify document IDs
  2. Entailment Score (DeBERTa-v3-NLI) - Evidence entails claim
  3. Semantic Similarity (sentence-transformers) - Cosine similarity
  4. Paraphrase Tolerance - Accept valid rephrasings

- **Citation Extraction Patterns:**
  - `Document NNNNNN`
  - `doc_id: NNNNNN`
  - `[NNNNN]` (5+ digit IDs)

---

### citation_validation.py

**Purpose:** Citation hallucination detection - verify that cited documents exist in the source corpus.

#### Data Structures

```python
@dataclass
class CitationValidationResult:
    is_valid: bool
    cited_docs: Set[str]
    valid_docs: Set[str]
    invalid_docs: Set[str]
    hallucination_count: int
```

#### Public Functions

```python
def extract_document_ids(text: str) -> Set[str]
def validate_citations(prompt: str, completion: str) -> CitationValidationResult
def compute_citation_penalty(result: CitationValidationResult, penalty_weight: float = 2.0) -> float
def batch_validate_citations(prompts: list[str], completions: list[str]) -> list[CitationValidationResult]
def citation_validation_stats(results: list[CitationValidationResult]) -> dict[str, float]
```

#### Statistics Output

```python
{
    "valid_rate": float,
    "mean_hallucinations": float,
    "total_hallucinations": int,
    "total_samples": int,
}
```

#### External Dependencies

- `re` - Regular expressions

#### Citation Patterns Detected

- `Document 12345:`
- `(Document 12345)`
- `CLAIM[c1] (Document 12345)`

---

### evaluation.py

**Purpose:** Evaluation harness for SciFact-style claim extraction using 4-stage semantic validation.

#### Classes

```python
class Evaluator:
    def __init__(self, config: EvaluationConfig, state: Optional["PipelineState"] = None)
    def run(self) -> Dict[str, float]
```

#### Public Functions

```python
def evaluate_semantic_match(predicted: List[str], gold_sentences: List[str]) -> float
def build_evaluation_series(samples: List[Dict[str, Any]], moving_avg_window: int = 5) -> Dict[str, Any]
```

#### Metrics Computed

**4-Stage Semantic Validation Metrics:**
- `schema_compliance_rate` (target: >= 95%)
- `citation_accuracy_rate` (hard gate)
- `mean_entailment_score` (threshold: >= 0.75)
- `entailment_pass_rate`
- `mean_semantic_similarity` (threshold: >= 0.70)
- `semantic_similarity_rate` (target: >= 60%)
- `paraphrase_acceptance_rate`
- `overall_pass_rate`

**Topology/Chirality Metrics:**
- `mean_beta1`
- `beta1_nonzero_rate`
- `mean_chirality_score`
- `mean_fisher_rao_distance`
- `std_entailment_score`
- `std_semantic_similarity`

**Legacy Metrics (for comparison):**
- `c1_exact_match_rate_LEGACY`
- `evidence_exact_match_avg_LEGACY`

#### Series Output (for dashboards)

```python
{
    "indices": List[int],
    "timestamps": List[str],
    "cumulative_rates": {
        "schema_valid": List[float],
        "citation_valid": List[float],
        ...
    },
    "value_series": {...},
    "moving_averages": {...},
}
```

#### External Dependencies

- `json` - JSON I/O
- `statistics` - pstdev
- `transformers` - AutoModelForCausalLM, AutoTokenizer
- `peft` - PeftModel
- `torch` - PyTorch
- Optional: `tinker` SDK for Tinker backend

#### Supported Backends

- `hf_peft` - HuggingFace + PEFT/LoRA
- `tinker` - Tinker API

---

### training.py

**Purpose:** Training backends for Thinker including HuggingFace PEFT and Tinker.

#### Data Structures

```python
@dataclass
class TrainingReport:
    checkpoint_dir: Path
    metrics: Dict[str, Any]
    backend: str
```

#### Classes

```python
class TrainingBackend:  # Protocol
    def train(self) -> TrainingReport

class LocalPEFTTrainer(TrainingBackend):
    def __init__(self, config_path: Path)
    def train(self) -> TrainingReport

class TinkerScriptTrainer(TrainingBackend):
    def __init__(self, config: LocalTrainingConfig)
    def train(self) -> TrainingReport

class CitationAwareDataCollator:
    def __init__(self, tokenizer, config: Dict[str, Any], mlm: bool = False)
    def __call__(self, features) -> dict

class CitationAwareTrainer:
    def __init__(self, *args, **kwargs)
    def compute_loss(self, model, inputs, return_outputs=False)
    def train(self, *args, **kwargs)
    def save_model(self, *args, **kwargs)
```

#### Public Functions

```python
def create_training_backend(config: LocalTrainingConfig) -> TrainingBackend
```

#### External Dependencies

- `transformers` - AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
- `peft` - LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftModel
- `datasets` - load_dataset
- `torch` - PyTorch
- `yaml` - YAML config loading
- `subprocess` - For Tinker script execution

#### Key Features

- **LoRA Configuration:**
  - r, lora_alpha, target_modules, lora_dropout, bias

- **4-bit Quantization:**
  - BitsAndBytes NF4 quantization
  - Double quantization

- **Citation-Aware Training:**
  - Validates citations during training
  - Adds penalty to loss for hallucinated citations
  - Configurable via `validate_citations_during_training` and `citation_validity_weight`

---

### data.py

**Purpose:** Data bootstrap utilities for downloading and converting SciFact and FEVER datasets.

#### Constants

```python
REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "cns-support-models" / "scripts"
FEVER_RAW_DIR = REPO_ROOT / "cns-support-models" / "data" / "raw" / "fever"
SCIFACT_PROCESSED_DATASET = REPO_ROOT / "cns-support-models" / "data" / "processed" / "scifact_claim_extractor.jsonl"
SCIFACT_CLEAN_DATASET = REPO_ROOT / "cns-support-models" / "data" / "processed" / "scifact_claim_extractor_clean.jsonl"
```

#### Public Functions

```python
def run(cmd: list[str], *, cwd: Path | None = None) -> None
def run_make_scifact() -> None
def run_data_setup(
    dataset: str,
    claims_path: Path,
    corpus_path: Path,
    output_path: Path,
    clean_output: Path | None,
    filter_invalid: bool,
    fever_claims: Path,
    fever_wiki_dir: Path,
    fever_output: Path,
    fever_include_nei: bool,
    *,
    skip_validation: bool,
    validation_mode: str,
    similarity_threshold: float,
) -> dict
def run_fever_download() -> None
```

#### External Dependencies

- `subprocess` - Running shell commands
- `shutil` - File copying
- `pathlib` - Path handling

#### Supported Datasets

- **SciFact:** Scientific fact verification
- **FEVER:** Fact Extraction and VERification

---

### cli.py

**Purpose:** Command-line interface for Thinker pipeline orchestration.

#### Commands

- `info` - Show pipeline and environment details
- `manifest` - Show latest Tinker adapter manifest
- `validate` - Run pytest + dataset validation
- `train` - Run training after validation
- `eval` - Evaluate a checkpoint
- `run` - Validate, train, and evaluate
- `antagonist` - Run Antagonist heuristics on evaluation artifacts
- `data setup` - Download/prepare datasets (SciFact, FEVER)

#### Public Functions

```python
def build_parser() -> argparse.ArgumentParser
def main(argv: Optional[list[str]] = None) -> int
```

#### External Dependencies

- `argparse` - Argument parsing
- `json` - JSON handling
- `importlib.metadata` - Package version detection
- `platform` - System information

---

### config.py

**Purpose:** Configuration models for the Thinker pipeline.

#### Data Structures

```python
@dataclass(frozen=True)
class SchemaField:
    name: str
    type: str = "string"  # "string", "array", "object"
    required: bool = True
    allow_empty: bool = False

    def validate(self, payload: Dict[str, Any]) -> tuple[bool, str | None]

@dataclass(frozen=True)
class TestSuiteConfig:
    path: Path = Path("tests")
    markers: Optional[str] = None
    args: List[str] = field(default_factory=list)
    enabled: bool = True

@dataclass(frozen=True)
class DatasetValidationConfig:
    path: Path
    schema: List[SchemaField]
    max_examples: Optional[int] = None
    enabled: bool = True
    evidence_mode: str = "schema"  # "schema", "exact", "embedding"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    similarity_threshold: float = 0.75
    claims_path: Optional[Path] = None
    corpus_path: Optional[Path] = None
    relation_field: Optional[str] = None
    require_relations: bool = False

@dataclass(frozen=True)
class LocalTrainingConfig:
    config_path: Path
    backend: str = "hf_peft"  # "hf_peft", "tinker"
    enabled: bool = True
    tinker_config_path: Optional[Path] = None
    tinker_script: Optional[Path] = None
    log_dir: Optional[Path] = None

@dataclass(frozen=True)
class EvaluationConfig:
    claims_file: Path
    corpus_file: Path
    backend: str = "hf_peft"
    base_model: Optional[str] = None
    checkpoint_dir: Optional[Path] = None
    max_samples: int = 50
    output_path: Path = Path("eval_results.jsonl")
    enabled: bool = True
    tinker_manifest_path: Optional[Path] = None
    tinker_adapter_name: Optional[str] = None
    tinker_adapter_path: Optional[str] = None
    tinker_max_tokens: int = 256
    tinker_temperature: float = 0.0

@dataclass(frozen=True)
class PipelineConfig:
    project_root: Path
    tests: Optional[TestSuiteConfig] = None
    data_validation: Optional[DatasetValidationConfig] = None
    training: Optional[LocalTrainingConfig] = None
    evaluation: Optional[EvaluationConfig] = None
```

#### Public Functions

```python
def load_pipeline_config(path: Path) -> PipelineConfig
```

#### External Dependencies

- `yaml` - YAML configuration loading

---

### pipeline.py

**Purpose:** High-level orchestration for Thinker workflows.

#### Data Structures

```python
@dataclass
class PipelineState:
    validation_ran: bool = False
    training_completed: bool = False
    tinker_adapter_name: Optional[str] = None
    tinker_adapter_path: Optional[str] = None
    tinker_adapter_manifest: Optional[Path] = None
    tinker_base_model: Optional[str] = None
```

#### Classes

```python
class ThinkerPipeline:
    def __init__(self, config: PipelineConfig)
    def validate(self) -> Dict[str, Any]
    def train(self, backend: Optional[str] = None, skip_validation: bool = False) -> TrainingReport | None
    def evaluate(self, skip_validation: bool = False) -> dict
    def run(self, backend: Optional[str] = None) -> dict
```

#### External Dependencies

- Internal: `config`, `evaluation`, `training`, `validation`

---

### claim_schema.py

**Purpose:** Minimal helpers for parsing CLAIM[...] and RELATION formatted completions.

#### Constants

```python
CLAIM_LINE_RE = re.compile(r"^CLAIM\[(?P<id>[^\]]+)\]\s*(?:\(Document\s+\d+\))?\s*:\s*(?P<body>.*)$", re.IGNORECASE)
RELATION_LINE_RE = re.compile(
    r"^RELATION\s*[:\-]?\s*(?P<src>\S+)\s+(?P<label>supports|refutes|contrasts)\s+(?P<dst>\S+)",
    re.IGNORECASE,
)
```

#### Data Structures

```python
@dataclass
class Claim:
    identifier: str
    text: str
```

#### Public Functions

```python
def parse_claim_lines(lines: Iterable[str]) -> Dict[str, Claim]
def parse_relation_line(line: str) -> Tuple[str, str, str] | None
```

#### External Dependencies

- `re` - Regular expressions

#### Supported Formats

- `CLAIM[c1]: The hypothesis text`
- `CLAIM[c2] (Document 12345): Supporting claim`
- `RELATION: c2 supports c1`
- `RELATION: c3 refutes c1`
- `RELATION: c4 contrasts c2`

---

### validation.py

**Purpose:** Test suite runner and dataset validator utilities.

#### Data Structures

```python
@dataclass
class DatasetValidationResult:
    path: Path
    total_examples: int
    errors: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors
```

#### Classes

```python
class TestSuiteRunner:
    def __init__(self, config: TestSuiteConfig)
    def run(self) -> None

class DatasetValidator:
    def __init__(self, config: DatasetValidationConfig)
    def validate(self) -> DatasetValidationResult
```

#### External Dependencies

- `json` - JSON parsing
- `subprocess` - Running external validators
- `pytest` - Test execution

---

## External Dependencies Summary

### Core Dependencies

| Package | Version | Usage |
|---------|---------|-------|
| `networkx` | - | Graph construction, cycle detection (betti.py) |
| `numpy` | - | Numerical computations (chirality.py) |
| `torch` | - | Deep learning inference (semantic_validation.py, evaluation.py, training.py) |
| `transformers` | - | NLI models, tokenizers, training (semantic_validation.py, evaluation.py, training.py) |
| `sentence-transformers` | - | Embedding models (semantic_validation.py, chirality.py) |
| `peft` | - | LoRA/parameter-efficient fine-tuning (evaluation.py, training.py) |
| `datasets` | - | Dataset loading (training.py) |
| `pyyaml` | - | Configuration loading (config.py, training.py) |
| `pytest` | - | Test execution (validation.py) |

### Optional Dependencies

| Package | Usage |
|---------|-------|
| `tinker` | Tinker training/inference SDK (evaluation.py, training.py) |
| `bitsandbytes` | 4-bit quantization (training.py) |

### Pre-trained Models

| Model | Source | Usage |
|-------|--------|-------|
| `cross-encoder/nli-deberta-v3-large` | HuggingFace | NLI entailment scoring |
| `all-MiniLM-L6-v2` | sentence-transformers | Semantic similarity, chirality |

---

## Data Flow Architecture

```
                    ┌─────────────┐
                    │   CLI       │
                    │  (cli.py)   │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Pipeline   │
                    │(pipeline.py)│
                    └──────┬──────┘
                           │
         ┌─────────┬───────┼───────┬─────────┐
         │         │       │       │         │
    ┌────▼────┐ ┌──▼──┐ ┌──▼───┐ ┌─▼──┐ ┌───▼────┐
    │Validate │ │Train│ │ Eval │ │Data│ │Antagon.│
    │         │ │     │ │      │ │    │ │        │
    └────┬────┘ └──┬──┘ └──┬───┘ └─┬──┘ └───┬────┘
         │         │       │       │        │
         │    ┌────▼────┐  │       │        │
         │    │Training │  │       │        │
         │    │Backends │  │       │        │
         │    └─────────┘  │       │        │
         │                 │       │        │
    ┌────▼─────────────────▼───────▼────────▼────┐
    │              Semantic Validation            │
    │           (semantic_validation.py)          │
    │  ┌─────────┐  ┌─────────┐  ┌──────────┐    │
    │  │Citation │  │Entail.  │  │Similarity│    │
    │  │Check    │  │(DeBERTa)│  │(MiniLM)  │    │
    │  └─────────┘  └─────────┘  └──────────┘    │
    └─────────────────┬───────────────────────────┘
                      │
    ┌─────────────────▼───────────────────┐
    │         Graph Topology              │
    │  ┌─────────┐  ┌──────────────┐     │
    │  │Betti    │  │ Chirality    │     │
    │  │Numbers  │  │ (Fisher-Rao) │     │
    │  └─────────┘  └──────────────┘     │
    └─────────────────┬───────────────────┘
                      │
    ┌─────────────────▼───────────────────┐
    │         Metrics Emission            │
    │          (emitter.py)               │
    └─────────────────────────────────────┘
```

---

## Porting Priorities

### High Priority (Core CNS Logic)

1. **logic/betti.py** - Graph topology, Betti numbers
2. **metrics/chirality.py** - Fisher-Rao distance, chirality scoring
3. **claim_schema.py** - CLAIM/RELATION parsing
4. **citation_validation.py** - Hallucination detection

### Medium Priority (Validation Pipeline)

5. **semantic_validation.py** - 4-stage validation (requires ML model integration)
6. **antagonist.py** - Quality flagging
7. **validation.py** - Dataset validation

### Lower Priority (Orchestration)

8. **evaluation.py** - Evaluation harness
9. **config.py** - Configuration models
10. **pipeline.py** - Pipeline orchestration
11. **training.py** - Training backends
12. **data.py** - Data utilities
13. **cli.py** - CLI interface
14. **metrics/emitter.py** - Metrics persistence

---

## Notes for Elixir Port

1. **Pure Functions First:** `betti.py`, `chirality.py`, `claim_schema.py`, `citation_validation.py` are largely pure and can be ported directly.

2. **GenServer Patterns:** `AntagonistRunner`, `Evaluator`, `ThinkerPipeline` map well to OTP GenServers.

3. **ML Model Integration:** The semantic validation components require either:
   - Nx/Axon for native Elixir ML
   - Python interop via Ports/NIFs
   - External API calls

4. **Configuration:** The `@dataclass(frozen=True)` pattern maps directly to Elixir structs with `@enforce_keys`.

5. **Graph Operations:** NetworkX operations can be replaced with `:libgraph` or custom implementations.

# CNS Master Porting Plan: Python Thinker to Elixir CNS

**Version:** 1.0.0
**Date:** 2025-11-21
**Status:** Comprehensive Planning Document

---

## 1. Executive Summary

### What Exists Today

| Component | Elixir CNS | Python Thinker |
|-----------|------------|----------------|
| **Core Modules** | 27 modules | 14 modules |
| **Status** | Fully structured, heuristic-based | Production ML pipeline |
| **LLM Integration** | None (templated) | Full (transformers, PEFT) |
| **ML Models** | None | DeBERTa-NLI, MiniLM embeddings |
| **Graph Analysis** | libgraph-based | NetworkX-based |
| **Training** | Stubbed (Tinkex) | Complete (PEFT, Tinker) |

**Elixir CNS** provides a well-architected OTP scaffold with:
- Complete three-agent system (Proposer, Antagonist, Synthesizer)
- Five critics with weighted scoring
- Comprehensive graph modules
- Proper data structures (SNO, Evidence, Challenge, Provenance)

**Python Thinker** provides production-ready:
- 4-stage semantic validation pipeline
- Fisher-Rao chirality computation
- Betti number cycle detection
- LoRA training with citation-aware loss
- Complete evaluation harness

### Total Gap Scope

**Critical Gaps:** 9 features requiring immediate attention
**High Priority Gaps:** 14 features for core functionality
**Medium Priority Gaps:** 11 features for completeness
**Total Gaps Identified:** 34 features across all categories

### Overall Effort Estimate

| Phase | Duration | Effort (Hours) |
|-------|----------|----------------|
| Foundation | Weeks 1-2 | 60 |
| Core Algorithms | Weeks 3-4 | 88 |
| Validation Pipeline | Weeks 5-6 | 96 |
| Data Pipeline | Weeks 7-8 | 80 |
| Training Integration | Weeks 9-10 | 64 |
| **Total** | **10 weeks** | **388 hours** |

---

## 2. Current State Assessment

### 2.1 Elixir CNS: 27 Modules

**Fully Implemented (23 modules):**

| Category | Modules | Key Features |
|----------|---------|--------------|
| Core | CNS, Pipeline, Config, Metrics, Topology, Training | Entry points, orchestration |
| Agents | Proposer, Antagonist, Synthesizer | Three-agent dialectic |
| Data Structures | SNO, Evidence, Provenance, Challenge | Core data types |
| Critics | Grounding, Logic, Causal, Bias, Novelty | Multi-critic evaluation |
| Graph | Builder, Topology, Traversal, Visualization | Graph operations |

**Behaviour Only (4 modules):**
- CrucibleFramework.Datasets
- CrucibleFramework.Sampling
- CrucibleFramework.Ensemble.ML
- CrucibleFramework.Lora

**Stub (1 module):**
- CNS.Application (empty supervisor)

**Working Today:**
- Heuristic-based claim extraction via regex
- Template-based synthesis
- Graph cycle detection
- Basic quality scoring
- JSON serialization

### 2.2 Python Thinker: 14 Modules

| Module | Purpose | Port Priority |
|--------|---------|---------------|
| `logic/betti.py` | Graph topology, Betti numbers | **Critical** |
| `metrics/chirality.py` | Fisher-Rao chirality | **Critical** |
| `claim_schema.py` | CLAIM/RELATION parsing | **Critical** |
| `semantic_validation.py` | 4-stage validation | **Critical** |
| `citation_validation.py` | Hallucination detection | High |
| `antagonist.py` | Quality flagging | High |
| `evaluation.py` | Evaluation harness | High |
| `training.py` | PEFT/LoRA training | High |
| `config.py` | Configuration | Medium |
| `pipeline.py` | Orchestration | Medium |
| `validation.py` | Dataset validation | Medium |
| `data.py` | Data utilities | Medium |
| `cli.py` | CLI interface | Low |
| `metrics/emitter.py` | Metrics persistence | Low |

### 2.3 Support Scripts Requiring Elixir Equivalents

| Script | Purpose | Priority |
|--------|---------|----------|
| `convert_scifact.py` | SciFact → training format | **High** |
| `validate_dataset.py` | JSONL validation | **High** |
| `convert_fever.py` | FEVER → training format | Medium |
| `csv_to_claim_jsonl.py` | CSV → JSONL | Medium |
| `record_lineage.py` | SHA-256 lineage | Medium |
| `train_claim_extractor.py` | LoRA training | Low |
| `eval_scifact_dev.py` | Evaluation | Low |

---

## 3. Gap Prioritization Matrix

### Critical Priority (Must Have)

| Gap | Effort | Dependencies | Design Doc |
|-----|--------|--------------|------------|
| **Claim Schema Parsing** | 14h | None | 07_PORTING_DESIGN_CORE_ALGORITHMS.md §5 |
| **Betti Number Computation** | 14h | libgraph | 07_PORTING_DESIGN_CORE_ALGORITHMS.md §2 |
| **Fisher-Rao Chirality** | 24h | Nx, Bumblebee | 07_PORTING_DESIGN_CORE_ALGORITHMS.md §3 |
| **4-Stage Semantic Validation** | 36h | Bumblebee (NLI, embeddings) | 07_PORTING_DESIGN_CORE_ALGORITHMS.md §4 |
| **SNO Topological/Geometric Fields** | 8h | Betti, Chirality | 06_GAP_ANALYSIS_VS_SPEC.md §2.1 |
| **NLI-based Grounding Critic** | 16h | Bumblebee | 06_GAP_ANALYSIS_VS_SPEC.md §3.4 |
| **LLM Integration for Agents** | 24h | LLM API | 06_GAP_ANALYSIS_VS_SPEC.md §3 |
| **Polarity Conflict Detection** | 4h | Graph module | 05_GAP_ANALYSIS_ELIXIR_VS_PYTHON.md |
| **Citation Validation Enhancement** | 8h | None | 05_GAP_ANALYSIS_ELIXIR_VS_PYTHON.md §2.3 |

### High Priority

| Gap | Effort | Dependencies | Design Doc |
|-----|--------|--------------|------------|
| SciFact Converter | 24h | claim_schema | 08_PORTING_DESIGN_DATA_PIPELINE.md §1 |
| Dataset Validation | 40h | Bumblebee | 08_PORTING_DESIGN_DATA_PIPELINE.md §2 |
| Evaluation Harness | 32h | semantic_validation | 05_GAP_ANALYSIS_ELIXIR_VS_PYTHON.md §3.2 |
| Antagonist Enhancement | 16h | chirality | 05_GAP_ANALYSIS_ELIXIR_VS_PYTHON.md §2.2 |
| Trust Score Computation | 8h | Critics | 06_GAP_ANALYSIS_VS_SPEC.md §4.3 |
| Composite Loss Function | 12h | Training | 06_GAP_ANALYSIS_VS_SPEC.md §4.5 |
| Constrained Decoding (KCTS) | 16h | LLM | 06_GAP_ANALYSIS_VS_SPEC.md §1.6 |
| GAT Logic Critic | 24h | Axon | 06_GAP_ANALYSIS_VS_SPEC.md §1.4 |
| Evidence Retrieval System | 16h | Embeddings | 06_GAP_ANALYSIS_VS_SPEC.md §3.2 |

### Medium Priority

| Gap | Effort | Dependencies | Design Doc |
|-----|--------|--------------|------------|
| FEVER Converter | 32h | claim_schema | 08_PORTING_DESIGN_DATA_PIPELINE.md §1.2.2 |
| Lineage Tracking | 16h | None | 08_PORTING_DESIGN_DATA_PIPELINE.md §3 |
| Configuration Extension | 12h | None | 05_GAP_ANALYSIS_ELIXIR_VS_PYTHON.md §2.5 |
| Metrics Emitter | 8h | None | 05_GAP_ANALYSIS_ELIXIR_VS_PYTHON.md §3.4 |
| Sentence-BERT Novelty Critic | 12h | Bumblebee | 06_GAP_ANALYSIS_VS_SPEC.md §3.4 |
| Source Reliability Network | 16h | Graph | 06_GAP_ANALYSIS_VS_SPEC.md §2.3 |
| Simplicial Complex Representation | 12h | None | 06_GAP_ANALYSIS_VS_SPEC.md §2.4 |
| DPO Training Support | 16h | Training | 06_GAP_ANALYSIS_VS_SPEC.md §5.1 |
| Tinkex LoRA Integration | 16h | Tinkex | 05_GAP_ANALYSIS_ELIXIR_VS_PYTHON.md §3.3 |

### Low Priority

| Gap | Effort | Dependencies | Design Doc |
|-----|--------|--------------|------------|
| CLI Interface | 12h | All modules | 05_GAP_ANALYSIS_ELIXIR_VS_PYTHON.md §3.8 |
| CSV Converter | 8h | claim_schema | 08_PORTING_DESIGN_DATA_PIPELINE.md §1.2.3 |
| Human Evaluation Infrastructure | 24h | All | 06_GAP_ANALYSIS_VS_SPEC.md §5.4 |
| Bregman Manifold Extension | 16h | FIM | 06_GAP_ANALYSIS_VS_SPEC.md §8 |
| Causal Critic (FCI/PC) | 24h | Graph | 06_GAP_ANALYSIS_VS_SPEC.md §3.4 |

---

## 4. Phased Implementation Plan

### Phase 1: Foundation (Weeks 1-2)

**Goal:** Establish core parsing and graph analysis

**Modules/Functions:**
- `CNS.Schema.Parser` - CLAIM/RELATION regex parsing
- `CNS.Logic.Betti` - Graph stats, beta1, polarity conflict
- `CNS.Validation.Citation` - Document ID extraction, hallucination detection
- SNO struct extensions (τ, γ, χ fields)

**Deliverables:**
- [ ] `CNS.Schema.Parser.parse/1` with full test coverage
- [ ] `CNS.Logic.Betti.compute_graph_stats/2` matching Python behavior
- [ ] `CNS.Logic.Betti.polarity_conflict?/2` implemented
- [ ] `CNS.Validation.Citation.extract_document_ids/1`
- [ ] Extended SNO struct with topological/geometric fields
- [ ] Unit tests achieving >90% coverage

**Success Criteria:**
- All regex patterns match Python equivalents
- Betti numbers match for identical graphs
- Polarity conflict detection 100% accurate
- SNO serialization backwards compatible

**Estimated Effort:** 60 hours

### Phase 2: Core Algorithms (Weeks 3-4)

**Goal:** Implement Fisher-Rao chirality and embedding support

**Modules/Functions:**
- `CNS.Metrics.Chirality` - FisherRaoStats, ChiralityResult
- `CNS.Metrics.Chirality.build_fisher_rao_stats/2`
- `CNS.Metrics.Chirality.fisher_rao_distance/3`
- `CNS.Metrics.Chirality.compare/6`
- `CNS.Embedders.Bumblebee` adapter

**Dependencies to Install:**
```elixir
# mix.exs
{:nx, "~> 0.7"},
{:exla, "~> 0.7"},
{:bumblebee, "~> 0.5"},
```

**Models to Download:**
- `sentence-transformers/all-MiniLM-L6-v2` (384-dim embeddings)

**Deliverables:**
- [ ] `CNS.Metrics.Chirality` module with Nx.Defn functions
- [ ] Embedder behaviour and Bumblebee adapter
- [ ] GPU-accelerated distance computation
- [ ] Chirality score formula matching Python: `norm_distance * 0.6 + overlap_factor * 0.2 + conflict_penalty`
- [ ] Integration tests with real embeddings

**Success Criteria:**
- Fisher-Rao distance within 1e-6 of Python for same inputs
- Chirality scores match Python implementation
- GPU inference <10ms per embedding

**Estimated Effort:** 88 hours

### Phase 3: Validation Pipeline (Weeks 5-6)

**Goal:** Complete 4-stage semantic validation with neural models

**Modules/Functions:**
- `CNS.Validation.Semantic` - GenServer with model state
- `CNS.Validation.Semantic.validate_claim/5`
- `CNS.Validation.Semantic.validate_batch/3`
- Integration with `CNS.Critics.Grounding`

**Models to Download:**
- `cross-encoder/nli-deberta-v3-large` (NLI entailment)

**Bumblebee/Nx Setup:**
```elixir
# Initialize NLI model
{:ok, model_info} = Bumblebee.load_model({:hf, "cross-encoder/nli-deberta-v3-large"})
{:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "microsoft/deberta-v3-large"})

serving = Bumblebee.Text.text_classification(model_info, tokenizer,
  compile: [batch_size: 1, sequence_length: 512],
  defn_options: [compiler: EXLA]
)
```

**Deliverables:**
- [ ] 4-stage pipeline: Citation → Entailment → Similarity → Paraphrase
- [ ] `ValidationResult` struct with all stage outputs
- [ ] Thresholds: entailment ≥ 0.75, similarity ≥ 0.7
- [ ] Batch validation with concurrent processing
- [ ] Antagonist integration with severity escalation

**Success Criteria:**
- Validation results match Python for same inputs
- Entailment scores within 0.01 of Python
- Batch processing >100 examples/minute
- Memory stable under load

**Estimated Effort:** 96 hours

### Phase 4: Data Pipeline (Weeks 7-8)

**Goal:** Complete dataset converters and validation infrastructure

**Modules/Functions:**
- `CNS.Pipeline.Converters.SciFact`
- `CNS.Pipeline.Converters.FEVER`
- `CNS.Pipeline.Validation.Validator`
- `CNS.Pipeline.Lineage.Tracker`

**Dependencies:**
```elixir
{:nimble_csv, "~> 1.2"},
{:yaml_elixir, "~> 2.9"},
```

**Deliverables:**
- [ ] SciFact converter producing identical JSONL to Python
- [ ] FEVER converter with streaming wiki loading
- [ ] Dataset validator with schema, claim, relation checks
- [ ] Embedding-based semantic matching
- [ ] Lineage tracker with SHA-256, git commit
- [ ] crucible_datasets integration

**Success Criteria:**
- Converters produce identical output to Python scripts
- Validation catches all errors Python catches
- Lineage records match Python format
- Integration with crucible_datasets complete

**Estimated Effort:** 80 hours

### Phase 5: Training Integration (Weeks 9-10)

**Goal:** Complete Tinkex integration and evaluation harness

**Modules/Functions:**
- Complete `CNS.Training` with actual Tinkex calls
- `CNS.Evaluation.Harness` for benchmark execution
- Evaluation metrics aggregation

**Tinkex Integration:**
```elixir
# Complete training flow
{:ok, session} = Tinkex.LoRA.create_experiment(%{
  base_model: "meta-llama/Llama-2-7b-hf",
  lora_rank: 16,
  lora_alpha: 32,
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
})

for batch <- training_data do
  Tinkex.LoRA.train_step(session, batch)
end
```

**Deliverables:**
- [ ] Working Tinkex LoRA training
- [ ] Citation-aware loss computation
- [ ] Evaluation harness with SciFact/FEVER
- [ ] Metrics: schema_compliance, citation_accuracy, mean_entailment
- [ ] Series output for dashboards

**Success Criteria:**
- Training produces checkpoints
- Evaluation metrics match Python
- Dashboard data exports correctly
- End-to-end pipeline functional

**Estimated Effort:** 64 hours

---

## 5. Dependencies & Setup

### 5.1 Hex Packages Required

```elixir
# mix.exs
defp deps do
  [
    # Existing
    {:jason, "~> 1.4"},
    {:uuid, "~> 1.1"},
    {:libgraph, "~> 0.16"},

    # New - Numerical/ML
    {:nx, "~> 0.7"},
    {:exla, "~> 0.7"},      # CUDA/GPU support
    {:bumblebee, "~> 0.5"},
    {:axon, "~> 0.6"},

    # New - Data Pipeline
    {:nimble_csv, "~> 1.2"},
    {:yaml_elixir, "~> 2.9"},
    {:ecto, "~> 3.11"},

    # New - HTTP
    {:req, "~> 0.4"},

    # Testing
    {:stream_data, "~> 0.6", only: :test}
  ]
end
```

### 5.2 ML Model Downloads

| Model | Size | Purpose | HuggingFace ID |
|-------|------|---------|----------------|
| MiniLM-L6-v2 | 22MB | Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| DeBERTa-v3-NLI | 1.5GB | Entailment | `cross-encoder/nli-deberta-v3-large` |

**Pre-download Command:**
```bash
# Ensure models are cached before runtime
mix run -e 'Bumblebee.load_model({:hf, "sentence-transformers/all-MiniLM-L6-v2"})'
mix run -e 'Bumblebee.load_model({:hf, "cross-encoder/nli-deberta-v3-large"})'
```

### 5.3 Configuration

**config/config.exs:**
```elixir
config :exla, :clients,
  default: [platform: :cuda]  # or :host for CPU

config :cns, :models,
  embedding: "sentence-transformers/all-MiniLM-L6-v2",
  nli: "cross-encoder/nli-deberta-v3-large"

config :cns, :thresholds,
  entailment: 0.75,
  similarity: 0.70,
  chirality: 0.55
```

---

## 6. Risk Assessment

### 6.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Bumblebee model incompatibility | Medium | High | Test early, have Python fallback via Port |
| EXLA GPU setup issues | Medium | Medium | Document CPU-only path, use EXLA.Backend |
| Memory pressure from large models | High | Medium | Model serving with batching, limit concurrency |
| NetworkX → libgraph behavior differences | Low | Medium | Comprehensive test coverage, property tests |
| Tinkex unavailability | Medium | High | Design interface, stub initially |

### 6.2 Schedule Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Bumblebee learning curve | High | Medium | Allocate 1 week buffer for Phase 2 |
| Integration complexity | Medium | High | Continuous integration testing |
| Scope creep | Medium | Medium | Strict prioritization, defer Low priority |

### 6.3 Mitigation Strategies

**For ML Model Risks:**
1. Start with CPU-only path, add GPU later
2. Implement external API fallback for NLI/embeddings
3. Use Nx.Serving for batched inference
4. Cache model outputs aggressively

**For Integration Risks:**
1. Match Python behavior exactly before optimizing
2. Property-based testing for parsers
3. Golden file tests for converters
4. Integration tests per phase

**For Schedule Risks:**
1. Daily progress tracking
2. Weekly stakeholder updates
3. Defer Low priority items
4. Parallel development where possible

---

## 7. Document Index

| # | Document | Description |
|---|----------|-------------|
| 01 | `01_ELIXIR_CNS_INVENTORY.md` | Complete inventory of 27 Elixir modules with APIs, data structures, completeness status |
| 02 | `02_PYTHON_THINKER_INVENTORY.md` | Complete inventory of 14 Python modules with data structures, algorithms, dependencies |
| 03 | `03_SUPPORT_SCRIPTS_INVENTORY.md` | Analysis of 9 data pipeline scripts for dataset conversion and training |
| 04 | `04_CNS3_THEORETICAL_SPEC.md` | CNS 3.0 theoretical specification with algorithms, metrics, training strategies |
| 05 | `05_GAP_ANALYSIS_ELIXIR_VS_PYTHON.md` | Feature-by-feature gap analysis between Elixir CNS and Python Thinker |
| 06 | `06_GAP_ANALYSIS_VS_SPEC.md` | Gap analysis between current implementation and CNS 3.0 theoretical spec |
| 07 | `07_PORTING_DESIGN_CORE_ALGORITHMS.md` | Detailed Elixir designs for Betti, Chirality, Semantic Validation, Claim Parsing |
| 08 | `08_PORTING_DESIGN_DATA_PIPELINE.md` | Detailed Elixir designs for Converters, Validation, Lineage, Schemas |

---

## 8. Next Steps

### Immediate Actions (This Week)

1. **Environment Setup**
   - Add Nx/EXLA/Bumblebee to mix.exs
   - Test GPU/CUDA availability
   - Download and cache ML models

2. **Start Phase 1**
   - Create `CNS.Schema.Parser` module
   - Port regex patterns from claim_schema.py
   - Write comprehensive tests

3. **Architecture Decision**
   - Confirm Bumblebee for NLI (vs external API)
   - Confirm Tinkex availability timeline
   - Set up CI pipeline for new modules

### Success Metrics

| Milestone | Target Date | Verification |
|-----------|-------------|--------------|
| Phase 1 Complete | Week 2 | All unit tests pass, parsers match Python |
| Phase 2 Complete | Week 4 | Chirality scores within 1e-6 of Python |
| Phase 3 Complete | Week 6 | End-to-end validation pipeline functional |
| Phase 4 Complete | Week 8 | All converters produce identical output |
| Phase 5 Complete | Week 10 | Training and evaluation functional |

---

## Appendix A: Quick Reference

### Key Thresholds (CNS 3.0)

- Schema compliance: ≥ 95%
- Citation accuracy: ≥ 95%
- Mean entailment: ≥ 0.50
- Entailment pass: ≥ 0.75
- Semantic similarity: ≥ 0.70
- Chirality trigger: ≥ 0.55

### Key Formulas

**Betti Number:**
```
β₁ = edges - nodes + components
```

**Fisher-Rao Distance:**
```
distance = sqrt(Σ (diff * inv_var * diff))
```

**Chirality Score:**
```
norm_distance = distance / (distance + 1.0)
overlap_factor = 1.0 - clamp(evidence_overlap, 0, 1)
conflict_penalty = 0.25 if polarity_conflict else 0.0
chirality_score = min(1.0, norm_distance * 0.6 + overlap_factor * 0.2 + conflict_penalty)
```

### Critic Weights

| Critic | Weight | Focus |
|--------|--------|-------|
| Grounding | 0.4 | Factual accuracy, evidence |
| Logic | 0.3 | Consistency, circular reasoning |
| Novelty | 0.15 | Originality, information density |
| Causal | 0.1 | Causal validity |
| Bias | 0.05 | Fairness, loaded language |

---

*End of Master Porting Plan*

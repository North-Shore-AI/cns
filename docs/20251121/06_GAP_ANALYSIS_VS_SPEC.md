# CNS Gap Analysis: Implementation vs Theoretical Specification

**Date:** 2025-11-21
**Comparison:** Elixir CNS Implementation vs CNS 3.0 Theoretical Specification

---

## Executive Summary

The current Elixir CNS implementation provides a solid structural foundation but lacks the core mathematical and machine learning components that define CNS 3.0. The implementation has **23 fully functional modules** but is fundamentally **heuristic-based** rather than utilizing the information-geometric and topological methods specified in CNS 3.0.

**Critical Gaps:**
- No Fisher Information Metric (FIM) computation
- No persistent homology / Betti number calculation
- No neural network critics (GAT, DeBERTa)
- No LLM integration for agents
- Heuristic chirality rather than geometric chirality

---

## 1. Core Algorithms

### 1.1 Fisher Information Metric (FIM)

| Aspect | Spec Requirement | Current Implementation | Gap |
|--------|-----------------|------------------------|-----|
| **Definition** | `g_μν(θ) = E[∂log p(x\|θ)/∂θ_μ · ∂log p(x\|θ)/∂θ_ν]` | None | Complete absence |
| **Computation** | Forward-backward pass, gradient construction | None | No neural network integration |
| **Fisher-Rao distance** | `fisher_rao_distance(dist1, dist2)` | **Implemented in CNS.Metrics** | Incomplete - uses simple distribution comparison, not true FIM |
| **Usage** | Topological analysis distance matrix | None | Not used for topology |

**Gap Description:** The spec requires FIM as the canonical Riemannian metric for measuring semantic distinguishability. Current implementation has a `fisher_rao_distance/2` function but it's a simplified version that doesn't compute actual Fisher information from model gradients.

**Priority:** Critical
**Complexity:** High - Requires neural network integration and gradient computation
**Dependencies:** LLM model access, Nx tensors, automatic differentiation

---

### 1.2 Persistent Homology / Betti Numbers

| Aspect | Spec Requirement | Current Implementation | Gap |
|--------|-----------------|------------------------|-----|
| **Vietoris-Rips filtration** | Build from FIM distance matrix | None | Complete absence |
| **Betti number computation** | Track birth/death of topological features | Graph-based Betti in `CNS.Topology` | **Wrong approach** - uses simple graph connectivity |
| **β₀ interpretation** | Argumentative coherence / fragmentation | Component count in graphs | Partially aligned |
| **β₁ interpretation** | Circular reasoning patterns | Cycle detection in graphs | **Simplified** - not persistence-based |
| **β₂ interpretation** | Surface coherence lacking internal support | None | Missing |

**Gap Description:** The spec requires persistent homology computed via filtrations on point clouds derived from SNO embeddings with FIM distances. Current implementation calculates Betti numbers from graph structure only, missing the key insight of tracking topological features across scales.

**Priority:** Critical
**Complexity:** High - Requires TDA library (Ripser, GUDHI) integration
**Dependencies:** FIM computation, embeddings, TDA library bindings

---

### 1.3 Narrative Chirality Computation

| Aspect | Spec Requirement | Current Implementation | Gap |
|--------|-----------------|------------------------|-----|
| **Formula** | `χ(S₁, S₂) = α·(1 - cos_g(H₁, H₂))·(T₁·T₂) + β·GraphConflict(G₁, G₂)` | Challenge severity weighting | **Fundamentally different** |
| **cos_g** | Cosine with Fisher metric | None | Missing |
| **GraphConflict** | Typed contradiction density from simplicial cycles | Challenge count heuristic | Simplified |
| **Embeddings** | On statistical manifold | None | No embeddings |

**Current Implementation (`CNS.Antagonist.score_chirality/1`):**
```elixir
# Simple weighted average based on challenge severity
challenges |> Enum.map(&chirality_score/1) |> avg
```

**Gap Description:** Spec defines chirality as geometric asymmetry between thesis/antithesis embeddings. Current implementation is a heuristic score based on challenge counts and severity levels.

**Priority:** Critical
**Complexity:** High - Requires embeddings, FIM, graph conflict detection
**Dependencies:** Embedding model, FIM, simplicial complex representation

---

### 1.4 Graph Attention Network (GAT) for Logic Critic

| Aspect | Spec Requirement | Current Implementation | Gap |
|--------|-----------------|------------------------|-----|
| **Architecture** | 3 layers, 8 attention heads, 768-dim | None | Complete absence |
| **Edge types** | {entailment, contradiction, support, refutation} | {supports, cites, contradicts, child_of} | Partial overlap |
| **Training** | 50K synthetic + 5K human graphs | None | No training infrastructure |
| **Output** | Coherence score, detected cycles, β₁ | Pattern-based analysis | Heuristic only |

**Gap Description:** Spec requires a trained GAT neural network for logic criticism. Current `CNS.Critics.Logic` uses pattern matching and graph cycle detection heuristics.

**Priority:** High
**Complexity:** High - Requires GNN implementation (Axon), training data
**Dependencies:** Axon library, training pipeline, annotated graph data

---

### 1.5 Synthesis Operator

| Aspect | Spec Requirement | Current Implementation | Gap |
|--------|-----------------|------------------------|-----|
| **Definition** | `S: SNO × SNO × Evidence → SNO` | Template-based text merging | No formal operator |
| **Properties** | Commutativity, Associativity, Identity, Absorption, Monotonicity | None verified | No algebraic properties |
| **Evidence-preserving** | `{e: w(e) > τ_min} ∩ (E_A ∪ E_B) ⊆ E_C` | Simple union of evidence | Weight threshold missing |
| **Graph projection** | Remove contradiction cycles | None | Missing |

**Gap Description:** Spec defines synthesis as a formal algebraic operator with provable properties. Current implementation is template-based text manipulation without mathematical guarantees.

**Priority:** High
**Complexity:** Medium - Requires formalization of current approach
**Dependencies:** None (structural change)

---

### 1.6 Constrained Decoding (KCTS)

| Aspect | Spec Requirement | Current Implementation | Gap |
|--------|-----------------|------------------------|-----|
| **Process** | Generate → Extract claims → Verify → Re-run if contradictions | None | Complete absence |
| **Iterations** | Up to 3 | N/A | N/A |
| **Verification** | Grounding Critic check | None | No LLM generation |

**Gap Description:** Spec requires Knowledge-Constrained Tree Search during synthesis. Current implementation doesn't use LLM generation, so constrained decoding is not applicable.

**Priority:** High
**Complexity:** Medium - After LLM integration
**Dependencies:** LLM integration, Grounding Critic neural model

---

### 1.7 LoRA Fine-Tuning

| Aspect | Spec Requirement | Current Implementation | Gap |
|--------|-----------------|------------------------|-----|
| **Formula** | `h = W₀x + (α/r)B(A(x))` | `CNS.Training` module | **Stubbed** |
| **Configuration** | r=16, α=32, all linear layers | Config structure present | Not functional |
| **Integration** | Tinkex | Returns `:tinkex_not_available` | Placeholder only |

**Gap Description:** Training module exists with correct structure but actual Tinkex integration is stubbed out.

**Priority:** Medium
**Complexity:** Medium - Tinkex is external dependency
**Dependencies:** Tinkex library, training data preparation

---

## 2. Data Structures

### 2.1 Structured Narrative Object (SNO)

| Spec Definition | Current Implementation | Gap |
|----------------|------------------------|-----|
| **5-tuple:** `(E, L, τ, γ, χ)` | **Different structure** | Major structural difference |
| **E:** Events | `evidence: [Evidence.t()]` | Partial match |
| **L:** CausalLinks | `children: [t()]`, provenance parent_ids | Partial - no explicit causal links |
| **τ:** TopologicalSignature {β₀, β₁, β₂} | **Missing** | Critical gap |
| **γ:** GeometricSignature (FIM eigenvalues) | **Missing** | Critical gap |
| **χ:** Narrative Chirality | **Missing** | Critical gap |

**Current SNO Structure:**
```elixir
%CNS.SNO{
  id: String.t(),
  claim: String.t(),
  evidence: [Evidence.t()],
  confidence: float(),
  provenance: Provenance.t() | nil,
  metadata: map(),
  children: [t()],
  synthesis_history: [map()]
}
```

**Missing Fields:**
1. `topological_signature: %{b0: integer(), b1: integer(), b2: integer()}`
2. `geometric_signature: [float()]` (FIM eigenvalues)
3. `chirality: float()`
4. `trust_score: float()` (from critic ensemble)
5. `uncertainty: map()` (calibrated confidence, epistemic intervals)

**Priority:** Critical
**Complexity:** Low - Structural addition
**Dependencies:** Computation of τ, γ, χ values

---

### 2.2 Extended SNO-3 (7-tuple)

| Field | Spec Definition | Current Status |
|-------|----------------|----------------|
| **H** | Hypothesis embedding ∈ ℝ^d | Missing |
| **G** | Reasoning graph (V, E_G, ρ, τ) | Partial - Graph modules exist |
| **E** | Evidence set {(e_i, s(e_i), t_i, q_i)} | Partial - has validity/relevance |
| **T** | Trust score from critic ensemble | Missing |
| **M** | Metadata (provenance, domain, licensing) | Partial - has provenance |
| **U** | Uncertainty (calibrated confidence) | Missing - only has confidence |
| **Θ_t** | Temporal evolution kernel | Missing |

**Priority:** High
**Complexity:** Medium
**Dependencies:** Embedding model, critic ensemble

---

### 2.3 Evidence Structure

| Spec Requirement | Current Implementation | Gap |
|-----------------|------------------------|-----|
| **Fields:** `(e_i, s(e_i), t_i, q_i)` | `(source, content, validity, relevance, timestamp)` | Mostly aligned |
| **Source reliability** | Via Bayesian PageRank network | None | Missing |
| **Weight formula** | `w(e) = q_e · c(s(e)) · e^(-λ(t - t_e))` | `score = (validity + relevance) / 2` | Simplified |

**Current Evidence Structure is adequate** but missing:
- Source centrality scoring
- Temporal decay calculation
- Integration with source reliability network

**Priority:** Medium
**Complexity:** Medium
**Dependencies:** Source reliability network implementation

---

### 2.4 Simplicial Complex Representation

| Spec Requirement | Current Implementation | Gap |
|-----------------|------------------------|-----|
| **0-simplices** | Claims as vertices | Graph vertices | Aligned |
| **1-simplices** | Relations as edges | Graph edges | Aligned |
| **k-simplices** | Higher-order relations | **Missing** | No higher-order support |

**Gap Description:** Current graph representation is limited to edges (1-simplices). Spec requires support for higher-order simplices like 2-simplices for `c_i ∧ c_j ⟹ c_k`.

**Priority:** Medium
**Complexity:** Medium
**Dependencies:** Simplicial complex library or custom implementation

---

### 2.5 Antagonist Output Structure

**Spec Requires:**
```json
{
  "claim_id": "string",
  "issue_type": "POLARITY_CONTRADICTION | CIRCULAR_REASONING | EVIDENCE_INCONSISTENCY",
  "chirality_delta": "float",
  "evidence": ["snippets"],
  "critic_scores": {"entailment": "float", "chirality": "float", "beta1": "int"},
  "severity": "LOW | MED | HIGH"
}
```

**Current Challenge Structure:**
```elixir
%CNS.Challenge{
  id: String.t(),
  target_id: String.t(),
  challenge_type: :contradiction | :evidence_gap | :scope | :logical | :alternative,
  description: String.t(),
  counter_evidence: [Evidence.t()],
  severity: :high | :medium | :low,
  confidence: float(),
  resolution: :accepted | :rejected | :partial | :pending,
  metadata: map()
}
```

**Missing:**
- `chirality_delta: float()`
- `critic_scores: map()` with entailment, chirality, beta1
- Issue type mapping differs (more granular in spec)

**Priority:** Medium
**Complexity:** Low
**Dependencies:** Chirality and critic score computation

---

## 3. Agent Behaviors

### 3.1 Proposer Agent

| Spec Behavior | Current Implementation | Gap |
|--------------|------------------------|-----|
| **Extract claims** | `extract_claims/2` | **Implemented** but heuristic |
| **Generate embeddings** | Per-claim embeddings | Missing |
| **Compute signatures** | Initial β₁ and χ scores | Missing |
| **Output format** | SNO manifests (JSONL) | JSON/map only |

**Gap Description:** Proposer extracts claims via regex patterns rather than LLM-based extraction. No embeddings or topological signatures computed.

**Priority:** High
**Complexity:** Medium
**Dependencies:** LLM integration, embedding model

---

### 3.2 Antagonist Agent

| Spec Behavior | Current Implementation | Gap |
|--------------|------------------------|-----|
| **Objective** | MAXIMIZE β₁ and χ | FIND issues (no optimization) | Different paradigm |
| **Triggers** | χ ≥ 0.55, polarity_conflict, β₁ > 0 | Heuristic patterns | No numeric thresholds |
| **Polarity stress tests** | Regex negation + embedding anti-neighbors | Find contradictions heuristically | Simplified |
| **Evidence consistency** | Entailment score < 0.5 | Pattern matching | No NLI model |
| **Counter-evidence retrieval** | Actual retrieval | None | Missing |

**Current Challenge Types:**
- `:contradiction` - partial match to polarity conflicts
- `:evidence_gap` - partial match to evidence inconsistency
- `:scope` - not in spec
- `:logical` - partial match to circular reasoning
- `:alternative` - not in spec

**Missing Capabilities:**
1. No actual evidence retrieval
2. No embedding-based anti-neighbor finding
3. No critic score integration
4. No β₁ monitoring/reporting

**Priority:** Critical
**Complexity:** High
**Dependencies:** Embedding model, retrieval system, NLI model

---

### 3.3 Synthesizer Agent

| Spec Behavior | Current Implementation | Gap |
|--------------|------------------------|-----|
| **Objective** | MINIMIZE β₁ and χ | Merge text | No optimization |
| **Evidence-preserving** | Reference evidence IDs inline | Include evidence in output | Partial |
| **Address all conflicts** | Required | Template mentions conflicts | Superficial |
| **Training** | Constitutional AI + DPO + KCTS | None | Complete absence |
| **Output targets** | β₁ ≈ 0, χ ≈ 0, 100% citation traceability | None measured | No targets |

**Gap Description:** Synthesizer uses template-based text merging without LLM generation. No actual synthesis training or constrained decoding.

**Priority:** Critical
**Complexity:** High
**Dependencies:** LLM integration, training pipeline, Constitutional AI data

---

### 3.4 Critic Panel

#### Logic Critic

| Spec | Current | Gap |
|------|---------|-----|
| **Architecture** | GAT (3 layers, 8 heads) | Pattern matching + graph cycles | Completely different |
| **Output** | Coherence score, cycles, β₁ | Issues list, heuristic score | Missing β₁ output |

**Priority:** High
**Complexity:** High
**Dependencies:** Axon GNN, training data

#### Grounding Critic

| Spec | Current | Gap |
|------|---------|-----|
| **Architecture** | DeBERTa-v3-large (434M) | Heuristic checks | No neural model |
| **Output** | 3-class probability [Entails, Neutral, Contradicts] | Score based on patterns | No NLI |
| **Training data** | FEVER + MultiFC | None | No training |

**Priority:** Critical
**Complexity:** High
**Dependencies:** DeBERTa model, inference infrastructure

#### Novelty Critic

| Spec | Current | Gap |
|------|---------|-----|
| **Architecture** | Sentence-BERT twin network | Jaccard similarity | No embeddings |
| **Method** | ANN index comparison | Word overlap | Fundamentally different |

**Priority:** Medium
**Complexity:** Medium
**Dependencies:** Sentence-BERT model, ANN index

#### Evidence Verification Critic

| Spec | Current | Gap |
|------|---------|-----|
| **Architecture** | Cross-encoder (MiniLM-L-12) | Part of Grounding Critic | No separate critic |
| **Source reliability** | Bayesian PageRank | None | Missing |

**Priority:** Medium
**Complexity:** Medium
**Dependencies:** Cross-encoder model, source network

#### Missing Critics

- **Causal Critic** (FCI/PC-style) - Current has heuristic causal critic
- **Bias Critic** (Disparity proxy) - Current has pattern-based bias critic
- **Completeness Critic** - Not implemented

**Priority:** Low (initially)
**Complexity:** High
**Dependencies:** Causal discovery library

---

## 4. Metrics and Formulas

### 4.1 Implemented Metrics

| Metric | Spec Formula | Implementation Status |
|--------|-------------|----------------------|
| `quality_score` | Composite | **Implemented** - `CNS.Metrics.quality_score/1` |
| `entailment` | NLI-based | **Implemented** - Heuristic version |
| `citation_accuracy` | Precision/Recall | **Implemented** - `CNS.Metrics.citation_accuracy/1` |
| `pass_rate` | Threshold check | **Implemented** - `CNS.Metrics.pass_rate/2` |
| `chirality` | Geometric | **Implemented** - Simplified version |
| `fisher_rao_distance` | FIM-based | **Implemented** - Simplified distribution comparison |
| `schema_compliance` | Validation | **Implemented** - `CNS.Metrics.schema_compliance/1` |
| `mean_entailment` | NLI average | **Implemented** - `CNS.Metrics.mean_entailment/1` |
| `convergence_delta` | Iteration diff | **Implemented** - `CNS.Metrics.convergence_delta/2` |
| `coherence_score` | Multiple factors | **Implemented** - `CNS.Synthesizer.coherence_score/1` |

### 4.2 Missing Metrics

| Metric | Spec Formula | Priority |
|--------|-------------|----------|
| **Chirality Score (full)** | `α·(1 - cos_g(H₁, H₂))·(T₁·T₂) + β·GraphConflict(G₁, G₂)` | Critical |
| **Evidential Entanglement** | `EScore(S_i, S_j) = Σw(e) / Σw(e)` with source reliability | High |
| **Trust Score** | `T(S) = softmax(w)^T · [critic_scores]` | High |
| **Critic Score** | `1 - entailment_score` | Medium |
| **Composite Loss** | `L_CE + λ₁·L_β₁ + λ₂·L_χ` | High |
| **Bias Amplification Factor** | `α = bias(H_syn) / max(bias(H₁), bias(H₂))` | Medium |
| **Information Preservation** | `I(H_syn) ≥ min(I(H₁), I(H₂)) · (1 - δ)` | Medium |
| **Bregman Divergence** | `F(θ) = A(θ) + λ·β₁(S)` | Low (future) |
| **BERTScore F1** | External metric | High |
| **NovAScore** | Novelty metric | Medium |
| **SummaC Consistency** | Factual consistency | Medium |

### 4.3 Metric Implementation Quality

**Well-Implemented:**
- Basic quality scoring
- Citation accuracy
- Schema compliance

**Needs Upgrade:**
- `fisher_rao_distance` - needs actual FIM
- `chirality` - needs geometric computation
- `entailment` - needs NLI model

**Priority:** High
**Complexity:** Medium-High (depends on underlying models)
**Dependencies:** Neural models for accurate metrics

---

## 5. Training / Validation

### 5.1 Training Capabilities

| Spec Requirement | Current Implementation | Gap |
|-----------------|------------------------|-----|
| **LoRA fine-tuning** | Config + stubs | Tinkex not integrated |
| **Dataset preparation** | `prepare_dataset/2` | **Implemented** |
| **Triplet format** | `triplet_to_example/3` | **Implemented** |
| **Checkpointing** | `save_checkpoint/2`, `load_checkpoint/1` | **Implemented** |
| **Evaluation** | `evaluate/2` | **Implemented** |

**Missing Training Components:**
1. Constitutional AI generation (5-stage bootstrap)
2. DPO preference optimization
3. Curriculum learning
4. Hybrid training loop with topological loss
5. Weak supervision (Snorkel)

### 5.2 Missing Loss Functions

| Loss | Formula | Priority |
|------|---------|----------|
| **Topological Loss** | `L_β₁ = REINFORCE(-β₁)` | Critical |
| **Geometric Loss** | `L_χ` | Critical |
| **DPO Loss** | `L_DPO = -E[log σ(β(...) - β(...))]` | High |
| **Focal Loss** | For hard contradictions | Medium |

### 5.3 Validation Benchmarks

**Spec Datasets:**
- SciFact (1,261/450/300) - Target F1 ≥ 0.75
- FEVER (145K/10K/10K) - Target Accuracy 85-92%
- SYNTH-DIAL - 1,000 triplets
- HIST-SCI - 3-500 debates
- DEBAGREEMENT - 42,894 pairs
- LegalPrecedent - 1,200 cases

**Current Implementation:**
- Crucible dataset contracts defined (behaviour only)
- No actual dataset loading

**Priority:** Medium
**Complexity:** Medium
**Dependencies:** Dataset acquisition, Crucible integration

### 5.4 Human Evaluation Infrastructure

**Missing:**
- Inter-rater reliability (κ > 0.75)
- 1-7 Likert scale collection
- Active learning pipeline
- Expert annotation workflow

**Priority:** Low (after core algorithms)
**Complexity:** High (process, not code)
**Dependencies:** Human annotators, annotation UI

---

## 6. Priority Summary

### Critical (Must Have for CNS 3.0)

1. **Fisher Information Metric** - Foundation for all geometric/topological analysis
2. **Persistent Homology** - Core algorithmic differentiator
3. **Full Chirality Computation** - Key metric for dialectical synthesis
4. **LLM Integration** - Agents need actual generation
5. **SNO Topological/Geometric Fields** - Data structure completion
6. **NLI-based Grounding Critic** - Foundational critic

### High Priority

7. GAT Logic Critic
8. Synthesis Operator formalization
9. Constrained Decoding (KCTS)
10. Trust Score computation
11. Embedding model integration
12. Evidence retrieval system
13. Composite loss function
14. DPO training support

### Medium Priority

15. LoRA Tinkex integration
16. Extended Evidence structure
17. Simplicial complex representation
18. Sentence-BERT Novelty Critic
19. Source reliability network
20. Benchmark dataset loading
21. Curriculum learning

### Low Priority (Future/Advanced)

22. Bregman manifold extension
23. Causal critic (FCI/PC)
24. Human evaluation infrastructure
25. Completeness critic

---

## 7. Implementation Roadmap

### Phase 1: Core Infrastructure (Weeks 1-4)

**Goal:** Enable geometric/topological analysis

1. Integrate Nx/Axon for tensor operations
2. Implement actual FIM computation
3. Integrate TDA library (Ripser bindings)
4. Compute real Betti numbers from filtrations
5. Add missing SNO fields (τ, γ, χ)

### Phase 2: Neural Critics (Weeks 5-8)

**Goal:** Replace heuristics with neural models

1. DeBERTa Grounding Critic (use pre-trained)
2. GAT Logic Critic (train or pre-train)
3. Sentence-BERT Novelty Critic
4. Full chirality score with embeddings

### Phase 3: LLM Integration (Weeks 9-12)

**Goal:** Real agent generation

1. LLM provider integration (Anthropic/OpenAI)
2. Proposer claim extraction via LLM
3. Synthesizer generation with constraints
4. KCTS implementation
5. Evidence retrieval integration

### Phase 4: Training Pipeline (Weeks 13-16)

**Goal:** Enable model improvement

1. Complete Tinkex integration
2. Constitutional AI data generation
3. DPO implementation
4. Curriculum learning
5. Hybrid training loop with topological loss

### Phase 5: Validation (Weeks 17-20)

**Goal:** Verify against benchmarks

1. Dataset loading (SciFact, FEVER)
2. Automated evaluation pipeline
3. Ablation study infrastructure
4. Performance targets validation

---

## 8. Dependency Graph

```
FIM Computation
    └── Persistent Homology (Betti numbers)
    └── Full Chirality Score
    └── Topological Loss
    └── SNO τ, γ, χ fields

Embedding Model
    └── Chirality Score
    └── Novelty Critic
    └── Anti-neighbor retrieval (Antagonist)

LLM Provider
    └── Proposer (claim extraction)
    └── Synthesizer (generation)
    └── KCTS
    └── Constitutional AI

NLI Model (DeBERTa)
    └── Grounding Critic
    └── Evidence consistency (Antagonist)
    └── Trust Score

GAT Model
    └── Logic Critic
    └── β₁ calculation

Retrieval System
    └── Evidence retrieval (Antagonist)
    └── Counter-evidence finding

Tinkex
    └── LoRA training
    └── DPO
    └── Curriculum learning
```

---

## 9. Conclusion

The current Elixir CNS implementation provides a well-structured **scaffold** for CNS 3.0 but operates fundamentally differently from the spec:

**What exists:** Solid OTP architecture, correct data flow, comprehensive module structure, heuristic-based analysis

**What's missing:** The mathematical core (FIM, persistent homology), neural network critics, LLM integration, training infrastructure

**Effort estimate:** Approximately 16-20 weeks of focused development to reach spec compliance, assuming availability of:
- Nx/Axon for neural networks
- TDA library bindings
- LLM API access
- Training data

The codebase is well-positioned for this evolution - the module boundaries, data structures, and flow are correct. The primary work is replacing heuristic implementations with the mathematical and neural components specified in CNS 3.0.

---

*End of Gap Analysis*

# CNS 3.0 Theoretical Specification

**Document Type:** Theoretical Specification - What the System SHOULD Do
**Date:** 2025-11-21
**Source Documents:** CNS3 theoretical documentation from `/home/home/p/g/North-Shore-AI/tinkerer/cns3/`

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Core Algorithms](#2-core-algorithms)
3. [Data Structures](#3-data-structures)
4. [Metrics and Formulas](#4-metrics-and-formulas)
5. [Agent Behaviors](#5-agent-behaviors)
6. [Training Strategies](#6-training-strategies)
7. [Validation Benchmarks and Targets](#7-validation-benchmarks-and-targets)
8. [Theoretical Foundations](#8-theoretical-foundations)

---

## 1. Executive Summary

CNS 3.0 (Chiral Narrative Synthesis / Contradiction and Narrative Synthesis) is a neuro-symbolic framework for automated knowledge discovery from conflicting information. The system transforms dialectical reasoning into tractable optimization with convergence guarantees.

### Core Thesis
1. **Logical integrity is quantifiable by topological invariants** - The logical soundness of text is encoded in the *shape* of its semantic representation, measurable via persistent homology (Betti numbers)
2. **Semantic stability is quantifiable by geometric curvature** - The "fragility" or ambiguity of arguments is encoded in local geometry, measurable via Fisher Information Metric (FIM)

### Key Innovation
Unlike RAG systems (treat conflict as noise) or multi-agent debate (lack formal semantics), CNS explicitly represents thesis-antithesis relationships through algebraic topology and resolves contradictions through information-geometric synthesis operators.

---

## 2. Core Algorithms

### 2.1 Fisher Information Metric (FIM)

The canonical Riemannian metric for the statistical manifold:

```
g_μν(θ) = E[∂log p(x|θ)/∂θ_μ · ∂log p(x|θ)/∂θ_ν]
```

For discrete distributions (LLM vocabulary):
```
g_μν(θ) = Σ_x p(x|θ) · ∂log p(x|θ)/∂θ_μ · ∂log p(x|θ)/∂θ_ν
```

**Properties:**
- Symmetric Riemannian metric (unlike KL divergence which is asymmetric)
- Invariant to reparameterization
- Measures "distinguishability" between probability distributions
- High eigenvalues indicate semantic fragility regions

**Computation:**
1. Execute forward-backward pass on model
2. Construct FIM from gradients
3. Compute pairwise Fisher-Rao distance matrix
4. Use for topological analysis

### 2.2 Persistent Homology / Betti Numbers

**Workflow:**
1. Generate point cloud from SNO embeddings
2. Compute Fisher-Rao distances between all pairs
3. Build Vietoris-Rips filtration from distance matrix
4. Track birth/death of topological features as threshold ε increases
5. Extract Betti numbers

**Betti Number Interpretations:**
- **β₀ (connected components):** Argumentative coherence - elevated β₀ indicates semantic fragmentation
- **β₁ (1-dimensional loops):** Circular reasoning patterns - core hypothesis: β₁ correlates with logical circularity
- **β₂ (2-dimensional voids):** Surface coherence lacking internal support (speculative)

**Central Hypothesis:** Sound argumentation exhibits β₁ ≈ 0 and β₂ ≈ 0

### 2.3 Narrative Chirality Computation

Quantifies argumentative bias through geometric asymmetry:

```
χ(S₁, S₂) = α·(1 - cos_g(H₁, H₂))·(T₁·T₂) + β·GraphConflict(G₁, G₂)
```

Where:
- `cos_g` uses Fisher metric
- `GraphConflict` is typed contradiction density from simplicial complex cycles
- Achiral narrative: symmetric thesis/antithesis representation
- Chiral narrative: asymmetric embedding space

### 2.4 Graph Attention Network (GAT) for Logic Critic

**Architecture:**
- 3 layers
- 8 attention heads per layer
- 768-dim hidden dimensions (or 256/128 in some configs)
- Edge types: {entailment, contradiction, support, refutation}

**Attention Mechanism:**
```
e_ij = LeakyReLU(a^T [W h_i || W h_j || r_ij])
α_ij = softmax_j(e_ij)
h'_i = σ(Σ α_ij W h_j)
```

**Training:** 50K synthetic + 5K human-annotated graphs with binary cross-entropy

### 2.5 Synthesis Operator

Maps thesis-antithesis pairs to synthesis:
```
S: SNO × SNO × Evidence → SNO
```

**Properties:**
- Commutativity
- Associativity
- Identity
- Absorption
- Monotonicity

**Constraints:**
1. Evidence-preserving: `{e: w(e) > τ_min} ∩ (E_A ∪ E_B) ⊆ E_C`
2. Template-constrained generation: reference evidence IDs, no unsupported claims
3. Graph projection: remove contradiction cycles

### 2.6 Constrained Decoding (KCTS)

Knowledge-Constrained Tree Search during synthesis:
1. Generate synthesis
2. Extract factual claims
3. Verify each claim against evidence using Grounding Critic
4. If contradictions found, re-run with refinement prompts
5. Up to 3 iterations

### 2.7 LoRA Fine-Tuning

**Formulation:**
```
h = W₀x + (α/r)B(A(x))
```

Where:
- W₀ ∈ ℝ^(d×k) is pre-trained weight matrix
- B ∈ ℝ^(d×r), A ∈ ℝ^(r×k)
- r ≪ min(d, k) (rank)
- α is scaling factor

**Recommended Configuration:**
- Rank r = 16
- Alpha α = 32 (achieving α/r = 2)
- Target modules: ALL linear layers (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
- Learning rate: 2e-4

---

## 3. Data Structures

### 3.1 Structured Narrative Object (SNO)

**Basic Definition (5-tuple):**
```
SNO = (E, L, τ, γ, χ)
```

Where:
- **E**: Set of Events (text components, claims, evidence)
- **L**: Set of CausalLinks (dependencies between events)
- **τ**: TopologicalSignature {β₀, β₁, β₂}
- **γ**: GeometricSignature (FIM eigenvalues for fragile events)
- **χ**: Narrative Chirality (bias metric)

**Extended SNO-3 (7-tuple):**
```
S = (H, G, E, T, M, U, Θ_t)
```

Where:
- **H**: Hypothesis embedding ∈ ℝ^d on statistical manifold with Fisher metric
- **G**: Reasoning graph (V, E_G, ρ, τ) - typed DAG with relations
- **E**: Evidence set {(e_i, s(e_i), t_i, q_i)} with source, timestamp, quality
- **T**: Trust score ∈ [0,1] from critic ensemble
- **M**: Metadata (provenance, domain, licensing)
- **U**: Uncertainty (calibrated confidence, epistemic intervals)
- **Θ_t**: Temporal evolution kernel

**Alternative 6-tuple Definition:**
```
S = (H, E, G, τ, ρ, μ)
```
- **H**: Hypothesis in natural language
- **E**: Evidence set with quality scores
- **G**: Reasoning graph
- **τ**: Timestamp vector
- **ρ**: Source reliability weights
- **μ**: Metadata (chirality, confidence, provenance)

### 3.2 Simplicial Complex Representation

SNO as simplicial complex K:
- **0-simplices (vertices):** Claims C = {c_i}
- **1-simplices (edges):** Relations R = {(c_i, c_j)} for evidence, entailment, contradiction
- **k-simplices:** Higher-order relations (e.g., 2-simplex for c_i ∧ c_j ⟹ c_k)

### 3.3 Relation Types

```
R = {supports, contradicts, implies, equivalent, refines, causes}
```

Each edge has:
- Relation type ρ
- Confidence τ ∈ [0,1]

### 3.4 Antagonist Output Structure

```json
{
  "claim_id": "string",
  "issue_type": "POLARITY_CONTRADICTION | CIRCULAR_REASONING | EVIDENCE_INCONSISTENCY",
  "chirality_delta": "float",
  "evidence": ["snippets"],
  "critic_scores": {
    "entailment": "float",
    "chirality": "float",
    "beta1": "int"
  },
  "severity": "LOW | MED | HIGH"
}
```

---

## 4. Metrics and Formulas

### 4.1 Chirality Score

**Basic:**
```
χ(H₁, H₂) = Σᵢⱼ importance(cᵢ, H₁) · importance(cⱼ, H₂) · (1 - cos(embed(cᵢ), embed(cⱼ)))
```

Normalized χ̃ ∈ [0,1], with χ̃ > 0.7 indicating high opposition

**Extended with Fisher metric:**
```
CScore(S_i, S_j) = α·(1 - cos_g(H_i, H_j))·(T_i·T_j) + β·GraphConflict(G_i, G_j)
```

### 4.2 Evidential Entanglement

**Basic:**
```
EE(H₁, H₂) = Σᵢ wᵢ · q(eᵢ) · overlap(eᵢ, H₁, H₂) · exp(-λ · age(eᵢ))
```

**With source reliability network:**
```
w(e) = q_e · c(s(e)) · e^(-λ(t_now - t_e))

EScore(S_i, S_j) = Σ_{e ∈ E_i ∩ E_j} w(e) / Σ_{e ∈ E_i ∪ E_j} w(e)
```

Where:
- q_e = evidence quality
- c(s(e)) = source centrality (Bayesian PageRank)
- λ = temporal decay parameter

### 4.3 Trust Score

Adaptive weighting from critic ensemble:
```
T(S) = softmax(w)^T · [Score_G, Score_L, Score_N, Score_V, ...]
```

Where w is learned with human labels + outcome supervision

### 4.4 Critic Score

For Antagonist evidence consistency check:
```
critic_score = 1 - entailment_score
```

Flag when entailment_score < 0.5

### 4.5 Composite Loss Function

**For Synthesizer training:**
```
L_total = L_CE + λ₁·L_β₁ + λ₂·L_χ
```

Where:
- L_CE = standard cross-entropy
- L_β₁ = topological loss from Betti number
- L_χ = geometric loss from chirality

**REINFORCE for non-differentiable β₁:**
```
∇_θ J ≈ E[r · ∇_θ log p_θ(y|x)]
```
With r = -β₁ (minimize loops)

### 4.6 Bias Amplification Factor

```
α = bias(H_syn) / max(bias(H₁), bias(H₂))
α ≤ 1 + ρ
```

Where ρ = correlation(bias₁, bias₂)
- Independent biases: α ≤ 1
- Aligned biases: α ≤ 2

### 4.7 Information Preservation Bound

```
I(H_syn) ≥ min(I(H₁), I(H₂)) · (1 - δ)
```

Where δ = 1 - α (overlap coefficient)

For evidence-preserving synthesis:
```
I_C(θ) = Σ_{e ∈ E_C} w(e)·I_e(θ) ≥ min{I_A(θ), I_B(θ)}
```

### 4.8 Bregman Divergence

When extending to Bregman manifold:
```
F(θ) = A(θ) + λ·β₁(S)
```

Where A(θ) is log-partition from exponential family

---

## 5. Agent Behaviors

### 5.1 Proposer Agent (Thesis)

**Role:** Extract primary claims and evidence, generate initial SNOs

**Objective:** Structure raw text into computable representations

**Process:**
1. Ingest source text (corpus, papers, reports)
2. Extract all primary claims and evidence
3. Generate initial SNO set
4. Compute initial topological/geometric signatures

**Outputs:**
- SNO manifests (`snos.jsonl`, `manifest.json`)
- Per-claim embeddings
- Initial β₁ and chirality scores

### 5.2 Antagonist Agent (Antithesis)

**Role:** Find flaws, generate counter-arguments

**Objective Function:** MAXIMIZE β₁ (logical loops) and χ (chirality/bias)

**Primary Triggers:**
- chirality.score ≥ 0.55
- chirality.polarity_conflict == True
- β₁ > 0 (fallback for cycle detection)

**Behaviors:**

1. **Polarity Stress Tests (primary)**
   - Invert claim polarity (regex negation + embedding anti-neighbors)
   - Emit POLARITY_CONTRADICTION flags
   - Retrieve counter-evidence

2. **Evidence Consistency Check**
   - Use citation IDs + entailment scores
   - Flag when entailment < 0.5 even if β₁ = 0
   - Compute critic_score = 1 - entailment_score

3. **Residual β₁ Monitoring (fallback)**
   - Keep betti hook alive for future datasets
   - Generate cycle-busting countergraphs when β₁ > 0

**Outputs:**
- `artifacts/antagonist/<run>/flags.jsonl`
- Updated manifest with counter-SNOs
- Evidence snippets with conflict scores

**Target Metrics:**
- Precision ≥ 0.8
- Recall ≥ 0.7
- β₁ estimation error ≤ 10%
- Chirality delta coverage ≥ 0.9

### 5.3 Synthesizer Agent (Synthesis)

**Role:** Resolve conflicts, generate higher-level SNOs

**Objective Function:** MINIMIZE β₁ and χ

**Process:**
1. Receive original SNOs (Thesis) and Antagonist output (Antithesis)
2. Generate new text resolving contradictions
3. Balance biases
4. Maintain evidence traceability

**Constraints:**
- Evidence-preserving
- No unsupported claims
- Reference evidence IDs inline
- Address all conflict points

**Output:** Resolved SNO with:
- β₁ ≈ 0
- χ ≈ 0
- 100% citation traceability

**Training:**
- Constitutional AI with 6 dialectical principles
- DPO on preference pairs
- Constrained decoding (KCTS)

### 5.4 Critic Panel

**Logic Critic:**
- Architecture: GAT (3 layers, 8 heads)
- Input: Reasoning graph
- Output: Coherence score, detected cycles, β₁ calculation
- Justification: Superior for dense, nuanced SNO graphs

**Grounding Critic:**
- Architecture: DeBERTa-v3-large (434M params)
- Input: (claim, evidence) pairs
- Output: 3-class probability [Entails, Neutral, Contradicts]
- Final score: mean "Entails" probability

**Novelty Critic:**
- Architecture: Sentence-BERT twin network
- Method: Compare synthesis hypothesis against ANN index of existing hypotheses
- Score: Semantic distance to nearest neighbor

**Evidence Verification Critic:**
- Architecture: Cross-encoder (MiniLM-L-12)
- Computes quality scores q_i
- Updates source reliability network via Bayesian PageRank

**Optional Advanced Critics:**
- Causal Critic: FCI/PC-style verification
- Bias Critic: Disparity proxy computation
- Completeness Critic: Missing counter-argument detection

---

## 6. Training Strategies

### 6.1 Cold Start / Bootstrap Protocol

**Five-Stage Process:**

1. **Week 1 - Transfer Learning**
   - From pretrained NLI/GNN models
   - DeBERTa-v3 from MNLI/FEVER

2. **Weeks 2-3 - Weak Supervision**
   - Snorkel framework on 50K examples
   - LLM weak labels for critic targets

3. **Weeks 4-6 - Constitutional AI**
   - Generate 100K synthetic dialectics with GPT-4
   - Filter to 40K high-quality
   - Six dialectical principles

4. **Weeks 7-12 - Active Learning**
   - Uncertainty sampling
   - 5K expert-annotated examples
   - Target κ = 0.73

5. **Ongoing - Self-Improvement**
   - DPO on preference pairs
   - Generate → Score → Select → Fine-tune

### 6.2 Synthesizer Training

**Phase 1: Constitutional AI**
```
1. Generate synthetic with GPT-4
2. Self-critique and revision
3. Filter top 40%
4. Finetune 3 epochs, lr=5e-6
```

**Phase 2: Direct Preference Optimization (DPO)**
```
L_DPO = -E[log σ(β log(π_θ(y_w|x)/π_ref(y_w|x)) - β log(π_θ(y_l|x)/π_ref(y_l|x)))]
```

Hyperparameters:
- β = 0.1
- lr = 5e-7
- batch 64
- 3 epochs
- LoRA rank 16

**Improvements:** +9% synthesis quality, +5% factual consistency, +7% coherence vs SFT alone

### 6.3 Critic Training

**GAT Logic Critic:**
- 50K synthetic graphs + 5K human-annotated
- Binary cross-entropy
- 50 epochs, Adam lr=1e-3

**DeBERTa Grounding Critic:**
- FEVER + MultiFC + bootstrapped SNO pairs
- LR 2e-5, batch 64, 3-5 epochs
- Focal loss for hard contradictions

### 6.4 Curriculum Learning

**Progression:**
- Level 1: Simple contradictions (10K, 90% target)
- Level 2: Multi-premise arguments (15K, 85%)
- Level 3: Complex dialectics (10K, 80%)
- Level 4: Real-world synthesis (5K, 75%)

Competence-based with automatic difficulty scoring

### 6.5 Hybrid Training Loop

```python
for batch in dataset:
    # Forward pass
    outputs = model.forward_backward(batch)

    # Differentiable losses
    loss_ce = cross_entropy(outputs, targets)
    loss_chi = compute_chirality(outputs)

    # Topological features (expensive, periodic)
    if step % topo_freq == 0:
        fim = construct_FIM(model, outputs)
        distances = fisher_rao_distances(fim)
        betti_1 = persistent_homology(distances)
        reward = -betti_1
        loss_topo = reward * log_probs.detach()  # REINFORCE

    # Combined loss
    loss = loss_ce + lambda_chi * loss_chi + lambda_topo * loss_topo
    model.optim_step(loss)
```

**Computational Efficiency:**
- topo_freq = 10-100 steps
- FIM approximation via subset sampling or Kronecker-factored estimates

---

## 7. Validation Benchmarks and Targets

### 7.1 Primary Datasets

**SciFact:**
- 1,261 train / 450 dev / 300 test scientific claims
- 5,183 scientific abstracts as evidence
- **Target F1 ≥ 0.75-0.79** (SOTA: 77.8 MultiVerS)

**FEVER:**
- 145,449 training claims
- 9,999 dev / 9,999 test
- **Target Accuracy: 85-92%** (SOTA: >90%)

### 7.2 Novel Benchmarks

**SYNTH-DIAL:**
- 1,000 thesis-antithesis-synthesis triplets
- 10 domains
- Gold syntheses by 3 experts (κ > 0.8)

**HIST-SCI:**
- 3-500 historical scientific debates
- Germ theory, plate tectonics, quantum interpretation
- Ground truth: modern consensus

**DEBAGREEMENT:**
- 42,894 comment-reply pairs from Reddit
- Reformulated as synthesis task

**LegalPrecedent:**
- 1,200 cases with conflicting precedents
- IP (400), employment (300), constitutional (500)
- Expert annotation (κ = 0.71)

**INTEL-SIM:**
- Declassified contradictory reports
- Verified outcomes

### 7.3 Target Metrics

**Primary Hypothesis:**
- CNS ≥ 20% improvement over RAG baseline
- 100% evidence traceability

**Automated Metrics:**
- BERTScore F1: 0.883 (vs 0.854 baseline)
- Citation Precision: 99.8%
- Citation Recall: 99.2%
- NovAScore: 0.71
- SummaC Consistency: 0.892

**Human Evaluation (1-7 Likert):**
- Coherence: 6.2
- Novelty: 5.5
- Overall: 6.1
- Target κ > 0.75 inter-rater reliability

### 7.4 Ablation Targets

| Removed Component | Expected Degradation |
|-------------------|---------------------|
| Logic Critic | -7.9% quality, -11% coherence |
| Grounding Critic | -13.3% quality, -40% citation precision |
| Constitutional AI | -6.9% |
| DPO | -5.2% |
| Hybrid Retrieval | -10.5% quality |

### 7.5 Hypothesis Validation

**H1 (Component Necessity):**
- Each critic contributes uniquely and measurably

**H2 (Scaling Law):**
- Quality scales ~logarithmically with N
- Cost O(N log N)
- Stable 4.18-4.21 quality across 10³-10⁶ SNOs

**H3 (Domain Transfer):**
- Zero-shot retains ≥70% source performance

**H4 (Entanglement Utility):**
- High-C/high-E pairs > high-C/low-E by ≥15%

### 7.6 Core Assumption Validation

**Hypothesis 1 (β₁ ↔ Circular Reasoning):**
- 200-500 expert-annotated examples
- Target correlation r > 0.4 with p < 0.01

**Hypothesis 2 (FIM ↔ Semantic Fragility):**
- High-curvature regions show 2-3× larger semantic shifts
- Compare against simple heuristics

**Hypothesis 3 (χ ↔ Argumentative Bias):**
- Compare against MBIC, AllSides ratings
- Target AUC > 0.65

**Go/No-Go:** If H1-2 fail, pivot to TDA as auxiliary features

### 7.7 Antagonist MVP Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| Precision | TRUE flags / issued flags | ≥ 0.8 |
| Recall | TRUE flags / known contradictions | ≥ 0.7 |
| β₁ estimation error | vs logic/betti.py | ≤ 10% |
| Chirality delta coverage | High-χ cases with flags | ≥ 0.9 |

### 7.8 Statistical Protocol

- Power analysis: n ≥ 64 per condition for d=0.5 at α=0.05
- Paired t-tests (Bonferroni corrected)
- Wilcoxon for human ratings
- Report Cohen's d and bootstrap CIs

---

## 8. Theoretical Foundations

### 8.1 Theorem: Synthesis Coherence

**Statement:** If (i) G_A, G_B are individually consistent; (ii) EScore ≥ κ; (iii) verifiers have bounded error rates; (iv) Φ enforces evidence-preservation and graph projection, then:

```
P[S_C is coherent] ≥ 1 - ε
ε ≤ ε_NLI + ε_logic + ε_verify - δ
```

**Proof Sketch:** Union bound with overlap correction; projection Π eliminates homology-1 contradiction cycles

### 8.2 Theorem: Dialectical Convergence

**Statement:** Under contractivity (∃k ∈ (0,1)), monotonicity, and completeness, sequence {S_n} converges to unique fixed point S* with exponential rate:

```
||S_n - S*|| ≤ k^n ||S₀ - S*||
```

**Proof Sketch:** Banach Fixed-Point Theorem. Show T = Π ∘ Φ is γ-contractive in product metric d = d_H ⊕ d_G ⊕ d_E

**Practical:** k < 0.9 converges within 10 iterations; DPO-trained achieves k ≈ 0.7-0.85

### 8.3 Theorem: Information Preservation

**Statement:** For evidence-preserving synthesis with MLE:

```
I_C(θ) ≥ min{I_A(θ), I_B(θ)}
```

**Proof Sketch:** Chentsov's theorem (FIM is unique invariant metric); additivity of observed Fisher information for independent observations

### 8.4 Theorem: Bias Amplification Bounds

**Statement:** For f_C = αf_A + βf_B + Δ with Lip(L)-constrained Δ:

```
B(f_C; P) ≤ α·B(f_A; P) + β·B(f_B; P) + L·Disc(P)
```

**Proof Sketch:** Triangle inequality and Lipschitz stability; Δ is small when Bias Critic penalizes disparity

**Key Insight:** Synthesis selecting high-chirality pairs forces GNN to process graph with LOW homophily, breaking bias-amplifying feedback

### 8.5 Complexity Analysis

- Pairing: O(N log N) with ANN pre-filtering
- Graph conflict: O(|V_i||V_j|) (near-linear with sparse adjacency)
- GNN inference: O(|V| + |E|) per graph
- Overall: Quasi-linear in SNO population

---

## Appendix A: Key Configuration Parameters

### A.1 LoRA Configuration
```yaml
rank: 16
alpha: 32
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
learning_rate: 2e-4
```

### A.2 GAT Configuration
```yaml
layers: 3
attention_heads: 8
hidden_dim: 768  # or 256
edge_types: [entailment, contradiction, support, refutation]
training_samples: 55000
epochs: 50
optimizer: Adam
lr: 1e-3
```

### A.3 Antagonist Thresholds
```yaml
chirality_trigger: 0.55
polarity_conflict_trigger: true
max_retrieval_fanout: TBD
critic_weights: TBD
evidence_entailment_threshold: 0.5
```

### A.4 Synthesis Constraints
```yaml
temperature: 0.2
nucleus_p: 0.9
max_iterations: 3
evidence_id_format: "[evidence_id]"
constraint_enforcement: regex_gating + post_hoc_verification
```

### A.5 Topological Loss
```yaml
topo_freq: 10-100  # compute every N steps
lambda_chi: TBD
lambda_topo: TBD
```

---

## Appendix B: Model Specifications

| Component | Model | Parameters | Key Specs |
|-----------|-------|------------|-----------|
| Retrieval | BGE-M3 | 568M | 1024-dim, 8192 context |
| Logic Critic | GAT | ~50M | 3-layer, 8-head |
| Grounding Critic | DeBERTa-v3-large | 434M | 91.2% MNLI |
| Synthesis | Llama-3.1-70B | 70B | 128K context |
| Novelty | Sentence-BERT | 110M | all-mpnet-base-v2 |

---

## Appendix C: Bregman Manifold Extension (Future)

**Convex Potential:**
```
F(θ) = A(θ) + λ·β₁(S)
```

**Hessian Metric:**
```
g = ∇²F
```

**Dual Connections:**
- Proposer operates in natural coordinates (θ)
- Antagonist evaluates in dual coordinates (η = ∇F(θ))
- Synthesizer walks geodesics averaging both

**Bregman Divergence:**
Replaces cosine distance in Chirality Score while maintaining β₁-weighted conflict density

---

## Appendix D: File Locations

**Source Documentation:**
- `tinkerer/cns3/20251109_revised_cns_proposal_for_thinking_machines.md`
- `tinkerer/cns3/20251118_antagonist_mvp_rfc.md`
- `tinkerer/cns3/Bregman_Manifold_Design_Note.md`
- `tinkerer/cns3/cns3_gemini_deepResearch.md`
- `tinkerer/cns3/cns3_gpt5.md`
- `tinkerer/cns3/20251109_technicalValidation_CNSSupportModelsScientificProposal.md`
- `tinkerer/cns3/CNS_3_0_A_DIALECTICAL_FRAMEWORK_FOR_AUTOMATED_KNOWLEDGE_DISCOVERY.md`

**Artifact Outputs:**
- `runs/thinker_eval/<run>.jsonl`
- `snos.jsonl`, `manifest.json`
- `artifacts/antagonist/<run>/flags.jsonl`
- `artifacts/logic/*.jsonl`

---

*End of CNS 3.0 Theoretical Specification*

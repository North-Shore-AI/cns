# CNS Architecture Redesign - Technical Specification

**Date:** 2025-11-24
**Status:** Design Proposal
**Author:** Architecture Analysis Agent

---

## Executive Summary

This document proposes a comprehensive architectural redesign of the CNS (Chiral Narrative Synthesis) library to:

1. **Clean separation** between pure topology/math and CNS domain logic
2. **Simplified public API** that matches original mathematical intentions
3. **Remove stub implementations** and consolidate duplicate functionality
4. **Enable Crucible integration** while maintaining CNS as standalone library
5. **Defer full TDA** until surrogate validation proves correlation

**Key Decision:** Do NOT create a separate `ex_topology` library yet. Keep minimal topology implementations within CNS until validation proves the need for full persistent homology.

---

## Table of Contents

1. [Current State Analysis](#current-state-analysis)
2. [Original Math Requirements](#original-math-requirements)
3. [Gap Analysis](#gap-analysis)
4. [Proposed New Surface](#proposed-new-surface)
5. [Module Responsibility Matrix](#module-responsibility-matrix)
6. [Deprecation Plan](#deprecation-plan)
7. [Topology Strategy](#topology-strategy)
8. [Migration Guide](#migration-guide)
9. [Implementation Phases](#implementation-phases)

---

## Current State Analysis

### Existing Module Inventory

Based on exploration of `S:\cns\lib\cns\`:

#### **Core Data Structures** (KEEP - Production Ready)

| Module | Lines | Status | Purpose |
|--------|-------|--------|---------|
| `sno.ex` | 409 | ✅ Production | Structured Narrative Objects |
| `evidence.ex` | ~150 | ✅ Production | Evidence with provenance |
| `challenge.ex` | ~120 | ✅ Production | Antagonist challenges |
| `provenance.ex` | ~100 | ✅ Production | Lineage tracking |
| `config.ex` | ~180 | ✅ Production | Configuration management |

#### **Dialectical Agents** (KEEP - Production Ready)

| Module | Lines | Status | Purpose |
|--------|-------|--------|---------|
| `proposer.ex` | 8,323 bytes | ✅ Production | Claim extraction, thesis generation |
| `antagonist.ex` | 10,617 bytes | ✅ Production | Challenge generation, antithesis |
| `synthesizer.ex` | 11,128 bytes | ✅ Production | Dialectical synthesis |
| `pipeline.ex` | 7,961 bytes | ✅ Production | Orchestration, convergence |

#### **Validation Layer** (KEEP - Production Ready)

| Module | Lines | Status | Purpose |
|--------|-------|--------|---------|
| `validation/semantic.ex` | 12,055 bytes | ✅ Production | NLI-based validation |
| `validation/citation.ex` | ~120 | ✅ Production | Citation checking |
| `validation/model_loader.ex` | ~120 | ✅ Production | Bumblebee model management |

#### **Critics System** (KEEP - Production Ready)

| Module | Status | Purpose |
|--------|--------|---------|
| `critics/critic.ex` | ✅ Behaviour | Base critic interface |
| `critics/logic.ex` | ✅ Production | Logical consistency |
| `critics/grounding.ex` | ✅ Production | Evidence grounding |
| `critics/causal.ex` | ✅ Production | Causal reasoning |
| `critics/bias.ex` | ✅ Production | Bias detection |
| `critics/novelty.ex` | ✅ Production | Novelty assessment |

#### **Topology & Metrics** (MIXED - Needs Consolidation)

| Module | Status | Notes |
|--------|--------|-------|
| `topology.ex` | ⚠️ Mixed | Graph analysis, Betti numbers, cycles |
| `topology/surrogates.ex` | ✅ NEW | β₁ and fragility surrogates |
| `metrics.ex` | ✅ Production | Quality metrics, chirality |
| `metrics/chirality.ex` | ⚠️ Duplicate? | May overlap with metrics.ex |

#### **Training** (TRANSITION - V2 Replaces V1)

| Module | Status | Notes |
|--------|--------|-------|
| `training.ex` | ⚠️ Legacy | Uses old crucible_contracts |
| `training_v2.ex` | ✅ NEW | Uses Crucible IR |
| `training/evaluation.ex` | ✅ Production | Metrics computation |

#### **Crucible Integration** (NEW)

| Module | Status | Notes |
|--------|--------|-------|
| `crucible_adapter.ex` | ✅ NEW | Implements Crucible.CNS.Adapter |
| `crucible_contracts/*` | ❌ DEPRECATED | Old integration contracts |

#### **Pipeline Schema** (KEEP)

| Module | Status | Purpose |
|--------|--------|---------|
| `pipeline/schema.ex` | ✅ Production | CLAIM[...] parsing |
| `pipeline/converters.ex` | ✅ Production | SciFact converters |

#### **Supporting Modules** (KEEP)

| Module | Status | Purpose |
|--------|--------|---------|
| `application.ex` | ✅ Production | OTP application |
| `util.ex` | ✅ Production | Utilities |

### Total Module Count

- **Production-ready**: ~35 modules
- **New (workstream)**: 3 modules
- **Deprecated**: 4 modules (crucible_contracts)
- **Needs work**: ~5 modules (consolidation, cleanup)

---

## Original Math Requirements

### From Tinkerer Research

Analysis of `S:\tinkerer\brainstorm\` documents reveals:

#### **CNS 2.0 Mathematical Foundations**

**Structured Narrative Object (SNO) - 4-6 Tuple:**
```
SNO = (H, E, G, T, [C], [M])

Where:
  H = Hypothesis/claim (text)
  E = Evidence set {e₁, e₂, ..., eₙ}
  G = Reasoning graph (causal links between evidence and hypothesis)
  T = Trust score ∈ [0, 1]
  C = Confidence intervals (optional)
  M = Metadata (provenance, timestamps, etc.)
```

**Chirality Score - Fisher-Rao Distance:**
```
χ(SNO_A, SNO_B) = d_FR(P_A, P_B)

Where:
  P_A, P_B = Probability distributions over semantic space
  d_FR = Fisher-Rao geodesic distance
  High χ indicates semantic opposition
```

**Evidential Entanglement:**
```
ε(SNO_A, SNO_B) = |E_A ∩ E_B| / |E_A ∪ E_B|

Where evidence sets overlap but hypotheses oppose
High ε + High χ = Productive dialectical conflict
```

#### **CNS 3.0 Topological Extensions**

**Betti Numbers for Circular Reasoning Detection:**
```
β₁(G) = # independent cycles in reasoning graph G

β₁ = 0 → DAG structure (acyclic reasoning)
β₁ > 0 → Circular reasoning present
```

**Persistence Diagrams:**
```
PD = {(b_i, d_i) | feature i born at b_i, dies at d_i}

Long-lived features = robust reasoning
Short-lived features = fragile arguments
```

**Fragility via Embedding Variance:**
```
F(SNO) = Var(d(emb(SNO), emb(SNO + δ)))

Where δ = small semantic perturbation
High F = argument changes under slight rewording
```

#### **Dialectical Pipeline**

**Three-Agent Process:**
```
Thesis (Proposer)
    ↓
Antithesis (Antagonist)
    ↓
Synthesis (Synthesizer)
    ↓
Convergence Check → Iterate or Finalize
```

**Convergence Criteria:**
```
Converged(SNO_t, SNO_{t-1}) iff:
  1. Confidence(SNO_t) ≥ threshold
  2. Coherence(SNO_t) ≥ threshold
  3. |SNO_t - SNO_{t-1}| < ε
```

---

## Gap Analysis

### What's Missing from Current Implementation

#### 1. **Full Persistent Homology** (Intentionally Deferred)

**Status:** Surrogate implementations exist, full TDA not implemented

**Reasoning:**
- Gate 1 validation required before investing in O(N³) algorithms
- Surrogates (β₁ approximation via cycle detection) sufficient for validation
- Full TDA can be added if surrogates prove correlation

**Recommendation:** DEFER until Gate 1 passes

#### 2. **Fisher Information Metric** (Partially Implemented)

**Status:** Chirality score uses simplified distance metrics

**Current Implementation:**
```elixir
# CNS.Metrics.chirality/2
# Uses cosine distance, not true Fisher-Rao geodesic
```

**Gap:** No FIM computation for semantic embeddings

**Recommendation:**
- Keep simplified chirality for now
- Document as "chirality approximation"
- Add true FIM if needed for publication

#### 3. **Evidential Entanglement** (Not Implemented)

**Status:** Concept from CNS 2.0, not coded

**Gap:** No function to compute evidence set overlap with opposing hypotheses

**Recommendation:** ADD in Phase 2 as `CNS.Metrics.evidential_entanglement/2`

#### 4. **Confidence Intervals** (Not Implemented)

**Status:** SNO has confidence score but no intervals

**Gap:** No uncertainty quantification

**Recommendation:** DEFER - not critical for dialectical reasoning

### What's Duplicated

#### 1. **Topology Metrics**

**Files:**
- `topology.ex` - Contains Betti number computation
- `topology/surrogates.ex` - Contains β₁ approximation

**Issue:** Overlap in cycle detection

**Resolution:** Keep surrogates, use `topology.ex` for exact computation if needed

#### 2. **Training Modules**

**Files:**
- `training.ex` - Uses legacy crucible_contracts
- `training_v2.ex` - Uses Crucible IR

**Resolution:** Deprecate `training.ex`, make `training_v2.ex` canonical

#### 3. **Metrics Scattered**

**Files:**
- `metrics.ex` - General quality metrics
- `metrics/chirality.ex` - If exists, may duplicate

**Resolution:** Consolidate into single `metrics.ex` or clear namespace

### What Should Be Removed

| File/Module | Reason | Action |
|-------------|--------|--------|
| `crucible_contracts/*.ex` | Legacy integration | DELETE after migration |
| `training.ex` | Replaced by training_v2 | DEPRECATE → DELETE |
| Graph stubs | If exist, incomplete | DELETE or complete |

---

## Proposed New Surface

### Public API Design

The CNS library should expose **three primary interfaces**:

#### 1. **High-Level Dialectical API** (Primary)

```elixir
defmodule CNS do
  @moduledoc """
  Chiral Narrative Synthesis - Dialectical reasoning for claim extraction.

  CNS implements a three-agent dialectical process:
  - Proposer: Extracts claims from text (thesis)
  - Antagonist: Generates challenges (antithesis)
  - Synthesizer: Reconciles into refined claims (synthesis)

  ## Examples

      # Extract claims from text
      {:ok, claims} = CNS.extract_claims(scientific_text)

      # Run full dialectical pipeline
      config = CNS.Config.new(max_iterations: 5)
      {:ok, result} = CNS.run_pipeline(research_question, config)

      # Direct synthesis of opposing claims
      {:ok, synthesis} = CNS.synthesize(thesis, antithesis)
  """

  @type sno :: CNS.SNO.t()
  @type config :: CNS.Config.t()
  @type pipeline_result :: %{
    final_sno: sno(),
    iterations: non_neg_integer(),
    convergence_score: float(),
    trace: [sno()]
  }

  @doc """
  Extract claims from text using Proposer agent.
  """
  @spec extract_claims(text :: String.t(), opts :: keyword()) ::
    {:ok, [sno()]} | {:error, term()}

  @doc """
  Run full dialectical pipeline until convergence.
  """
  @spec run_pipeline(input :: String.t(), config) ::
    {:ok, pipeline_result()} | {:error, term()}

  @doc """
  Synthesize thesis and antithesis into refined claim.
  """
  @spec synthesize(thesis :: sno(), antithesis :: sno(), opts :: keyword()) ::
    {:ok, sno()} | {:error, term()}

  @doc """
  Validate claim against evidence corpus.
  """
  @spec validate(sno(), corpus :: [map()], opts :: keyword()) ::
    {:ok, validation_result()} | {:error, term()}
end
```

#### 2. **Topology Analysis API**

```elixir
defmodule CNS.Topology do
  @moduledoc """
  Topological analysis of claim networks and reasoning graphs.

  Provides lightweight surrogates for topological validation and
  optional full persistent homology computation.

  ## Surrogates (Fast - O(V+E))

  - β₁ approximation via cycle detection
  - Fragility via k-NN embedding variance

  ## Full TDA (Slow - O(N³), if enabled)

  - Exact Betti numbers
  - Persistence diagrams
  - Simplicial complex analysis
  """

  @doc """
  Analyze claim network for circular reasoning.

  Returns β₁ approximation (# independent cycles).
  """
  @spec analyze_claim_network([CNS.SNO.t()]) ::
    %{beta1: non_neg_integer(), cycles: [cycle()], dag?: boolean()}

  @doc """
  Detect circular reasoning in reasoning graph.
  """
  @spec detect_circular_reasoning(CNS.SNO.t()) ::
    {:ok, [cycle()]} | {:error, term()}

  @doc """
  Compute fragility of claim under semantic perturbation.
  """
  @spec compute_fragility(CNS.SNO.t() | [CNS.SNO.t()], opts :: keyword()) ::
    float()

  @doc """
  Compute exact Betti numbers (requires full TDA).
  """
  @spec betti_numbers(reasoning_graph :: Graph.t(), max_dim :: pos_integer()) ::
    %{beta0: integer(), beta1: integer(), beta2: integer()}
end
```

#### 3. **Metrics API**

```elixir
defmodule CNS.Metrics do
  @moduledoc """
  CNS-specific metrics for dialectical quality.

  Implements mathematical measures from CNS 2.0/3.0 specifications.
  """

  @doc """
  Compute chirality score between opposing claims.

  Approximates Fisher-Rao distance via cosine distance in embedding space.
  High chirality indicates semantic opposition.
  """
  @spec chirality(CNS.SNO.t(), CNS.SNO.t()) :: float()

  @doc """
  Compute evidential entanglement between claims.

  Measures evidence set overlap for opposing hypotheses.
  High entanglement + high chirality = productive conflict.
  """
  @spec evidential_entanglement(CNS.SNO.t(), CNS.SNO.t()) :: float()

  @doc """
  Compute convergence score for dialectical iteration.

  Returns score ∈ [0, 1] indicating convergence stability.
  """
  @spec convergence_score(previous :: CNS.SNO.t(), current :: CNS.SNO.t()) ::
    float()

  @doc """
  Compute overall quality score per CNS 3.0 targets.

  Combines schema compliance, citation accuracy, and entailment.
  """
  @spec overall_quality(CNS.SNO.t(), opts :: keyword()) ::
    %{score: float(), meets_threshold: boolean(), breakdown: map()}
end
```

### Internal Module Organization

```
lib/cns/
├── cns.ex                           # Public facade (high-level API)
│
├── core/                            # Domain entities
│   ├── sno.ex                      # SNO structure
│   ├── evidence.ex                 # Evidence management
│   ├── challenge.ex                # Challenge structure
│   ├── provenance.ex               # Provenance tracking
│   └── config.ex                   # Configuration
│
├── agents/                          # Dialectical agents
│   ├── proposer.ex                 # Thesis generation
│   ├── antagonist.ex               # Antithesis generation
│   ├── synthesizer.ex              # Synthesis generation
│   └── pipeline.ex                 # Orchestration
│
├── topology/                        # Topological analysis
│   ├── topology.ex                 # Public API
│   ├── surrogates.ex               # Fast approximations
│   ├── graph_analysis.ex           # Graph algorithms
│   └── tda.ex                      # Full TDA (optional)
│
├── metrics/                         # CNS-specific metrics
│   ├── metrics.ex                  # Public API
│   ├── chirality.ex                # Opposition measurement
│   ├── entanglement.ex             # Evidence overlap
│   └── convergence.ex              # Iteration stability
│
├── validation/                      # Quality validation
│   ├── semantic.ex                 # NLI-based validation
│   ├── citation.ex                 # Citation checking
│   ├── schema.ex                   # CLAIM[...] parsing
│   └── model_loader.ex             # Bumblebee models
│
├── critics/                         # Multi-component evaluation
│   ├── critic.ex                   # Behaviour
│   ├── logic.ex                    # Logical consistency
│   ├── grounding.ex                # Evidence grounding
│   ├── causal.ex                   # Causal reasoning
│   ├── bias.ex                     # Bias detection
│   └── novelty.ex                  # Novelty assessment
│
├── crucible/                        # Crucible integration
│   └── adapter.ex                  # Crucible.CNS.Adapter impl
│
└── training/                        # Training pipeline
    ├── training.ex                 # Main API (was training_v2)
    └── evaluation.ex               # Metrics computation
```

---

## Module Responsibility Matrix

| Module | Responsibility | Dependencies | Public API? |
|--------|---------------|--------------|-------------|
| **CNS** | High-level facade | All internal modules | ✅ YES |
| **CNS.SNO** | Data structure | Ecto (for persistence) | ✅ YES |
| **CNS.Evidence** | Evidence management | None | ✅ YES |
| **CNS.Challenge** | Challenge structure | None | ✅ YES |
| **CNS.Provenance** | Lineage tracking | None | ✅ YES |
| **CNS.Config** | Configuration | NimbleOptions | ✅ YES |
| **CNS.Agents.Proposer** | Thesis generation | SNO, Evidence | ❌ Internal |
| **CNS.Agents.Antagonist** | Antithesis generation | SNO, Challenge | ❌ Internal |
| **CNS.Agents.Synthesizer** | Synthesis | SNO, Evidence | ❌ Internal |
| **CNS.Agents.Pipeline** | Orchestration | All agents | ❌ Internal |
| **CNS.Topology** | Topology API | Surrogates, TDA | ✅ YES |
| **CNS.Topology.Surrogates** | Fast approximations | :libgraph | ❌ Internal |
| **CNS.Topology.TDA** | Full persistent homology | External TDA lib | ❌ Internal |
| **CNS.Metrics** | Metrics API | Chirality, etc | ✅ YES |
| **CNS.Metrics.Chirality** | Opposition metric | Nx, Scholar | ❌ Internal |
| **CNS.Metrics.Entanglement** | Evidence overlap | None | ❌ Internal |
| **CNS.Metrics.Convergence** | Iteration stability | None | ❌ Internal |
| **CNS.Validation.Semantic** | NLI validation | Bumblebee | ✅ YES |
| **CNS.Validation.Citation** | Citation checking | None | ✅ YES |
| **CNS.Validation.Schema** | CLAIM parsing | None | ❌ Internal |
| **CNS.Critics.\*** | Quality evaluation | SNO, Metrics | ✅ YES |
| **CNS.Crucible.Adapter** | Crucible integration | All | ❌ Internal |
| **CNS.Training** | Training pipeline | Crucible IR | ✅ YES |

---

## Deprecation Plan

### Phase 1: Mark as Deprecated (Week 1)

Add deprecation warnings to:

```elixir
# lib/cns/training.ex
@deprecated "Use CNS.Training (formerly training_v2) instead"
defmodule CNS.TrainingLegacy do
  # ...
end

# lib/cns/crucible_contracts/*.ex
@deprecated "Use CNS.Crucible.Adapter instead"
defmodule CNS.CrucibleContracts.Lora do
  # ...
end
```

### Phase 2: Remove from Documentation (Week 2)

- Remove from README examples
- Update guides to use new API
- Add migration guide

### Phase 3: Delete Code (Week 4)

After ensuring no internal usage:

```bash
# Remove legacy files
rm lib/cns/training.ex
rm -rf lib/cns/crucible_contracts/

# Consolidate if needed
# (e.g., merge scattered metric files)
```

### Migration Path

```elixir
# Old (training.ex)
CNS.Training.train(snos, base_model: "llama", format: :dialectical)

# New (training_v2.ex → training.ex)
CNS.Training.train(snos, base_model: "llama", format: :dialectical)
# (Same API, different implementation)

# Old (crucible_contracts)
adapter = CNS.CrucibleContracts.Lora
adapter.train_step(...)

# New (crucible adapter)
{:ok, context} = CrucibleFramework.run(experiment)
# CNS integration via Crucible.CNS.Adapter behaviour
```

---

## Topology Strategy

### Decision: Do NOT Extract to ex_topology

**Reasoning:**

1. **ex_topology doesn't exist** - Would require creating new library
2. **CNS-specific topology** - Much of the topology is claim-network specific
3. **Surrogates are sufficient** - O(V+E) approximations work for validation
4. **Deferred TDA** - Full persistent homology only if Gate 1 passes
5. **Maintenance burden** - Separate library adds complexity

### Topology Module Structure

```elixir
defmodule CNS.Topology do
  @moduledoc """
  Topological analysis for CNS claim networks.

  ## Surrogates (Default)
  Fast O(V+E) approximations suitable for validation.

  ## Full TDA (Optional)
  Requires enabling :full_tda in config and installing TDA backend.
  """

  @doc "Analyze claim network (uses surrogates by default)"
  def analyze_claim_network(snos, opts \\ [])

  @doc "Compute β₁ (cycle count)"
  def beta1(snos_or_graph, mode: :surrogate | :exact)

  @doc "Compute fragility (embedding variance)"
  def fragility(snos, opts \\ [])
end

defmodule CNS.Topology.Surrogates do
  @moduledoc """
  Lightweight surrogates for topological validation.

  Uses Tarjan's SCC for cycle detection (β₁ approximation)
  and k-NN variance for fragility.
  """

  def compute_beta1_surrogate(causal_links)
  def compute_fragility_surrogate(embeddings, opts \\ [])
end

defmodule CNS.Topology.TDA do
  @moduledoc """
  Full topological data analysis (optional).

  Only loaded if :full_tda enabled in config.
  Requires external TDA library (Python interop or native).
  """

  def compute_betti_numbers(distance_matrix, max_dim)
  def compute_persistence_diagram(point_cloud)
end
```

### Dependencies

```elixir
# mix.exs
defp deps do
  [
    # Core (required)
    {:libgraph, "~> 0.16"},    # Graph algorithms
    {:nx, "~> 0.7"},           # Numerical computing
    {:scholar, "~> 0.2"},      # Distance metrics

    # Optional (for full TDA)
    {:python, "~> 0.4", optional: true},  # Python interop
    {:ripser, github: "...", optional: true}  # If native Elixir TDA
  ]
end
```

### Configuration

```elixir
# config/config.exs
config :cns,
  # Topology mode: :surrogates (default) or :full_tda
  topology_mode: :surrogates,

  # If :full_tda, specify backend
  tda_backend: :python  # or :ripser_native
```

---

## Migration Guide

### For Library Users

#### Updating Import Statements

```elixir
# Old
alias CNS.{Proposer, Antagonist, Synthesizer, Pipeline}

# New
alias CNS  # Use high-level API
# Or for specific modules:
alias CNS.{Topology, Metrics, Validation}
```

#### Updating Function Calls

```elixir
# Extract Claims
# Old
{:ok, claims} = Proposer.extract_claims(text)

# New
{:ok, claims} = CNS.extract_claims(text)

# Run Pipeline
# Old
config = Pipeline.Config.new()
{:ok, result} = Pipeline.run(input, config)

# New
config = CNS.Config.new()
{:ok, result} = CNS.run_pipeline(input, config)

# Validate
# Old
{:ok, validation} = CNS.Validation.Semantic.validate(sno)

# New
{:ok, validation} = CNS.validate(sno, corpus)
```

#### Topology Functions

```elixir
# Surrogates
# Old
beta1 = CNS.Topology.Surrogates.compute_beta1_surrogate(links)

# New
%{beta1: beta1} = CNS.Topology.analyze_claim_network(snos)
# Or directly:
beta1 = CNS.Topology.beta1(snos, mode: :surrogate)

# Fragility
# Old
fragility = CNS.Topology.Surrogates.compute_fragility_surrogate(embeddings)

# New
fragility = CNS.Topology.fragility(snos)
```

### For Contributors

#### Module Reorganization

If contributing to CNS core:

1. **Don't add to root `lib/cns/`** - Use namespaces (agents, topology, metrics, etc.)
2. **Update tests** - Move tests to match new structure
3. **Add @moduledoc** - All public modules need docs
4. **Add @spec** - All public functions need typespecs

#### Testing

```bash
# Test specific namespace
mix test test/cns/topology/
mix test test/cns/agents/

# Test public API
mix test test/cns_test.exs
```

---

## Implementation Phases

### Phase 1: Reorganization (Week 1)

**Goal:** Clean module structure without breaking existing code

**Tasks:**
1. Create new directory structure:
   ```bash
   mkdir -p lib/cns/{agents,topology,metrics,crucible,training}
   ```

2. Move files to new locations (use `git mv` to preserve history):
   ```bash
   git mv lib/cns/proposer.ex lib/cns/agents/
   git mv lib/cns/antagonist.ex lib/cns/agents/
   git mv lib/cns/synthesizer.ex lib/cns/agents/
   git mv lib/cns/pipeline.ex lib/cns/agents/
   ```

3. Update module names:
   ```elixir
   # Old
   defmodule CNS.Proposer
   # New
   defmodule CNS.Agents.Proposer
   ```

4. Add aliases for backward compatibility:
   ```elixir
   # lib/cns/proposer.ex (temporary)
   defmodule CNS.Proposer do
     @deprecated "Use CNS.Agents.Proposer or CNS.extract_claims/2"
     defdelegate extract_claims(text, opts \\ []),
       to: CNS.Agents.Proposer
   end
   ```

5. Update all internal references

6. Run full test suite - ensure no breaks

**Deliverable:** New structure with backward compatibility

---

### Phase 2: Public API Stabilization (Week 2)

**Goal:** Define and document public API

**Tasks:**
1. Create `lib/cns.ex` facade module:
   ```elixir
   defmodule CNS do
     @moduledoc """
     Chiral Narrative Synthesis - Dialectical reasoning for claims.
     """

     defdelegate extract_claims(text, opts \\ []),
       to: CNS.Agents.Proposer

     defdelegate run_pipeline(input, config),
       to: CNS.Agents.Pipeline, as: :run

     defdelegate synthesize(thesis, antithesis, opts \\ []),
       to: CNS.Agents.Synthesizer

     defdelegate validate(sno, corpus, opts \\ []),
       to: CNS.Validation.Semantic
   end
   ```

2. Create `lib/cns/topology.ex` facade:
   ```elixir
   defmodule CNS.Topology do
     defdelegate analyze_claim_network(snos, opts \\ []),
       to: CNS.Topology.Surrogates

     def beta1(snos_or_graph, opts \\ []) do
       mode = Keyword.get(opts, :mode, :surrogate)
       # Dispatch to surrogate or exact
     end

     defdelegate fragility(snos, opts \\ []),
       to: CNS.Topology.Surrogates, as: :compute_fragility_surrogate
   end
   ```

3. Create `lib/cns/metrics.ex` facade:
   ```elixir
   defmodule CNS.Metrics do
     defdelegate chirality(sno_a, sno_b),
       to: CNS.Metrics.Chirality, as: :compute

     def evidential_entanglement(sno_a, sno_b) do
       # Implement evidence set overlap
     end

     defdelegate convergence_score(prev, curr),
       to: CNS.Metrics.Convergence, as: :compute
   end
   ```

4. Add comprehensive @moduledoc to all public modules

5. Add @spec to all public functions

6. Generate ExDoc documentation:
   ```bash
   mix docs
   open doc/index.html
   ```

**Deliverable:** Complete public API documentation

---

### Phase 3: Deprecation & Cleanup (Week 3)

**Goal:** Remove legacy code

**Tasks:**
1. Mark legacy modules as deprecated:
   ```elixir
   @deprecated "Use CNS.Training instead"
   defmodule CNS.TrainingLegacy
   ```

2. Add warnings to legacy crucible_contracts:
   ```elixir
   @moduledoc """
   DEPRECATED: Use CNS.Crucible.Adapter instead.

   This module will be removed in v0.3.0.
   """
   ```

3. Update all examples in README

4. Update guides in `docs/`

5. Add migration guide: `docs/MIGRATION_V0_2_TO_V0_3.md`

6. Run deprecation warnings:
   ```bash
   mix compile --warnings-as-errors
   # Fix any usage of deprecated functions
   ```

**Deliverable:** Deprecation warnings in place

---

### Phase 4: Feature Additions (Week 4+)

**Goal:** Add missing features identified in gap analysis

**Tasks:**
1. Implement `CNS.Metrics.evidential_entanglement/2`:
   ```elixir
   def evidential_entanglement(sno_a, sno_b) do
     evidence_a = MapSet.new(sno_a.evidence)
     evidence_b = MapSet.new(sno_b.evidence)

     intersection = MapSet.intersection(evidence_a, evidence_b)
     union = MapSet.union(evidence_a, evidence_b)

     MapSet.size(intersection) / MapSet.size(union)
   end
   ```

2. Enhance chirality to optionally use Fisher-Rao:
   ```elixir
   def chirality(sno_a, sno_b, method: :fisher_rao) do
     # Compute FIM-based geodesic distance
   end
   ```

3. Add convergence metrics:
   ```elixir
   defmodule CNS.Metrics.Convergence do
     def compute(prev_sno, curr_sno) do
       confidence_delta = abs(curr_sno.confidence - prev_sno.confidence)
       coherence_delta = compute_coherence_delta(prev_sno, curr_sno)
       structural_delta = compute_structural_delta(prev_sno, curr_sno)

       # Weighted combination
       1.0 - (0.4 * confidence_delta + 0.3 * coherence_delta + 0.3 * structural_delta)
     end
   end
   ```

4. Optionally: Full TDA implementation if Gate 1 passes

**Deliverable:** Feature-complete CNS library

---

## Summary: Before vs After

### Before (Current State)

```elixir
# Scattered, unclear API
alias CNS.{Proposer, Antagonist, Synthesizer, Pipeline, Topology, Metrics}

{:ok, thesis} = Proposer.extract_claims(text)
{:ok, challenges} = Antagonist.challenge(thesis)
{:ok, synthesis} = Synthesizer.synthesize(thesis, challenges)

beta1 = Topology.Surrogates.compute_beta1_surrogate(links)
chirality = Metrics.compute_chirality(sno_a, sno_b)

# Legacy training
CNS.Training.train(snos, opts)  # Uses old contracts
```

### After (Proposed)

```elixir
# Clean, clear API
alias CNS

# High-level operations
{:ok, claims} = CNS.extract_claims(text)
{:ok, result} = CNS.run_pipeline(research_question, config)
{:ok, synthesis} = CNS.synthesize(thesis, antithesis)

# Topology analysis
%{beta1: beta1} = CNS.Topology.analyze_claim_network(claims)
fragility = CNS.Topology.fragility(claims)

# Metrics
chirality = CNS.Metrics.chirality(claim_a, claim_b)
entanglement = CNS.Metrics.evidential_entanglement(claim_a, claim_b)

# Modern training
{:ok, context} = CNS.Training.train(snos, base_model: "llama")
```

---

## Conclusion

The CNS library is **fundamentally sound** but needs **organizational cleanup** rather than major rewrites. The proposed architecture:

1. ✅ **Preserves** all working functionality
2. ✅ **Simplifies** public API for users
3. ✅ **Clarifies** module responsibilities
4. ✅ **Removes** deprecated code
5. ✅ **Defers** expensive TDA until validated
6. ✅ **Enables** seamless Crucible integration

**Next Steps:**
1. Review this proposal with stakeholders
2. Begin Phase 1 reorganization
3. Execute phases sequentially with testing at each step

---

**Document Version:** 1.0
**Last Updated:** 2025-11-24
**Status:** Pending Review

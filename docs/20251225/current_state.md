# CNS Current State Documentation

**Generated:** 2025-12-25
**Version:** 0.1.0
**Repository:** /home/home/p/g/North-Shore-AI/cns

## Overview

CNS (Chiral Narrative Synthesis) is an Elixir-based dialectical reasoning framework for automated knowledge discovery. It implements a three-agent system inspired by Hegelian dialectics: Proposer (thesis), Antagonist (antithesis), and Synthesizer (synthesis).

## Architecture Summary

```
                    +-------------+
                    |  Proposer   |
                    | (Thesis)    |
                    +------+------+
                           |
                           v
                    +-------------+
                    | Antagonist  |
                    | (Antithesis)|
                    +------+------+
                           |
                           v
                    +-------------+
                    | Synthesizer |
                    | (Synthesis) |
                    +------+------+
                           |
                    +------v------+
                    | Convergence |
                    |   Check     |
                    +-------------+
                           |
              +------------+------------+
              |                         |
         Converged?                Not Converged
              |                         |
              v                         v
        Final SNO              Feed back to Proposer
```

## Core Module Structure

### Entry Point

| File | Module | Purpose |
|------|--------|---------|
| `/home/home/p/g/North-Shore-AI/cns/lib/cns.ex` | `CNS` | Main entry point with `synthesize/3`, `run/2`, `extract_claims/2`, `run_pipeline/2`, `validate/3` |

### Core Data Types

| File | Module | Purpose | Key Functions |
|------|--------|---------|---------------|
| `/home/home/p/g/North-Shore-AI/cns/lib/cns/sno.ex` | `CNS.SNO` | Structured Narrative Object - core claim data structure | `new/2` (L78), `validate/1` (L108), `to_map/1` (L132), `from_map/1` (L173), `add_evidence/2` (L254), `update_confidence/2` (L269), `evidence_score/1` (L290), `quality_score/1` (L310), `with_topology/2` (L357), `with_chirality/2` (L376) |
| `/home/home/p/g/North-Shore-AI/cns/lib/cns/evidence.ex` | `CNS.Evidence` | Evidence records with source attribution and validity scores | `new/3`, `to_map/1`, `from_map/1`, `score/1` |
| `/home/home/p/g/North-Shore-AI/cns/lib/cns/challenge.ex` | `CNS.Challenge` | Antagonist challenges with severity levels | `new/4`, `chirality_score/1`, `critical?/1` |
| `/home/home/p/g/North-Shore-AI/cns/lib/cns/provenance.ex` | `CNS.Provenance` | Provenance chain tracking for claim derivation | `new/2`, `to_map/1`, `from_map/1` |
| `/home/home/p/g/North-Shore-AI/cns/lib/cns/config.ex` | `CNS.Config` | Configuration struct for pipeline settings | `new/1`, `quality_targets/0` |

### Agent Modules

| File | Module | Purpose | Key Functions |
|------|--------|---------|---------------|
| `/home/home/p/g/North-Shore-AI/cns/lib/cns/agents/proposer.ex` | `CNS.Agents.Proposer` | Thesis generation - extracts claims from text | `extract_claims/2` (L41), `generate_hypothesis/2` (L69), `score_confidence/1` (L106), `extract_evidence/2` (L138), `process/2` (L162) |
| `/home/home/p/g/North-Shore-AI/cns/lib/cns/agents/antagonist.ex` | `CNS.Agents.Antagonist` | Antithesis generation - challenges claims | `challenge/2` (L38), `find_contradictions/1` (L66), `find_evidence_gaps/1` (L114), `find_scope_issues/1` (L180), `find_logical_issues/1` (L231), `generate_alternatives/1` (L280), `score_chirality/1` (L317), `flag_issues/1` (L347), `process/2` (L358) |
| `/home/home/p/g/North-Shore-AI/cns/lib/cns/agents/synthesizer.ex` | `CNS.Agents.Synthesizer` | Synthesis reconciliation - merges thesis/antithesis | `synthesize/3` (L37), `ground_evidence/3` (L94), `resolve_conflicts/4` (L127), `coherence_score/1` (L149), `entailment_score/3` (L177), `process/4` (L198) |
| `/home/home/p/g/North-Shore-AI/cns/lib/cns/agents/pipeline.ex` | `CNS.Agents.Pipeline` | Pipeline orchestration | `run/2` (L38), `configure/1` (L74), `converged?/2` (L89), `iterate/2` (L107), `run_async/2` (L141), `status/1` (L149) |

### Critic Modules

| File | Module | Purpose | Key Functions |
|------|--------|---------|---------------|
| `/home/home/p/g/North-Shore-AI/cns/lib/cns/critics/critic.ex` | `CNS.Critics.Critic` | Base critic behaviour | `@callback evaluate/2`, `@callback score/2` |
| `/home/home/p/g/North-Shore-AI/cns/lib/cns/critics/grounding.ex` | `CNS.Critics.Grounding` | Evidence grounding validation | `evaluate/2`, `check_citations/2`, `evidence_coverage/1` |
| `/home/home/p/g/North-Shore-AI/cns/lib/cns/critics/logic.ex` | `CNS.Critics.Logic` | Logical consistency validation | `evaluate/2`, `check_consistency/1`, `detect_fallacies/1` |
| `/home/home/p/g/North-Shore-AI/cns/lib/cns/critics/novelty.ex` | `CNS.Critics.Novelty` | Novelty assessment | `evaluate/2`, `compare_to_corpus/2`, `novelty_score/1` |
| `/home/home/p/g/North-Shore-AI/cns/lib/cns/critics/bias.ex` | `CNS.Critics.Bias` | Bias detection | `evaluate/2`, `detect_bias_markers/1`, `neutrality_score/1` |
| `/home/home/p/g/North-Shore-AI/cns/lib/cns/critics/causal.ex` | `CNS.Critics.Causal` | Causal validity assessment | `evaluate/2`, `check_causal_claims/1`, `causal_chain_valid?/1` |

### Topology Modules

| File | Module | Purpose | Key Functions |
|------|--------|---------|---------------|
| `/home/home/p/g/North-Shore-AI/cns/lib/cns/topology.ex` | `CNS.Topology` | Topology facade for claim networks | `build_graph/1` (L21), `invariants/1` (L47), `betti_numbers/1` (L57), `detect_cycles/1` (L68), `is_dag?/1` (L83), `depth/1` (L88), `connectivity/1` (L130), `analyze_claim_network/2` (L172), `detect_circular_reasoning/1` (L190), `fragility/2` (L206), `beta1/2` (L218), `surrogates/2` (L228), `tda/2` (L243) |
| `/home/home/p/g/North-Shore-AI/cns/lib/cns/topology/surrogates.ex` | `CNS.Topology.Surrogates` | Lightweight beta1 and fragility surrogates | `compute_beta1_surrogate/1` (L69), `compute_fragility_surrogate/2` (L111), `compute_surrogates/2` (L177), `validate_correlation/2` (L218) |
| `/home/home/p/g/North-Shore-AI/cns/lib/cns/topology/persistence.ex` | `CNS.Topology.Persistence` | Persistent homology computation | `compute/2` (L144), `compute_from_embeddings/2` (L194), `compare/3` (L251), `compare_to_baseline/3` (L312), `barcodes/2` (L371), `summary/1` (L403), `has_circular_reasoning?/2` (L426), `complexity_score/2` (L453) |
| `/home/home/p/g/North-Shore-AI/cns/lib/cns/topology/adapter.ex` | `CNS.Topology.Adapter` | Bridge between CNS and ex_topology | `sno_embeddings/2` (L102), `extract_embedding/2` (L154), `claim_graph/3` (L200), `to_tensor/2` (L272), `interpret_betti/1` (L337), `interpret_fragility/1` (L401), `build_causal_graph/2` (L470), `has_cached_embedding?/1` (L513), `cache_embedding/2` (L538), `embedding_dimension/1` (L562) |
| `/home/home/p/g/North-Shore-AI/cns/lib/cns/topology/tda.ex` | `CNS.Topology.TDA` | Full topological data analysis | TDA computation helpers |
| `/home/home/p/g/North-Shore-AI/cns/lib/cns/topology/fragility.ex` | `CNS.Topology.Fragility` | CNS-specific fragility interpretation | Fragility computation and interpretation |
| `/home/home/p/g/North-Shore-AI/cns/lib/cns/logic/betti.ex` | `CNS.Logic.Betti` | Betti number computation | `compute/1`, `beta_zero/1`, `beta_one/1` |

### Metrics Modules

| File | Module | Purpose | Key Functions |
|------|--------|---------|---------------|
| `/home/home/p/g/North-Shore-AI/cns/lib/cns/metrics.ex` | `CNS.Metrics` | Quality metrics for pipeline evaluation | `quality_score/1` (L32), `entailment/2` (L60), `citation_accuracy/1` (L87), `pass_rate/2` (L110), `chirality/1` (L133), `fisher_rao_distance/2` (L156), `schema_compliance/1` (L191), `mean_entailment/1` (L221), `convergence_delta/2` (L246), `meets_targets?/1` (L260), `report/2` (L272), `evidential_entanglement/2` (L361), `convergence_score/2` (L382), `overall_quality/2` (L402) |
| `/home/home/p/g/North-Shore-AI/cns/lib/cns/metrics/chirality.ex` | `CNS.Metrics.Chirality` | Fisher-Rao distance for chirality | `build_fisher_rao_stats/2` (L53), `fisher_rao_distance/3` (L94), `compute_chirality_score/3` (L126), `compare/6` (L157) |

### Graph Utilities

| File | Module | Purpose |
|------|--------|---------|
| `/home/home/p/g/North-Shore-AI/cns/lib/cns/graph/builder.ex` | `CNS.Graph.Builder` | Build reasoning graphs from SNOs |
| `/home/home/p/g/North-Shore-AI/cns/lib/cns/graph/traversal.ex` | `CNS.Graph.Traversal` | Graph traversal algorithms |
| `/home/home/p/g/North-Shore-AI/cns/lib/cns/graph/topology.ex` | `CNS.Graph.Topology` | Topological operations |
| `/home/home/p/g/North-Shore-AI/cns/lib/cns/graph/visualization.ex` | `CNS.Graph.Visualization` | Graph visualization |

### Validation Modules

| File | Module | Purpose | Key Functions |
|------|--------|---------|---------------|
| `/home/home/p/g/North-Shore-AI/cns/lib/cns/validation/semantic.ex` | `CNS.Validation.Semantic` | 4-stage semantic validation pipeline | `extract_document_ids/1` (L85), `validate_citations/3` (L125), `compute_similarity/2` (L152), `compute_embedding_similarity/2` (L169), `validate_claim/6` (L253), `compute_entailment/3` (L331), `compute_nli_entailment/2` (L362) |
| `/home/home/p/g/North-Shore-AI/cns/lib/cns/validation/citation.ex` | `CNS.Validation.Citation` | Citation validity checking |
| `/home/home/p/g/North-Shore-AI/cns/lib/cns/validation/model_loader.ex` | `CNS.Validation.ModelLoader` | ML model loading utilities |

### Schema and Parsing

| File | Module | Purpose | Key Functions |
|------|--------|---------|---------------|
| `/home/home/p/g/North-Shore-AI/cns/lib/cns/schema/parser.ex` | `CNS.Schema.Parser` | Parse CLAIM[...] and RELATION formatted output | `parse_claims/1` (L46), `parse_relation/1` (L94), `parse_relations/1` (L118), `parse/1` (L143) |
| `/home/home/p/g/North-Shore-AI/cns/lib/cns/pipeline/schema.ex` | `CNS.Pipeline.Schema` | Pipeline data schemas | `TrainingExample`, `ClaimEntry`, `Lineage` |
| `/home/home/p/g/North-Shore-AI/cns/lib/cns/pipeline/converters.ex` | `CNS.Pipeline.Converters` | Format converters |

### Training Modules

| File | Module | Purpose | Key Functions |
|------|--------|---------|---------------|
| `/home/home/p/g/North-Shore-AI/cns/lib/cns/training/training.ex` | `CNS.Training` | Training integration (stub) | `prepare_dataset/2` (L30), `train/2` (L67), `lora_config/1` (L92), `save_checkpoint/2` (L122), `load_checkpoint/1` (L140), `evaluate/2` (L173), `triplet_to_example/3` (L204), `training_report/1` (L224) |
| `/home/home/p/g/North-Shore-AI/cns/lib/cns/training/evaluation.ex` | `CNS.Training.Evaluation` | Training evaluation harness | `compute_metrics/2` (L57), `evaluate_claims/3` (L95), `compute_f1/2` (L114), `evaluate_detailed/2` (L157) |

### Embedding Modules

| File | Module | Purpose |
|------|--------|---------|
| `/home/home/p/g/North-Shore-AI/cns/lib/cns/embedding/encoder.ex` | `CNS.Embedding.Encoder` | Embedding encoder interface |
| `/home/home/p/g/North-Shore-AI/cns/lib/cns/embedding/gemini.ex` | `CNS.Embedding.Gemini` | Gemini embedding integration |
| `/home/home/p/g/North-Shore-AI/cns/lib/cns/embedding/gemini_http.ex` | `CNS.Embedding.GeminiHTTP` | HTTP client for Gemini |

## SNO (Structured Narrative Object) Structure

```elixir
%CNS.SNO{
  id: String.t(),                    # Unique identifier (UUID)
  claim: String.t(),                 # The claim text
  evidence: [CNS.Evidence.t()],      # List of supporting evidence
  confidence: float(),               # Score from 0.0 to 1.0
  provenance: CNS.Provenance.t(),    # Derivation chain
  metadata: map(),                   # Additional metadata (topology, chirality)
  children: [CNS.SNO.t()],           # Child SNOs for hierarchical claims
  synthesis_history: [map()]         # Record of synthesis operations
}
```

## Evidence Structure

```elixir
%CNS.Evidence{
  id: String.t(),           # Unique identifier
  source: String.t(),       # Source attribution
  content: String.t(),      # Evidence content
  validity: float(),        # Validity score 0.0-1.0
  relevance: float(),       # Relevance score 0.0-1.0
  retrieval_method: atom()  # :citation | :inference | :search
}
```

## Challenge Structure

```elixir
%CNS.Challenge{
  id: String.t(),           # Unique identifier
  target_id: String.t(),    # ID of challenged SNO
  type: atom(),             # :contradiction | :evidence_gap | :scope | :logical | :alternative
  description: String.t(),  # Challenge description
  severity: atom(),         # :high | :medium | :low
  confidence: float()       # Confidence in challenge 0.0-1.0
}
```

## Topology Metrics

### Betti Numbers (Beta)

- **beta0**: Number of connected components (claim clusters)
- **beta1**: Number of independent cycles (circular reasoning patterns)
- **beta2**: Number of voids (higher-order structures)

### Chirality

Measures polarity conflict between thesis and antithesis using Fisher-Rao distance:

```
chirality_score = norm_distance * 0.6 + overlap_factor * 0.2 + conflict_penalty
```

### Fragility

Embedding variance in local neighborhoods - high fragility indicates semantic instability.

## Quality Targets (CNS 3.0)

```elixir
%{
  schema_compliance: 0.95,    # >= 95%
  citation_accuracy: 0.95,    # >= 95%
  mean_entailment: 0.50       # >= 0.50
}
```

## Dependencies

```elixir
# Core
{:nx, "~> 0.7"},
{:jason, "~> 1.4"},
{:nimble_parsec, "~> 1.4"},
{:uuid, "~> 1.1"},
{:libgraph, "~> 0.16"},
{:telemetry, "~> 1.2"},
{:ex_topology, path: "../ex_topology"},

# Optional ML
{:bumblebee, "~> 0.5", optional: true},
{:exla, "~> 0.7", optional: true},
{:gemini_ex, "~> 0.4", optional: true},
{:req, "~> 0.5", optional: true}
```

## Test Coverage

Tests are located in `/home/home/p/g/North-Shore-AI/cns/test/`:

- `cns_test.exs` - Main module tests
- `cns/sno_test.exs` - SNO struct tests
- `cns/evidence_test.exs` - Evidence struct tests
- `cns/challenge_test.exs` - Challenge struct tests
- `cns/provenance_test.exs` - Provenance struct tests
- `cns/config_test.exs` - Config struct tests
- `cns/proposer_test.exs` - Proposer agent tests
- `cns/antagonist_test.exs` - Antagonist agent tests
- `cns/synthesizer_test.exs` - Synthesizer agent tests
- `cns/pipeline_test.exs` - Pipeline orchestration tests
- `cns/metrics_test.exs` - Metrics computation tests
- `cns/topology_test.exs` - Topology analysis tests
- `cns/training_test.exs` - Training integration tests
- `cns/critics/logic_test.exs` - Logic critic tests
- `cns/schema/parser_test.exs` - Schema parser tests
- `cns/logic/betti_test.exs` - Betti number tests
- `cns/metrics/chirality_test.exs` - Chirality metrics tests
- `cns/validation/semantic_test.exs` - Semantic validation tests
- `cns/validation/citation_test.exs` - Citation validation tests
- `cns/topology/surrogates_test.exs` - Surrogate metrics tests
- `cns/pipeline/schema_test.exs` - Pipeline schema tests
- `cns/pipeline/converters_test.exs` - Converter tests
- `cns/training/evaluation_test.exs` - Evaluation harness tests

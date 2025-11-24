# CNS (Chiral Narrative Synthesis) Overview

## Version
- Date: 2025-11-21
- Status: Design Document
- Author: Architecture Team

## Executive Summary

CNS (Chiral Narrative Synthesis) is an Elixir/OTP application for synthesizing knowledge from conflicting information using dialectical reasoning, mathematical rigor via Fisher Information Metric, and topological validation through Betti numbers.

## System Architecture

```
+------------------------------------------------------------------+
|                           CNS Application                         |
+------------------------------------------------------------------+
|                                                                    |
|  +-------------------+    +-------------------+                    |
|  | Synthesis Engine  |    | Critic Pipeline   |                    |
|  | - Dialectical gen |    | - Logic Critic    |                    |
|  | - Evidence citing |    | - Grounding Critic|                    |
|  | - Constraint decoding   | - Novelty Critic |                    |
|  +-------------------+    | - Causal Critic   |                    |
|                           | - Bias Critic     |                    |
|                           +-------------------+                    |
|                                                                    |
|  +-------------------+    +-------------------+                    |
|  | SNO Manager       |    | Topology Analyzer |                    |
|  | - Graph operations|    | - Betti numbers   |                    |
|  | - Evidence links  |    | - Flaw detection  |                    |
|  | - Trust scoring   |    | - Consistency     |                    |
|  +-------------------+    +-------------------+                    |
|                                                                    |
+------------------------------------------------------------------+
                               |
                               v
+------------------------------------------------------------------+
|                    Crucible Framework API                         |
+------------------------------------------------------------------+
|  Lora | Ensemble | Bench | Telemetry | Harness | Datasets         |
+------------------------------------------------------------------+
                               |
                               v
+------------------------------------------------------------------+
|                         Tinkex SDK                                |
+------------------------------------------------------------------+
```

## Dependency Graph

```elixir
# mix.exs
defp deps do
  [
    {:crucible_framework, "~> 0.2.0"},
    {:libgraph, "~> 0.16"},
    {:nx, "~> 0.7"},
    {:scholar, "~> 0.3"},
    {:jason, "~> 1.4"}
  ]
end
```

## Core Concepts

### Structured Narrative Objects (SNOs)

SNOs are dialectical reasoning graphs that represent knowledge synthesis:

- **Thesis**: Initial claim or position
- **Antithesis**: Counter-claim or opposing position
- **Synthesis**: Reconciliation that acknowledges both perspectives
- **Evidence**: Supporting data with citations
- **Graph Structure**: Nodes = claims, edges = relationships

### Dialectical Reasoning

The synthesis process follows Hegelian dialectics:

1. **Thesis** - Initial proposition
2. **Antithesis** - Contradiction or opposing view
3. **Synthesis** - Higher-level reconciliation

### Mathematical Foundation

**Fisher Information Metric**
- Embeds hypotheses in a metric space
- Measures information content and distinguishability
- Enables geometric analysis of claim relationships

**Topological Analysis**
- Betti-0: Number of connected components (fragmentation)
- Betti-1: Number of cycles (circular reasoning)
- Persistent homology for structural stability

### Critics

Five specialized critics evaluate synthesis quality:

1. **Logic Critic** - Identifies logical flaws and contradictions
2. **Grounding Critic** - Verifies evidence citations
3. **Novelty Critic** - Assesses information gain
4. **Causal Critic** - Validates causal claims
5. **Bias Critic** - Detects systematic biases

## Application Structure

```
cns/
  lib/
    cns.ex                    # Main application entry
    cns/
      application.ex          # OTP application
      sno.ex                  # SNO struct and operations
      sno/
        graph.ex              # Graph operations
        evidence.ex           # Evidence linking
        trust.ex              # Trust scoring
      critic.ex               # Critic behaviour
      critics/
        logic.ex              # Logic critic
        grounding.ex          # Grounding critic
        novelty.ex            # Novelty critic
        causal.ex             # Causal critic
        bias.ex               # Bias critic
      synthesis/
        engine.ex             # Synthesis generation
        constraints.ex        # Decoding constraints
        evidence_linker.ex    # Evidence attribution
      topology/
        analyzer.ex           # Topological analysis
        betti.ex              # Betti number computation
        surrogate.ex          # Topology validation surrogate
      fisher/
        metric.ex             # Fisher information metric
        embedding.ex          # Hypothesis embedding
      experiment/
        harness.ex            # crucible_harness integration
        datasets.ex           # SciFact/FEVER loading
        metrics.ex            # CNS-specific metrics
  test/
    cns_test.exs
    sno_test.exs
    critics_test.exs
    synthesis_test.exs
```

## OTP Architecture

### Supervision Tree

```elixir
defmodule CNS.Application do
  use Application

  @impl true
  def start(_type, _args) do
    children = [
      # Registry for SNO processes
      {Registry, keys: :unique, name: CNS.SNORegistry},

      # Dynamic supervisor for SNO workers
      {DynamicSupervisor, name: CNS.SNOSupervisor, strategy: :one_for_one},

      # Critic supervisor (5 critics)
      CNS.CriticSupervisor,

      # Synthesis engine pool
      {CNS.Synthesis.EnginePool, pool_size: 4},

      # Topology analyzer
      CNS.Topology.Analyzer,

      # Telemetry handler
      CNS.Telemetry
    ]

    opts = [strategy: :one_for_one, name: CNS.Supervisor]
    Supervisor.start_link(children, opts)
  end
end
```

### Process Model

```
CNS.Supervisor
  |
  +-- Registry (SNORegistry)
  |
  +-- DynamicSupervisor (SNOSupervisor)
  |     |
  |     +-- SNO.Worker (dynamic)
  |     +-- SNO.Worker (dynamic)
  |     ...
  |
  +-- CriticSupervisor
  |     |
  |     +-- LogicCritic
  |     +-- GroundingCritic
  |     +-- NoveltyCritic
  |     +-- CausalCritic
  |     +-- BiasCritic
  |
  +-- Synthesis.EnginePool
  |     |
  |     +-- SynthesisWorker
  |     +-- SynthesisWorker
  |     ...
  |
  +-- Topology.Analyzer
  |
  +-- Telemetry
```

## Mapping Python Concepts to Elixir

### Python CNS Components

| Python | Elixir | Notes |
|--------|--------|-------|
| `SNO` class | `CNS.SNO` struct | Immutable struct with functions |
| `Critic` base class | `CNS.Critic` behaviour | Callbacks for evaluation |
| `LogicCritic`, etc. | `CNS.Critics.Logic`, etc. | GenServer implementations |
| `SynthesisEngine` | `CNS.Synthesis.Engine` | Pooled workers |
| `TopologyValidator` | `CNS.Topology.Analyzer` | GenServer with Nx |
| `FisherMetric` | `CNS.Fisher.Metric` | Nx-based computation |
| Dataset loading | `Crucible.Datasets` | Via crucible_datasets |
| Training loop | `CrucibleFramework` pipeline (backend_call) | Via cns_experiments adapters |

### Key Differences

1. **Concurrency Model**
   - Python: Threading/multiprocessing, GIL limitations
   - Elixir: Lightweight processes, true parallelism, supervision

2. **State Management**
   - Python: Mutable objects
   - Elixir: Immutable data, GenServer for stateful processes

3. **Failure Handling**
   - Python: Try/except, manual recovery
   - Elixir: Let it crash, supervisor restarts

4. **ML Integration**
   - Python: Direct PyTorch/TensorFlow
   - Elixir: Tinkex SDK for remote ML, Nx for local compute

## Configuration

```elixir
# config/config.exs
config :cns,
  # Critic configuration
  critics: [
    logic: %{weight: 0.25, threshold: 0.7},
    grounding: %{weight: 0.30, threshold: 0.8},
    novelty: %{weight: 0.15, threshold: 0.3},
    causal: %{weight: 0.20, threshold: 0.6},
    bias: %{weight: 0.10, threshold: 0.5}
  ],

  # Synthesis configuration
  synthesis: %{
    max_iterations: 5,
    min_critic_score: 0.6,
    evidence_threshold: 0.8
  },

  # Topology configuration
  topology: %{
    max_betti_1: 0,  # No cycles allowed
    min_connectivity: 0.9
  }

# Crucible integration
config :crucible_framework,
  lora_adapter: Crucible.Tinkex

config :crucible_framework, Crucible.Tinkex,
  api_key: {:system, "TINKEX_API_KEY"},
  base_url: "https://api.tinker.example.com"
```

## Telemetry Events

CNS emits telemetry events for all major operations:

```elixir
# SNO operations
[:cns, :sno, :create | :update | :validate]

# Critic operations
[:cns, :critic, :evaluate]
[:cns, :critic, :logic | :grounding | :novelty | :causal | :bias]

# Synthesis operations
[:cns, :synthesis, :start | :complete | :iterate]

# Topology operations
[:cns, :topology, :analyze]
[:cns, :topology, :betti]
```

## API Overview

### Main Entry Points

```elixir
# Create and synthesize
{:ok, sno} = CNS.synthesize(
  claim: "Vaccine X is safe and effective",
  evidence: evidence_list,
  options: [max_iterations: 3]
)

# Validate existing SNO
{:ok, validation} = CNS.validate(sno)

# Run experiment
{:ok, results} = CNS.run_experiment(
  dataset: :scifact,
  config: experiment_config
)
```

### SNO Operations

```elixir
# Create SNO
sno = CNS.SNO.new(
  thesis: "X causes Y",
  antithesis: "X does not cause Y",
  evidence: [...]
)

# Add evidence
sno = CNS.SNO.add_evidence(sno, evidence_item)

# Generate synthesis
{:ok, sno} = CNS.SNO.synthesize(sno, options)

# Get topology metrics
{:ok, metrics} = CNS.SNO.topology_metrics(sno)
```

### Critic Operations

```elixir
# Evaluate with all critics
{:ok, results} = CNS.Critics.evaluate_all(sno)

# Evaluate with specific critic
{:ok, result} = CNS.Critics.Logic.evaluate(sno)

# Get aggregated score
score = CNS.Critics.aggregate_score(results)
```

## Integration with Crucible Framework

### Training CNS Models

```elixir
# Build Crucible IR experiment (handled in cns_experiments)
experiment = %Crucible.IR.Experiment{
  id: "cns_scifact",
  dataset: %Crucible.IR.DatasetRef{name: "scifact", options: %{input_key: :prompt, output_key: :completion}},
  pipeline: [
    %Crucible.IR.StageDef{name: :data_load},
    %Crucible.IR.StageDef{name: :data_checks},
    %Crucible.IR.StageDef{name: :guardrails},
    %Crucible.IR.StageDef{name: :backend_call},
    %Crucible.IR.StageDef{name: :cns_surrogate_validation},
    %Crucible.IR.StageDef{name: :cns_tda_validation},
    %Crucible.IR.StageDef{name: :cns_metrics},
    %Crucible.IR.StageDef{name: :bench},
    %Crucible.IR.StageDef{name: :report}
  ],
  backend: %Crucible.IR.BackendRef{id: :tinkex, profile: :lora_finetune}
}

{:ok, ctx} = CrucibleFramework.run(experiment)
```

### Running CNS Experiments

```elixir
# Use crucible_harness for orchestration
{:ok, results} = Crucible.Harness.run(experiment, fn ctx ->
  # Load dataset
  dataset = CNS.Experiment.load_dataset(:scifact, ctx)

  # Run CNS synthesis on each claim
  syntheses = Enum.map(dataset, fn claim ->
    CNS.synthesize(claim)
  end)

  # Evaluate with critics
  evaluations = Enum.map(syntheses, &CNS.validate/1)

  # Return for analysis
  {:ok, %{syntheses: syntheses, evaluations: evaluations}}
end)

# Analyze with crucible_bench
{:ok, analysis} = Crucible.Bench.analyze(results.evaluations)
```

### Ensemble Critics

```elixir
# Use multiple models for critic evaluation
critic_pool = Crucible.Ensemble.AdapterPool.create(
  adapters: trained_critic_adapters,
  session: session
)

# Ensemble voting for critic decisions
{:ok, result} = Crucible.Ensemble.Critics.evaluate(
  critic_pool, sno, :logic
)
```

## Next Steps

1. **01_sno_implementation.md** - Detailed SNO struct and graph operations
2. **02_critic_pipeline.md** - Critic architecture and implementations
3. **03_synthesis_engine.md** - Dialectical synthesis generation
4. **04_experiment_harness.md** - Running CNS experiments

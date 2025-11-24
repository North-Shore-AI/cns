# CNS (Chiral Narrative Synthesis) - Build Prompt

**Date:** 2025-11-21
**Target:** Complete CNS implementation in Elixir using Crucible Framework
**Duration:** Multi-week intensive development
**Strategy:** Parallel multi-agent development with iterative review cycles

> Note: The legacy `CNS.CrucibleAdapter` and `lib/cns/crucible_contracts/*`
> modules have been removed. Crucible integration now lives in the
> `cns_experiments` app via `CnsExperiments.Adapters.*` behaviours.

---

## Executive Summary

Build the complete Chiral Narrative Synthesis (CNS) system in Elixir, porting and reimagining the Python experiments. CNS synthesizes coherent knowledge from conflicting information using Structured Narrative Objects (SNOs), dialectical reasoning, and a panel of neuro-symbolic critics. This implementation uses crucible_framework for all ML operations via Tinkex.

---

## ⚠️ CRITICAL: Dependency Isolation

### CNS NEVER imports Tinkex directly

The dependency chain is strictly:

```
CNS → CrucibleFramework.* → Tinkex.*
```

**CORRECT:**
```elixir
alias CrucibleFramework.Lora.Trainer
alias CrucibleFramework.Sampling
alias CrucibleFramework.Ensemble.ML

# Use Crucible's API
{:ok, result} = CrucibleFramework.Sampling.generate(client, prompt, params)
```

**WRONG - NEVER DO THIS:**
```elixir
alias Tinkex.TrainingClient      # ❌ WRONG
alias Tinkex.SamplingClient      # ❌ WRONG

# Direct Tinkex calls
Tinkex.SamplingClient.generate(...)  # ❌ WRONG
```

This isolation allows:
- CNS to be built in parallel with crucible_framework
- Testing with mocks during development
- Future backend swaps without CNS changes

---

## Parallel Development Strategy

### Building CNS Before CrucibleFramework is Complete

CNS can be developed in parallel using **contract-first development**:

1. **Define behaviours** for CrucibleFramework APIs CNS needs
2. **Use Mox mocks** in all tests
3. **Build against contracts**, not implementations
4. **Integration test** once CrucibleFramework is complete

### CrucibleFramework Contracts for CNS

Define these behaviours in `lib/cns/crucible_contracts/`:

```elixir
# lib/cns/crucible_contracts/sampling.ex
defmodule CrucibleFramework.Sampling do
  @moduledoc """
  Behaviour contract for CrucibleFramework sampling API.
  CNS codes against this; mocked in tests.
  """

  @type sampling_params :: %{
    temperature: float(),
    max_tokens: pos_integer(),
    top_p: float(),
    stop_sequences: [String.t()]
  }

  @callback generate(client :: pid(), prompt :: String.t(), params :: sampling_params()) ::
    {:ok, String.t()} | {:error, term()}

  @callback generate_batch(client :: pid(), prompts :: [String.t()], params :: sampling_params()) ::
    {:ok, [String.t()]} | {:error, term()}
end

# lib/cns/crucible_contracts/lora.ex
defmodule CrucibleFramework.Lora do
  @moduledoc """
  Behaviour contract for CrucibleFramework LoRA training API.
  """

  @type config :: %{
    base_model: String.t(),
    lora_rank: pos_integer(),
    learning_rate: float(),
    batch_size: pos_integer()
  }

  @callback create_experiment(opts :: keyword()) ::
    {:ok, experiment :: map()} | {:error, term()}

  @callback start_session(experiment :: map()) ::
    {:ok, session :: pid()} | {:error, term()}
end

# lib/cns/crucible_contracts/ensemble.ex
defmodule CrucibleFramework.Ensemble.ML do
  @moduledoc """
  Behaviour contract for CrucibleFramework ensemble API.
  """

  @type infer_opts :: [
    strategy: :majority | :weighted_majority | :best_confidence,
    execution: :parallel | :hedged,
    timeout: pos_integer()
  ]

  @callback infer(pool :: pid(), prompt :: String.t(), opts :: infer_opts()) ::
    {:ok, String.t()} | {:error, term()}
end

# lib/cns/crucible_contracts/datasets.ex
defmodule CrucibleFramework.Datasets do
  @moduledoc """
  Behaviour contract for CrucibleFramework dataset loading.
  """

  @type dataset_opts :: [
    split: :train | :dev | :test,
    limit: pos_integer() | nil
  ]

  @callback load(name :: atom(), opts :: dataset_opts()) ::
    {:ok, [map()]} | {:error, term()}

  @callback stream(name :: atom(), opts :: dataset_opts()) ::
    Enumerable.t()
end
```

### Mox Setup for Testing

```elixir
# test/support/mocks.ex
Mox.defmock(CrucibleFramework.SamplingMock, for: CrucibleFramework.Sampling)
Mox.defmock(CrucibleFramework.LoraMock, for: CrucibleFramework.Lora)
Mox.defmock(CrucibleFramework.Ensemble.MLMock, for: CrucibleFramework.Ensemble.ML)
Mox.defmock(CrucibleFramework.DatasetsMock, for: CrucibleFramework.Datasets)

# test/test_helper.exs
Mox.defmock(CrucibleFramework.SamplingMock, for: CrucibleFramework.Sampling)
Mox.defmock(CrucibleFramework.LoraMock, for: CrucibleFramework.Lora)
Mox.defmock(CrucibleFramework.Ensemble.MLMock, for: CrucibleFramework.Ensemble.ML)
Mox.defmock(CrucibleFramework.DatasetsMock, for: CrucibleFramework.Datasets)

ExUnit.start()
```

### Application Config for Mocks

```elixir
# config/test.exs
import Config

config :cns,
  sampling_module: CrucibleFramework.SamplingMock,
  lora_module: CrucibleFramework.LoraMock,
  ensemble_module: CrucibleFramework.Ensemble.MLMock,
  datasets_module: CrucibleFramework.DatasetsMock

# config/prod.exs
import Config

config :cns,
  sampling_module: CrucibleFramework.Sampling,
  lora_module: CrucibleFramework.Lora,
  ensemble_module: CrucibleFramework.Ensemble.ML,
  datasets_module: CrucibleFramework.Datasets
```

### Using Contracts in CNS Code

```elixir
# lib/cns/synthesis/engine.ex
defmodule CNS.Synthesis.Engine do
  # Get configured module (real or mock)
  defp sampling_module do
    Application.get_env(:cns, :sampling_module, CrucibleFramework.Sampling)
  end

  defp generate_with_constraints(prompt, constraints, params, opts) do
    client = Keyword.fetch!(opts, :sampling_client)

    # Use the contract - works with real or mock
    case sampling_module().generate(client, prompt, params) do
      {:ok, response} ->
        text = extract_text(response)
        validated = Constraints.enforce(text, constraints)
        {:ok, validated}

      {:error, _} = error ->
        error
    end
  end
end
```

### Example Test with Mocks

```elixir
# test/cns/synthesis/engine_test.exs
defmodule CNS.Synthesis.EngineTest do
  use ExUnit.Case, async: true
  import Mox

  setup :verify_on_exit!

  describe "synthesize/3" do
    test "generates synthesis using sampling API" do
      sno = create_test_sno()

      # Setup mock expectations
      expect(CrucibleFramework.SamplingMock, :generate, fn _client, prompt, _params ->
        assert String.contains?(prompt, "THESIS")
        assert String.contains?(prompt, "ANTITHESIS")

        {:ok, "Both perspectives have merit [E1]. The thesis correctly identifies X, while the antithesis rightly notes Y [E2]."}
      end)

      # Run synthesis
      {:ok, engine} = CNS.Synthesis.Engine.start_link([])
      {:ok, result} = CNS.Synthesis.Engine.synthesize(engine, sno, [
        sampling_client: :mock_client,
        strategy: :single_shot
      ])

      assert result.synthesis != nil
      assert String.contains?(result.synthesis, "[E1]")
    end

    test "retries on failure" do
      sno = create_test_sno()

      # First call fails, second succeeds
      expect(CrucibleFramework.SamplingMock, :generate, fn _, _, _ ->
        {:error, :timeout}
      end)
      expect(CrucibleFramework.SamplingMock, :generate, fn _, _, _ ->
        {:ok, "Valid synthesis [E1]."}
      end)

      {:ok, engine} = CNS.Synthesis.Engine.start_link([])
      {:ok, result} = CNS.Synthesis.Engine.synthesize(engine, sno, [
        sampling_client: :mock_client,
        max_retries: 2
      ])

      assert result.synthesis != nil
    end
  end
end
```

### Integration Testing Phase

Once CrucibleFramework is complete:

```elixir
# test/cns/integration/crucible_integration_test.exs
defmodule CNS.Integration.CrucibleIntegrationTest do
  use ExUnit.Case

  @moduletag :integration
  @moduletag :requires_crucible

  setup do
    # Use real CrucibleFramework modules
    Application.put_env(:cns, :sampling_module, CrucibleFramework.Sampling)
    Application.put_env(:cns, :lora_module, CrucibleFramework.Lora)

    # Start real session
    {:ok, experiment} = CrucibleFramework.Lora.create_experiment(
      name: "cns-integration-test",
      config: %{base_model: "llama-3-8b", lora_rank: 16}
    )
    {:ok, session} = CrucibleFramework.Lora.start_session(experiment)

    on_exit(fn ->
      # Reset to mocks
      Application.put_env(:cns, :sampling_module, CrucibleFramework.SamplingMock)
    end)

    %{session: session}
  end

  test "full CNS pipeline with real CrucibleFramework", %{session: session} do
    sno = create_real_sno()

    {:ok, sampling_client} = CrucibleFramework.Lora.Trainer.create_sampler(session)

    {:ok, result} = CNS.synthesize(sno, [
      sampling_client: sampling_client,
      strategy: :iterative,
      max_iterations: 3
    ])

    assert result.synthesis != nil
    assert result.critic_scores.grounding > 0.7
  end
end
```

---

## Required Reading

### CNS Architecture Documents (READ ALL BEFORE STARTING)

```
S:\cns\docs\20251121\architecture\
├── 00_cns_overview.md           # OTP architecture, Python→Elixir mapping
├── 01_sno_implementation.md     # SNO struct, graph, trust scoring
├── 02_critic_pipeline.md        # 5 critics as GenServers
├── 03_synthesis_engine.md       # Constrained decoding, evidence linking
└── 04_experiment_harness.md     # Dataset loading, metrics, reports
```

### Crucible Framework Integration Docs

```
S:\crucible_framework\docs\20251121\tinkex_integration\
├── 00_architecture_overview.md  # How CNS uses Crucible
├── 01_tinkex_adapter.md         # Session management
├── 02_lora_training_interface.md # LoRA training API
└── 03_ensemble_ml_integration.md # Ensemble strategies
```

### Python CNS Experiments (STUDY FOR INSPIRATION)

```
S:\tinkerer\
├── thinker.sh                   # CLI for experiments
├── brainstorm\
│   ├── 20251121_ROADMAP.md              # Surrogate-first validation
│   ├── 20251121_EMERGENT_TOPOLOGICAL_TRUTH.md
│   ├── 20251121_LCM_TOPOLOGY_DISCOVERY.md
│   └── 20251121_SYNTHETIC_PROOF_BOOTSTRAPPING.md
├── cns3\
│   ├── cns3_gemini_deepResearch.md  # Full CNS 3.0 proposal
│   └── cns3_gpt5.md                 # CNS 3.0 specification
```

### Tinkex SDK (UNDERSTAND THE API)

```
S:\tinkex\
├── lib\tinkex\
│   ├── training_client.ex
│   ├── sampling_client.ex
│   └── types\
└── README.md
```

### Existing Crucible Libraries

```
S:\crucible_framework\lib\       # Main framework
S:\crucible_ensemble\lib\        # Voting strategies
S:\crucible_bench\lib\           # Statistical testing
S:\crucible_datasets\lib\        # Dataset loading
S:\crucible_telemetry\lib\       # Instrumentation
S:\crucible_harness\lib\         # Experiment orchestration
```

---

## Development Strategy

### Phase 1: Core Data Structures (Agents 1-3 in Parallel)

**Agent 1: SNO Implementation**
```elixir
# Build these modules:
lib/cns/
├── sno.ex                  # Core SNO struct and functions
├── sno/
│   ├── claim.ex            # Individual claims
│   ├── evidence.ex         # Evidence with provenance
│   ├── relation.ex         # Typed relations
│   └── validator.ex        # SNO validation
```

**Agent 2: Graph Operations**
```elixir
# Build these modules:
lib/cns/graph/
├── builder.ex              # Construct reasoning graphs
├── topology.ex             # Betti numbers, cycle detection
├── traversal.ex            # Path finding, reachability
└── visualization.ex        # Export to DOT/Mermaid
```

**Agent 3: Evidence & Trust**
```elixir
# Build these modules:
lib/cns/evidence/
├── linker.ex               # Link claims to evidence
├── trust_scorer.ex         # Bayesian trust propagation
├── source_registry.ex      # Track source reliability
└── temporal.ex             # Time decay for evidence
```

### Phase 2: Critics (Agents 4-8 in Parallel)

**Agent 4: Logic Critic**
```elixir
# Build these modules:
lib/cns/critics/
├── logic.ex                # GenServer for logical consistency
├── logic/
│   ├── cycle_detector.ex   # Find circular reasoning
│   ├── contradiction.ex    # Detect contradictions
│   └── entailment.ex       # Check logical entailment
```

**Agent 5: Grounding Critic**
```elixir
# Build these modules:
lib/cns/critics/
├── grounding.ex            # GenServer for factual accuracy
├── grounding/
│   ├── nli_client.ex       # NLI via Crucible/Tinkex
│   ├── citation_checker.ex # Verify citations
│   └── evidence_scorer.ex  # Score evidence quality
```

**Agent 6: Novelty Critic**
```elixir
# Build these modules:
lib/cns/critics/
├── novelty.ex              # GenServer for originality
├── novelty/
│   ├── embedding_store.ex  # Historical embeddings
│   ├── similarity.ex       # Cosine similarity
│   └── parsimony.ex        # Complexity penalty
```

**Agent 7: Causal Critic**
```elixir
# Build these modules:
lib/cns/critics/
├── causal.ex               # GenServer for causal validity
├── causal/
│   ├── claim_classifier.ex # Correlational vs causal
│   ├── scm_analyzer.ex     # Structural causal models
│   └── intervention.ex     # Do-calculus checks
```

**Agent 8: Bias Critic**
```elixir
# Build these modules:
lib/cns/critics/
├── bias.ex                 # GenServer for bias detection
├── bias/
│   ├── disparity.ex        # Group disparity metrics
│   ├── fairness.ex         # Fairness constraints
│   └── power_shadows.ex    # Systemic blind spots
```

### Phase 3: Synthesis Engine (Agents 9-11 in Parallel)

**Agent 9: Core Synthesis**
```elixir
# Build these modules:
lib/cns/synthesis/
├── engine.ex               # Main synthesis GenServer
├── strategies/
│   ├── single_shot.ex      # One-pass synthesis
│   ├── iterative.ex        # Refine until critics pass
│   └── ensemble.ex         # Multi-model synthesis
```

**Agent 10: Constrained Generation**
```elixir
# Build these modules:
lib/cns/synthesis/
├── constraints/
│   ├── evidence_citation.ex    # Must cite evidence
│   ├── conflict_resolution.ex  # Must address conflicts
│   └── no_hallucination.ex     # No unsupported claims
├── prompt_builder.ex           # Build dialectical prompts
└── post_processor.ex           # Verify constraints
```

**Agent 11: Evidence Linker**
```elixir
# Build these modules:
lib/cns/synthesis/
├── evidence_linker.ex      # Insert citations
├── span_aligner.ex         # Align to evidence spans
└── verification.ex         # Verify links are valid
```

### Phase 4: Experiment Infrastructure (Agents 12-14 in Parallel)

**Agent 12: Dataset Integration**
```elixir
# Build these modules:
lib/cns/datasets/
├── loader.ex               # Load via crucible_datasets
├── scifact.ex              # SciFact specific
├── fever.ex                # FEVER specific
├── synth_dial.ex           # Synthetic dialectics
└── preprocessor.ex         # Convert to SNOs
```

**Agent 13: Metrics & Evaluation**
```elixir
# Build these modules:
lib/cns/evaluation/
├── metrics.ex              # All CNS metrics
├── metrics/
│   ├── coherence.ex        # Logic consistency score
│   ├── grounding.ex        # Citation precision/recall
│   ├── novelty.ex          # NovAScore implementation
│   ├── topology.ex         # Betti number metrics
│   └── human_eval.ex       # Human evaluation interface
```

**Agent 14: Experiment Runner**
```elixir
# Build these modules:
lib/cns/experiment/
├── runner.ex               # Main experiment orchestrator
├── config.ex               # Experiment configuration
├── reporter.ex             # Generate reports
└── checkpointer.ex         # Save/restore state
```

### Phase 5: Application & Supervision (Agent 15)

**Agent 15: OTP Application**
```elixir
# Build these modules:
lib/cns/
├── application.ex          # OTP application
├── supervisor.ex           # Main supervisor
├── critic_supervisor.ex    # Supervise all critics
├── engine_pool.ex          # Pool of synthesis workers
└── telemetry.ex            # CNS telemetry events
```

---

## Test-Driven Development Instructions

### TDD Workflow for Each Agent

1. **Write tests FIRST** before any implementation
2. Run tests to see them fail (Red)
3. Implement minimum code to pass (Green)
4. Refactor while keeping tests green (Refactor)
5. Repeat for each function/module

### Test Structure

```elixir
# test/cns/sno_test.exs
defmodule CNS.SNOTest do
  use ExUnit.Case, async: true

  alias CNS.SNO
  alias CNS.SNO.{Claim, Evidence, Relation}

  describe "new/1" do
    test "creates SNO with valid thesis and antithesis" do
      thesis = %Claim{id: "c1", text: "Drug X is effective"}
      antithesis = %Claim{id: "c2", text: "Drug X is not effective"}

      {:ok, sno} = SNO.new(thesis: thesis, antithesis: antithesis)

      assert sno.thesis == thesis
      assert sno.antithesis == antithesis
      assert sno.synthesis == nil
      assert is_binary(sno.id)
    end

    test "returns error without thesis" do
      antithesis = %Claim{id: "c2", text: "Drug X is not effective"}

      assert {:error, :missing_thesis} = SNO.new(antithesis: antithesis)
    end
  end

  describe "add_evidence/3" do
    test "links evidence to claim" do
      {:ok, sno} = create_basic_sno()
      evidence = %Evidence{
        id: "e1",
        text: "Study shows 80% efficacy",
        source: "PubMed:12345"
      }

      {:ok, updated} = SNO.add_evidence(sno, "c1", evidence)

      assert evidence in updated.evidence
      assert {"c1", "e1"} in updated.evidence_links
    end
  end

  describe "compute_chirality/2" do
    test "returns high score for opposing claims" do
      {:ok, sno1} = create_sno_with_claim("Drug works")
      {:ok, sno2} = create_sno_with_claim("Drug doesn't work")

      score = SNO.compute_chirality(sno1, sno2)

      assert score > 0.7
    end

    test "returns low score for similar claims" do
      {:ok, sno1} = create_sno_with_claim("Drug is effective")
      {:ok, sno2} = create_sno_with_claim("Drug shows efficacy")

      score = SNO.compute_chirality(sno1, sno2)

      assert score < 0.3
    end
  end
end
```

### Critic Testing Pattern

```elixir
# test/cns/critics/logic_test.exs
defmodule CNS.Critics.LogicTest do
  use ExUnit.Case, async: true

  alias CNS.Critics.Logic
  alias CNS.SNO

  setup do
    {:ok, logic} = Logic.start_link([])
    %{logic: logic}
  end

  describe "evaluate/2" do
    test "returns high score for consistent SNO", %{logic: logic} do
      sno = create_consistent_sno()

      {:ok, result} = Logic.evaluate(logic, sno)

      assert result.score > 0.8
      assert result.cycles == 0
      assert result.contradictions == []
    end

    test "returns low score for circular reasoning", %{logic: logic} do
      sno = create_circular_sno()

      {:ok, result} = Logic.evaluate(logic, sno)

      assert result.score < 0.3
      assert result.cycles > 0
      assert "circular_reasoning" in result.issues
    end

    test "detects contradictions", %{logic: logic} do
      sno = create_contradictory_sno()

      {:ok, result} = Logic.evaluate(logic, sno)

      assert length(result.contradictions) > 0
    end
  end
end
```

### Integration Testing

```elixir
# test/cns/integration/synthesis_test.exs
defmodule CNS.Integration.SynthesisTest do
  use ExUnit.Case

  # Integration tests are not async

  @tag :integration
  test "full synthesis pipeline" do
    # Create opposing SNOs
    {:ok, thesis_sno} = create_thesis_sno()
    {:ok, antithesis_sno} = create_antithesis_sno()

    # Start synthesis engine
    {:ok, engine} = CNS.Synthesis.Engine.start_link([])

    # Run synthesis
    {:ok, result} = CNS.Synthesis.Engine.synthesize(
      engine,
      thesis_sno,
      antithesis_sno,
      strategy: :iterative,
      max_iterations: 3
    )

    # Verify result
    assert result.synthesis != nil
    assert result.critic_scores.logic > 0.7
    assert result.critic_scores.grounding > 0.8
    assert result.evidence_coverage > 0.9
  end
end
```

### Quality Gates (ALL MUST PASS)

```bash
# Run from S:\cns
wsl -d ubuntu-dev bash -c "cd /home/home/p/g/North-Shore-AI/cns && mix test"
wsl -d ubuntu-dev bash -c "cd /home/home/p/g/North-Shore-AI/cns && mix dialyzer"
wsl -d ubuntu-dev bash -c "cd /home/home/p/g/North-Shore-AI/cns && mix compile --warnings-as-errors"
wsl -d ubuntu-dev bash -c "cd /home/home/p/g/North-Shore-AI/cns && mix format --check-formatted"
wsl -d ubuntu-dev bash -c "cd /home/home/p/g/North-Shore-AI/cns && mix credo --strict"
```

---

## Agent Coordination Protocol

### Spawn Pattern

```
┌─────────────────────────────────────────────────────────────┐
│  ORCHESTRATOR AGENT                                         │
│  - Spawns parallel work agents per phase                    │
│  - Monitors progress                                        │
│  - Triggers review cycles between phases                    │
└─────────────────────────────┬───────────────────────────────┘
                              │
        Phase 1: Core Data    │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   ┌─────────┐           ┌─────────┐           ┌─────────┐
   │ Agent 1 │           │ Agent 2 │           │ Agent 3 │
   │   SNO   │           │  Graph  │           │Evidence │
   └────┬────┘           └────┬────┘           └────┬────┘
        └─────────────────────┼─────────────────────┘
                              ▼
                      ┌───────────────┐
                      │ REVIEW AGENT  │
                      └───────┬───────┘
                              │
        Phase 2: Critics      │
        ┌──────┬──────┬───────┼───────┬──────┐
        ▼      ▼      ▼       ▼       ▼      ▼
      [A4]   [A5]   [A6]    [A7]    [A8]   ...
      Logic Ground Novel   Causal  Bias
        │      │      │       │       │
        └──────┴──────┴───────┼───────┴──────┘
                              ▼
                      ┌───────────────┐
                      │ REVIEW AGENT  │
                      └───────┬───────┘
                              │
        [Continue for Phases 3, 4, 5...]
```

### Review Agent Instructions

After each parallel phase completes:

1. **Collect all work** from parallel agents
2. **Run full test suite**: `mix test --trace`
3. **Run dialyzer**: `mix dialyzer`
4. **Check warnings**: `mix compile --warnings-as-errors`
5. **Analyze failures** and categorize:
   - Type errors → specific fixes
   - Logic errors → may need agent re-work
   - Integration errors → cross-module issues
6. **Spawn fix agents** for each category
7. **Repeat** until all green

### Inter-Module Dependencies

```
Phase 1 (Core) → Phase 2 (Critics) → Phase 3 (Synthesis) → Phase 4 (Experiment) → Phase 5 (App)

Dependencies:
- Critics depend on SNO, Graph, Evidence
- Synthesis depends on Critics, SNO
- Experiment depends on Synthesis, Datasets, Metrics
- App ties everything together
```

---

## Implementation Details

### Key Types to Define

```elixir
# lib/cns/sno.ex
defmodule CNS.SNO do
  @type t :: %__MODULE__{
    id: String.t(),
    thesis: Claim.t(),
    antithesis: Claim.t(),
    synthesis: Claim.t() | nil,
    evidence: [Evidence.t()],
    evidence_links: [{claim_id :: String.t(), evidence_id :: String.t()}],
    graph: Graph.t(),
    trust: float(),
    metadata: map(),
    created_at: DateTime.t(),
    updated_at: DateTime.t()
  }

  defstruct [
    :id,
    :thesis,
    :antithesis,
    :synthesis,
    evidence: [],
    evidence_links: [],
    graph: nil,
    trust: 0.0,
    metadata: %{},
    created_at: nil,
    updated_at: nil
  ]
end

# lib/cns/sno/claim.ex
defmodule CNS.SNO.Claim do
  @type t :: %__MODULE__{
    id: String.t(),
    text: String.t(),
    embedding: Nx.Tensor.t() | nil,
    confidence: float(),
    source_ids: [String.t()]
  }
end

# lib/cns/sno/evidence.ex
defmodule CNS.SNO.Evidence do
  @type t :: %__MODULE__{
    id: String.t(),
    text: String.t(),
    source: String.t(),
    timestamp: DateTime.t(),
    quality_score: float(),
    embedding: Nx.Tensor.t() | nil
  }
end
```

### Critic Behaviour

```elixir
# lib/cns/critics/critic.ex
defmodule CNS.Critics.Critic do
  @callback evaluate(sno :: CNS.SNO.t()) ::
    {:ok, %{
      score: float(),
      issues: [String.t()],
      details: map()
    }} | {:error, term()}

  @callback name() :: atom()

  @callback weight() :: float()
end
```

### Synthesis Engine API

```elixir
# lib/cns/synthesis/engine.ex
defmodule CNS.Synthesis.Engine do
  use GenServer

  @type synthesis_result :: %{
    synthesis: CNS.SNO.Claim.t(),
    critic_scores: %{
      logic: float(),
      grounding: float(),
      novelty: float(),
      causal: float(),
      bias: float()
    },
    evidence_coverage: float(),
    iterations: pos_integer(),
    telemetry: map()
  }

  @spec synthesize(pid(), CNS.SNO.t(), CNS.SNO.t(), keyword()) ::
    {:ok, synthesis_result()} | {:error, term()}
  def synthesize(engine, thesis_sno, antithesis_sno, opts \\ []) do
    GenServer.call(engine, {:synthesize, thesis_sno, antithesis_sno, opts}, :infinity)
  end
end
```

### Using Crucible Framework (Contract-Based)

```elixir
# Example: Using Crucible for critic inference via contracts
defmodule CNS.Critics.Grounding do
  use GenServer

  # Get configured module (real or mock)
  defp sampling_module do
    Application.get_env(:cns, :sampling_module, CrucibleFramework.Sampling)
  end

  def evaluate(sno, opts \\ []) do
    # Use Crucible's sampling client for NLI
    claims = extract_claims(sno)
    evidence = sno.evidence
    client = Keyword.fetch!(opts, :sampling_client)

    results = Enum.map(claims, fn claim ->
      relevant_evidence = find_relevant_evidence(claim, evidence)

      Enum.map(relevant_evidence, fn ev ->
        # Call NLI model via Crucible contract (works with real or mock)
        {:ok, nli_result} = sampling_module().generate(
          client,
          build_nli_prompt(claim, ev),
          %{max_tokens: 10, temperature: 0.0}
        )

        parse_nli_result(nli_result)
      end)
    end)

    compute_grounding_score(results)
  end
end
```

### Using Datasets via Contract

```elixir
# Example: Loading datasets via contract
defmodule CNS.Datasets.Loader do
  defp datasets_module do
    Application.get_env(:cns, :datasets_module, CrucibleFramework.Datasets)
  end

  def load_scifact(opts \\ []) do
    # Works with real CrucibleFramework.Datasets or mock
    {:ok, data} = datasets_module().load(:scifact, opts)

    # Convert to SNOs
    Enum.map(data, &convert_to_sno/1)
  end
end
```

---

## Dependencies to Add

Update `mix.exs`:

```elixir
defp deps do
  [
    # Crucible Framework (provides Tinkex integration)
    {:crucible_framework, path: "../crucible_framework"},

    # Graph operations
    {:libgraph, "~> 0.16"},

    # Numerical computing
    {:nx, "~> 0.7"},

    # JSON
    {:jason, "~> 1.4"},

    # Telemetry
    {:telemetry, "~> 1.2"},

    # Testing
    {:mox, "~> 1.0", only: :test},
    {:stream_data, "~> 0.6", only: [:test]},

    # Development
    {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false},
    {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
    {:ex_doc, "~> 0.30", only: :dev, runtime: false}
  ]
end
```

---

## Success Criteria

### Minimum Viable CNS

- [ ] Can create and manipulate SNOs
- [ ] Can build reasoning graphs with cycle detection
- [ ] All 5 critics evaluate SNOs and return scores
- [ ] Synthesis engine generates new claims from thesis/antithesis
- [ ] Evidence citation enforcement works
- [ ] Can load SciFact/FEVER datasets
- [ ] Can run experiments with metrics
- [ ] All operations emit telemetry

### Quality Metrics

- [ ] 100% of public functions have typespecs
- [ ] 100% of modules have documentation
- [ ] >90% test coverage
- [ ] Zero dialyzer warnings
- [ ] Zero compilation warnings
- [ ] All tests pass

### End-to-End Verification

```elixir
# This test must pass:
test "complete CNS experiment on SciFact subset" do
  # Load dataset
  {:ok, dataset} = CNS.Datasets.load(:scifact, split: :dev, limit: 100)

  # Configure experiment
  config = %CNS.Experiment.Config{
    name: "scifact_baseline",
    dataset: dataset,
    synthesis_strategy: :iterative,
    max_iterations: 3,
    critic_weights: %{
      logic: 0.3,
      grounding: 0.4,
      novelty: 0.15,
      causal: 0.1,
      bias: 0.05
    }
  }

  # Run experiment
  {:ok, results} = CNS.Experiment.Runner.run(config)

  # Verify results
  assert results.num_synthesized > 0
  assert results.avg_critic_score > 0.6
  assert results.evidence_coverage > 0.8

  # Check metrics
  assert results.metrics.coherence > 0.7
  assert results.metrics.grounding_precision > 0.8
  assert results.metrics.novelty_score > 0.3

  # Verify report generated
  assert File.exists?(results.report_path)
end
```

---

## Iteration Protocol

### Week 1: Foundation (Phases 1-2)
- Phase 1 agents build SNO, Graph, Evidence
- Review and fix
- Phase 2 agents build all 5 critics
- Review and fix

### Week 2: Synthesis & Experiments (Phases 3-4)
- Phase 3 agents build synthesis engine
- Review and fix
- Phase 4 agents build experiment infrastructure
- Review and fix

### Week 3: Application & Integration (Phase 5)
- Phase 5 agent builds OTP application
- Full integration testing
- Cross-module issue resolution
- Performance optimization

### Week 4+: Hardening & Experiments
- Run actual experiments
- Tune critic weights
- Add more datasets
- Documentation and examples
- Real-world testing with Tinkex API

---

## Path Formats Reminder

- **File tools** (Read/Write/Edit/Grep/Glob): `\\wsl.localhost\ubuntu-dev\home\home\p\g\North-Shore-AI\cns\...`
- **Bash commands**: `wsl -d ubuntu-dev bash -c "cd /home/home/p/g/North-Shore-AI/cns && <command>"`

---

## Start Command

To begin implementation, an orchestrator agent should:

1. Read all required documents listed above
2. Spawn Phase 1 agents (1-3) in parallel
3. Wait for completion
4. Run review cycle
5. Spawn Phase 2 agents (4-8) in parallel
6. Run review cycle
7. Continue through all phases
8. Final integration testing

**The goal is maximum parallelization with quality gates at each phase, building a production-ready CNS system that can run real experiments.**

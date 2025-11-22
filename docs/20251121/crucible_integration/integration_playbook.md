# Crucible Experiment Integration Playbook

**Date:** 2025-11-21  
**Audience:** Experiment authors, infra maintainers, and UI clients consuming Crucible outputs.

---

## 1. Purpose and Scope

- Provide a detailed, end-to-end guide for authoring, running, and consuming experiments inside `crucible_framework`.
- Clarify boundaries between the core library (`cns`), experimentation harness (this repo), UI surfaces, and adapter SDKs (Tinkex).
- Standardize how runs are defined, executed, instrumented, and persisted so results remain publication-quality and reproducible.

## 2. Component Responsibilities

| Component | Responsibility | Notes |
| --- | --- | --- |
| `crucible_harness` | Experiment DSL + orchestration (`Crucible.Harness.MLExperiment`, runners) | Defines runs/stages, manages lifecycle. |
| `crucible_tinkex` | Adapter to Tinkex SDK | Single owner of Tinkex credentials and session management. |
| `crucible_datasets` | Benchmark loading/caching | Provides dataset manifests to runners. |
| `crucible_bench` | Statistical testing | 15+ tests; used in eval/analysis stages. |
| `crucible_ensemble` | Multi-model voting | For reliability experiments and A/B/C comparisons. |
| `crucible_hedging` | Latency/reliability hedging | Wrap inference calls to improve P99/P995. |
| `crucible_telemetry` | Instrumentation + export | Emits `:telemetry` events and structures them for storage/streaming. |
| Artifacts (`artifacts/`) | Outputs, checkpoints, manifests | One run directory per `run_id`. |

## 3. End-to-End Flow (Happy Path)

1. Define experiment with `Crucible.Harness.MLExperiment` (name, parameters, quality targets, stages).
2. Generate runs (hyperparameter grid) and hand each run to a runner (Tinkex-backed or custom).
3. Runner executes stages (`:train`, `:eval`, `:analysis`), emitting telemetry and writing artifacts.
4. Results are aggregated, bench tests applied, reports/exporters run.
5. UI/consumers read from artifacts + telemetry streams (no direct SDK access).

## 4. Authoring Experiments

Use the MLExperiment DSL to keep definitions declarative and sweep-friendly:

```elixir
alias Crucible.Harness.MLExperiment

{:ok, experiment} =
  MLExperiment.new(
    name: "scifact_claim_extractor",
    description: "LoRA fine-tune with citation penalties",
    parameters: %{learning_rate: [1.0e-4, 2.0e-4], lora_rank: [16, 32]},
    quality_targets: %{schema_compliance: 0.95, citation_accuracy: 0.95},
    output_dir: "artifacts/crucible/scifact_claim_extractor"
  )

experiment =
  experiment
  |> MLExperiment.add_stage(%{name: :train, type: :train, config: %{epochs: 5, batch_size: 8}})
  |> MLExperiment.add_stage(%{name: :eval, type: :eval, config: %{dataset: :scifact_dev}})

runs = MLExperiment.generate_runs(experiment)
```

Validation: `MLExperiment.validate/1` catches missing names, invalid stage configs, or out-of-range quality targets before scheduling.

## 5. Running with Tinkex

`Crucible.Harness.TinkexRunner` provides the execution shell for LoRA training/eval using the Tinkex adapter:

```elixir
alias Crucible.Harness.{MLExperiment, TinkexRunner}

{:ok, exp} = MLExperiment.new(name: "demo", stages: [%{name: :train, type: :train, config: %{epochs: 3}}])
{:ok, runner} = TinkexRunner.init(exp, output_dir: "artifacts/crucible/demo")
{:ok, runner} = TinkexRunner.run_stage(runner, :train, %{type: :train, config: %{epochs: 3, batch_size: 8}})
results = TinkexRunner.get_results(runner)
:ok = TinkexRunner.cleanup(runner)
```

Key integration points:
- Credentials live in Tinkex adapter config (`config :crucible_framework, Crucible.Tinkex, ...`); they are not accepted from callers.
- `run_stage/3` emits start/stop/error telemetry for each stage; consumers subscribe via `:telemetry`.
- Checkpoints and run outputs land under the configured `output_dir`; include a manifest per run for reproducibility.

## 6. Data and Datasets

- Prefer `crucible_datasets` loaders for benchmark splits to keep hashing/caching consistent.
- Store dataset manifests alongside run artifacts to reproduce training/eval inputs.
- When using custom datasets, register a loader module that returns schema + split metadata; avoid embedding raw paths in experiment code.

## 7. Reliability Features (Optional but Recommended)

- **Ensembles:** Wrap multiple adapters and strategies via `Crucible.Ensemble.create/1` and feed outputs into evaluation stages for majority/weighted votes.
- **Hedging:** Use `crucible_hedging` strategies for inference to cut tail latency; configure per stage when latency matters (e.g., eval with strict SLAs).
- **Causal Trace:** Enable causal tracing on runs that need provenance; persist traces next to checkpoints for later debugging.

## 8. Telemetry and Reporting

- All stages emit `:telemetry` events under the `[:crucible, :tinkex, ...]` namespace (start/stop/error and metrics payloads).
- Mirror telemetry to JSONL under `artifacts/crucible/<run_id>/telemetry.jsonl` to keep UI and batch analysis in sync.
- After eval/analysis stages, generate reports via the reporting modules (Markdown/HTML/LaTeX) and store them under the same run directory.

## 9. Configuration Checklist

- `config/runtime.exs`: set `:crucible_framework, :lora_adapter` (defaults to `Crucible.Tinkex`) and adapter credentials.
- Concurrency: use BEAM-level controls (pool sizes, rate limits) in the adapter config; keep experiment code free of hard-coded sleeps.
- Seeds: set `seed` in `MLExperiment.new/1` and propagate to stages for reproducibility.
- Paths: always set `output_dir`; avoid relying on `tmp/` for artifacts you need to keep.

## 10. Migration and Experiment Placement

- Keep reusable primitives in `cns`; author experiments here in `crucible_framework` (under `examples/` or `artifacts/` per run).
- UI clients (`cns_ui`, `crucible_ui`) consume artifacts + telemetry streams; they should not embed Tinkex credentials or direct SDK calls.
- When porting an experiment:
  1) Move the run definition into an `MLExperiment` module.  
  2) Replace direct SDK usage with `TinkexRunner`.  
  3) Point outputs to `artifacts/crucible/<experiment>/`.  
  4) Wire evaluation to `crucible_bench` metrics/tests.  
  5) Enable telemetry streaming for UI parity.

## 11. Acceptance Criteria for Integrations

- Experiments can be run headless (CLI/script) and yield deterministic artifacts with manifests.
- Telemetry for every stage is available via `:telemetry` and mirrored to disk.
- No UI or downstream client needs direct access to Tinkex; all interactions flow through runners and artifacts.
- Quality targets and statistical tests are captured in reports alongside raw outputs and checkpoints.

## 12. Next Steps

- Add example scripts under `examples/` that wrap `MLExperiment` + `TinkexRunner` for common datasets.
- Ship OpenAPI/JSON schema describing run manifests and telemetry payloads for UI teams.
- Provide per-strategy ensemble/hedging examples tied to real eval datasets to standardize reliability benchmarks.

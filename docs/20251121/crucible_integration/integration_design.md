# Crucible Integration Design: Examples, Schemas, Reliability Runs

**Date:** 2025-11-21  
**Purpose:** Define the deliverables and patterns to operationalize the integration playbook across runnable examples, schemas/OpenAPI, and standardized reliability benchmarks.

---

## Scope and Objectives

1) Ship runnable examples pairing `Crucible.Harness.MLExperiment` with `Crucible.Harness.TinkexRunner` for common datasets.  
2) Publish schemas/OpenAPI for run manifests and telemetry to unblock UI consumers.  
3) Provide ensemble/hedging example runs to standardize reliability benchmarks.

---

## 1) Runnable Examples (MLExperiment + TinkexRunner)

### Goals
- Quick-start scripts that can be executed headless (`mix run`) and produce artifacts under `artifacts/crucible/<experiment>/`.
- Cover at least two datasets: a classification/QA task and a citation-heavy task (e.g., SciFact).
- Demonstrate stage composition: `:train`, `:eval`, `:analysis`.

### Deliverables
- `examples/scifact_lora.exs`
  - Defines `MLExperiment` with params (`learning_rate`, `lora_rank`), stages (`train`, `eval`), and quality targets.
  - Uses `TinkexRunner` to execute and writes artifacts/manifests.
- `examples/qa_ensemble.exs`
  - Runs an eval-only flow against multiple adapters for QA/short-answer tasks.
- README snippet: how to run (`mix run examples/<file>.exs`) and where outputs land.

### Patterns/Contracts
- Require `output_dir` to be set; fail fast if missing.
- Emit telemetry to `artifacts/.../telemetry.jsonl` in parallel with `:telemetry` events.
- Include dataset manifest (hash + split) in run manifest.

---

## 2) Schemas & OpenAPI (Run Manifests + Telemetry)

### Goals
- Stable schema that UI clients can consume without depending on internal structs.
- OpenAPI/JSON Schema documents committed to `docs/schemas/` and linked from 20251121 docs.

### Deliverables
- `docs/schemas/run_manifest.schema.json`
  - Fields: `run_id`, `experiment_id`, `params`, `stages`, `dataset_manifest`, `output_dir`, `checkpoints[]`, `created_at`, `seed`, `quality_targets`.
- `docs/schemas/telemetry_event.schema.json`
  - Envelope: `timestamp`, `event_name`, `run_id`, `stage`, `payload`.
  - Payload variants: `stage_start`, `stage_stop`, `stage_error`, `metric` (with `name`, `value`, `step/epoch`), `checkpoint`.
- `docs/schemas/openapi.yaml`
  - Paths: `/v1/jobs` (submit), `/v1/jobs/{id}` (status), `/v1/jobs/{id}/stream` (SSE/WS telemetry).
  - Components reference the JSON Schemas above.

### Patterns/Contracts
- Version fields in schemas (`schema_version`).
- Backward compatibility policy: additive changes only without a major bump.
- Stream tokens for `/stream` documented (claims: `run_id`, `exp_id`, `exp`).

---

## 3) Ensemble/Hedging Reliability Runs

### Goals
- Provide ready-to-run reliability benchmarks with ensemble voting and hedging to measure P50/P95/P99 latency and agreement.
- Show how to wrap adapters and apply hedging strategies in eval stages.

### Deliverables
- `examples/reliability_ensemble.exs`
  - Builds an ensemble with at least two adapters and a majority/weighted strategy.
  - Runs eval with telemetry of agreement and disagreement rates.
- `examples/reliability_hedging.exs`
  - Demonstrates hedging strategy (e.g., percentile_75) over model calls; records latency distribution and success thresholds.
- Report template
  - Adds reliability metrics to reports (agreement %, tie rate, latency percentiles).

### Patterns/Contracts
- Ensemble stage emits per-adapter outputs and vote traces to artifacts.
- Hedging stage captures latency samples and hedging decisions; stored under `artifacts/.../hedging_metrics.json`.
- Quality gates: require schema compliance and citation accuracy checks where applicable.

---

## Implementation Plan (Sequenced)

1. Scaffold examples (`examples/*.exs`) with manifests/telemetry writing and README instructions.  
2. Define JSON Schemas and OpenAPI; link them from this doc and the integration playbook.  
3. Add reliability examples (ensemble + hedging) and ensure metrics land in artifacts and telemetry.  
4. Wire report generation to include reliability metrics for the reliability examples.  
5. Validate by running all example scripts locally; ensure artifacts and JSONL telemetry are produced and match schemas.  
6. Publish schema paths in `docs/20251121/crucible_integration/integration_playbook.md`.

---

## Acceptance Criteria

- All example scripts run via `mix run examples/<file>.exs` and write artifacts + telemetry JSONL.
- Schemas/OpenAPI exist under `docs/schemas/` and describe run manifests and telemetry events used by examples.
- Reliability examples emit ensemble/hedging metrics and include report outputs with latency/agreement statistics.
- UI consumers can parse artifacts/telemetry using the published schemas without referencing internal modules.

# CNS Gaps Analysis

**Generated:** 2025-12-25
**Repository:** /home/home/p/g/North-Shore-AI/cns

## Summary

This document identifies gaps, incomplete features, and areas requiring implementation or improvement in the CNS codebase.

---

## Critical Gaps

### 1. LLM Integration Missing

**Location:** All agent modules
**Status:** STUB IMPLEMENTATIONS

The Proposer, Antagonist, and Synthesizer agents use heuristic-based implementations rather than actual LLM calls:

- `/home/home/p/g/North-Shore-AI/cns/lib/cns/agents/proposer.ex` - Uses regex and word matching for claim extraction instead of LLM
- `/home/home/p/g/North-Shore-AI/cns/lib/cns/agents/antagonist.ex` - Uses pattern matching for challenge generation instead of LLM
- `/home/home/p/g/North-Shore-AI/cns/lib/cns/agents/synthesizer.ex` - Uses template-based synthesis instead of LLM

**Impact:** Core dialectical reasoning depends on simple heuristics rather than actual LLM-powered reasoning.

**Required:**
- Integration with LLM provider (OpenAI, Anthropic, or Gemini)
- Prompt templates for each agent role
- Response parsing and validation
- Error handling for LLM failures

### 2. Training Module is a Stub

**Location:** `/home/home/p/g/North-Shore-AI/cns/lib/cns/training/training.ex`
**Status:** STUB - Lines 8, 66-87

```elixir
# Line 8
NOTE: This is a stub implementation. The full Crucible IR dependencies are not yet available.

# Line 66-87 - train/2 returns mock data
def train(_dataset, _opts \\ []) do
  Logger.debug("CNS.Training.train/2 stub implementation - Crucible IR not available")
  {:ok, %{status: :completed, metrics: %{...}, outputs: %{...}}}
end
```

**Impact:** Cannot actually train LoRA adapters for CNS agents.

**Required:**
- Crucible IR integration
- Tinkex LoRA training integration
- Real checkpoint save/load functionality

### 3. Validate Function Not Implemented

**Location:** `/home/home/p/g/North-Shore-AI/cns/lib/cns.ex` Lines 115-119
**Status:** TODO

```elixir
def validate(sno, _corpus, _opts \\ []) do
  # TODO: Implement proper validation wrapper
  {:ok, %{valid: true, sno: sno}}
end
```

**Impact:** High-level validation API returns fake success.

---

## Moderate Gaps

### 4. Critic Modules Under-Implemented

**Location:** `/home/home/p/g/North-Shore-AI/cns/lib/cns/critics/`
**Status:** PARTIAL

Critics exist but lack deep implementation:

- `grounding.ex` - Basic citation checking only
- `logic.ex` - Pattern matching for fallacies, not semantic logic checking
- `novelty.ex` - Simple word overlap, not embedding-based novelty
- `bias.ex` - Keyword detection only
- `causal.ex` - Regex-based causal claim detection

**Required:**
- Integration with semantic models for deeper analysis
- Proper scoring calibration
- Cross-critic coordination

### 5. Embedding Encoder Incomplete

**Location:** `/home/home/p/g/North-Shore-AI/cns/lib/cns/embedding/encoder.ex`
**Status:** PARTIAL

The default encoder is not fully wired up. Code relies on:
- Optional Bumblebee models
- Optional Gemini embeddings
- Fallback to word overlap when models unavailable

**Required:**
- Reliable default embedding strategy
- Model caching and lazy loading
- Consistent embedding dimensions

### 6. NLI Model Integration Fragile

**Location:** `/home/home/p/g/North-Shore-AI/cns/lib/cns/validation/semantic.ex` Lines 362-378
**Status:** OPTIONAL DEPENDENCY

```elixir
def compute_nli_entailment(premise, hypothesis) do
  case ModelLoader.get_nli_model() do
    {:ok, serving} -> ...
    {:error, reason} -> {:error, reason}  # Falls back to word overlap
  end
end
```

**Impact:** Semantic validation degrades to word overlap without models.

---

## Minor Gaps

### 7. Missing Telemetry Events

**Location:** Multiple modules
**Status:** NOT IMPLEMENTED

The README mentions telemetry but no `:telemetry.execute/3` calls are present in agent modules.

**Required:**
- Add telemetry events for pipeline stages
- Metrics collection for training runs
- Performance tracking

### 8. No Mermaid Diagram Export

**Location:** Graph modules
**Status:** NOT IMPLEMENTED

The README mentions Mermaid export but no implementation exists in:
- `/home/home/p/g/North-Shore-AI/cns/lib/cns/graph/visualization.ex`

**Required:**
- Mermaid diagram generation from SNO graphs
- Export functions for claim networks

### 9. Pipeline Async Not Tested

**Location:** `/home/home/p/g/North-Shore-AI/cns/lib/cns/agents/pipeline.ex` Line 141-144
**Status:** IMPLEMENTED BUT UNTESTED

```elixir
def run_async(input, %Config{} = config \\ %Config{}) do
  Task.async(fn -> run(input, config) end)
end
```

No corresponding test in `pipeline_test.exs`.

### 10. Configuration Not Fully Wired

**Location:** `/home/home/p/g/North-Shore-AI/cns/lib/cns/config.ex`
**Status:** PARTIAL

Config struct exists but many options are not used by agents:
- `model` field in agent configs not used (no LLM integration)
- `temperature` not applicable without LLM
- Some thresholds not validated

---

## Documentation Gaps

### 11. Missing Guide Files

**Location:** `/home/home/p/g/North-Shore-AI/cns/docs/guides/`
**Status:** REFERENCED BUT MAY NOT EXIST

mix.exs references these guides but they may not exist:
- `getting_started.md`
- `claim_parsing.md`
- `topology_analysis.md`
- `validation_pipeline.md`
- `data_pipeline.md`
- `api_reference.md`

### 12. Incomplete Doctests

**Location:** Multiple modules
**Status:** PARTIAL

Some doctests are marked with `iex>` but:
- Some examples are incomplete
- Some examples don't match actual function behavior
- Missing doctests for complex functions

---

## Test Gaps

### 13. Integration Tests Missing

**Status:** NOT IMPLEMENTED

No integration tests for:
- Full pipeline end-to-end
- Multi-iteration convergence
- Real embedding + topology analysis

### 14. Property-Based Tests Limited

**Status:** PARTIAL

`stream_data` is a dependency but property-based tests are not extensive:
- SNO generation properties
- Evidence validation properties
- Convergence properties

### 15. Critic Tests Incomplete

**Location:** `/home/home/p/g/North-Shore-AI/cns/test/cns/critics/`
**Status:** PARTIAL

Only `logic_test.exs` exists. Missing tests for:
- `grounding_test.exs`
- `novelty_test.exs`
- `bias_test.exs`
- `causal_test.exs`

---

## Architecture Gaps

### 16. No Streaming Support

**Status:** NOT IMPLEMENTED

Pipeline runs synchronously. No support for:
- Streaming intermediate results
- Progress callbacks
- Partial result handling

### 17. No Persistence Layer

**Status:** NOT IMPLEMENTED

No built-in way to:
- Persist SNO graphs to database
- Cache intermediate results
- Resume interrupted pipelines

### 18. No Rate Limiting

**Status:** NOT IMPLEMENTED

When LLM integration is added, will need:
- Rate limiting for API calls
- Retry logic with backoff
- Queue management for batch processing

---

## Priority Recommendations

### High Priority (Blocking Core Functionality)
1. LLM Integration for agents
2. Training module implementation
3. Validate function implementation

### Medium Priority (Quality Improvement)
4. Critic module improvements
5. Embedding encoder reliability
6. Telemetry integration
7. Integration tests

### Low Priority (Nice to Have)
8. Mermaid export
9. Streaming support
10. Documentation guides
11. Additional property tests

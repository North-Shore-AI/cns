# Migration Guide: v0.2 to v0.3

## Summary

CNS v0.3 reorganizes modules into cleaner namespaces and provides a simplified high-level API.

## Module Renames

| Old | New |
|-----|-----|
| `CNS.Proposer` | `CNS.Agents.Proposer` (or use `CNS.extract_claims/2`) |
| `CNS.Antagonist` | `CNS.Agents.Antagonist` |
| `CNS.Synthesizer` | `CNS.Agents.Synthesizer` (or use `CNS.synthesize/3`) |
| `CNS.Pipeline` | `CNS.Agents.Pipeline` (or use `CNS.run_pipeline/2`) |
| `CNS.TrainingV2` | `CNS.Training` |

## API Changes

### Extracting Claims

```elixir
# Old
{:ok, claims} = CNS.Proposer.extract_claims(text)

# New (recommended)
{:ok, claims} = CNS.extract_claims(text)

# New (direct agent access)
{:ok, claims} = CNS.Agents.Proposer.extract_claims(text)
```

### Running Pipeline

```elixir
# Old
{:ok, result} = CNS.Pipeline.run(input, config)

# New (recommended)
{:ok, result} = CNS.run_pipeline(input, config)

# New (direct agent access)
{:ok, result} = CNS.Agents.Pipeline.run(input, config)
```

### Topology

```elixir
# Old
beta1 = CNS.Topology.Surrogates.compute_beta1_surrogate(links)

# New
analysis = CNS.Topology.analyze_claim_network(claims)
beta1 = analysis.beta1
```

### Metrics

```elixir
# New functions added
entanglement = CNS.Metrics.evidential_entanglement(claim_a, claim_b)
convergence = CNS.Metrics.convergence_score(prev, curr)
```

## Deprecations

The following modules are deprecated and will be removed in v0.4:

- `CNS.Training` (old) - Use `CNS.Training` (new, in training/ namespace)
- Direct agent imports - Use high-level CNS API instead

## Backward Compatibility

Temporary aliases are provided for v0.3 to ease migration. Update your code before v0.4 release.

## Examples

### Before

```elixir
alias CNS.{Proposer, Antagonist, Synthesizer, Pipeline}

{:ok, claims} = Proposer.extract_claims(text)
{:ok, challenges} = Antagonist.challenge(claims)
{:ok, synthesis} = Synthesizer.synthesize(thesis, antithesis)
```

### After

```elixir
# Use high-level API
{:ok, claims} = CNS.extract_claims(text)
{:ok, synthesis} = CNS.synthesize(thesis, antithesis)
{:ok, result} = CNS.run_pipeline(input, config)

# Topology analysis
analysis = CNS.Topology.analyze_claim_network(claims)
fragility = CNS.Topology.fragility(claims)

# New metrics
entanglement = CNS.Metrics.evidential_entanglement(a, b)
```
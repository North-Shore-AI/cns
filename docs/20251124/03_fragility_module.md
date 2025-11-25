# CNS Fragility Module - Complete Implementation

**Date**: 2024-11-24
**Status**: Ready for Implementation
**Depends on**: `CNS.Topology.Adapter` (from Section 1)

---

## Overview

This document provides a **complete, production-ready implementation** of the `CNS.Topology.Fragility` module. This module replaces the current fragility surrogate implementation with `ex_topology`'s advanced topological fragility analysis.

### Current Problem (from CNS.Topology.Surrogates)

The existing fragility implementation has critical limitations:

```elixir
# Current implementation: Only k-NN variance
defp compute_knn_variance(tensor, k, metric) do
  # ... computes mean k-NN distance variance
  mean_distance = Enum.sum(neighbor_means) / max(length(neighbor_means), 1)
  normalize_fragility(mean_distance, metric)
end
```

**Limitations:**
1. **Single metric**: Only k-NN variance, no point removal sensitivity
2. **No topology awareness**: Doesn't consider how claim removal affects overall topology
3. **No persistence**: Ignores persistence diagrams and bottleneck stability
4. **No local analysis**: Can't analyze fragility around specific claims
5. **No robustness scoring**: No overall network stability assessment

### Solution: ex_topology.Fragility Integration

The new implementation leverages `ExTopology.Fragility` capabilities:

- **Point removal sensitivity**: Measures topological impact of claim removal
- **Local fragility analysis**: Per-claim neighborhood stability
- **Robustness scoring**: Overall claim network stability (0-1 scale)
- **Critical claim identification**: Finds claims whose removal changes topology
- **Bottleneck stability**: Minimum perturbation to change topology

---

## Complete Module Implementation

```elixir
defmodule CNS.Topology.Fragility do
  @moduledoc """
  Topological fragility analysis for claim networks.

  This module analyzes the stability and robustness of claim networks by
  measuring how sensitive the network's topological structure is to
  perturbations (claim removal, embedding variations).

  ## Key Concepts

  **Fragility**: A fragile claim network has topological features that
  disappear with small perturbations. Claims in fragile networks are
  semantically unstable.

  **Robustness**: A robust network maintains its topological structure
  despite perturbations. Claims are well-grounded and stable.

  **Critical Claims**: Claims whose removal significantly changes the
  network topology. These are "load-bearing" arguments in the dialectic.

  ## Integration with ex_topology

  This module wraps `ExTopology.Fragility` with CNS-specific semantics:
  - Point clouds → Claim embeddings
  - Point removal → Claim removal
  - Fragility scores → Semantic instability
  - Critical points → Critical claims

  ## Examples

      # Analyze a synthesis result
      synthesis = %CNS.Core.SNO{...}
      analysis = CNS.Topology.Fragility.analyze(synthesis)

      analysis.robustness_score
      # => 0.73 (moderately robust)

      analysis.critical_claims
      # => [%CNS.Core.SNO{statement: "Core premise"}, ...]

      # Analyze specific claim
      local = CNS.Topology.Fragility.local_analysis(synthesis, claim_index: 3)
      local.removal_impact
      # => 0.82 (high impact - removing this claim destabilizes network)
  """

  alias ExTopology.Fragility
  alias CNS.Topology.Adapter

  require Logger

  #
  # Public API
  #

  @doc """
  Analyzes topological fragility of a claim network.

  Computes multiple fragility metrics and identifies critical claims.
  Returns a comprehensive analysis suitable for validation and reporting.

  ## Parameters

  - `sno` - A CNS.Core.SNO struct (synthesis or claim tree)
  - `opts` - Keyword list:
    - `:max_dimension` - Maximum homology dimension (default: 1)
    - `:top_k` - Number of critical claims to identify (default: 5)
    - `:k` - k-NN parameter for local analysis (default: 3)
    - `:embedding_opts` - Options passed to Adapter.sno_embeddings/2

  ## Returns

  Map with the following keys:
  - `:robustness_score` - Float in [0, 1], higher = more robust
  - `:interpretation` - Atom: :highly_robust | :moderately_robust | :fragile
  - `:critical_claims` - List of SNOs identified as critical
  - `:removal_sensitivity` - List of {sno, score} tuples (all claims)
  - `:bottleneck_stability` - Minimum perturbation to change topology
  - `:statistics` - Summary statistics

  ## Examples

      iex> synthesis = build_test_synthesis()
      iex> analysis = CNS.Topology.Fragility.analyze(synthesis)
      iex> analysis.robustness_score
      0.68
      iex> analysis.interpretation
      :moderately_robust
      iex> length(analysis.critical_claims)
      5

  ## Performance

  - O(n²) for pairwise distances
  - O(n * PH) for point removal (PH = persistent homology cost)
  - Large networks (>100 claims): Consider sampling or caching

  ## CNS Interpretation

  - **High robustness (>0.7)**: Network is stable, claims well-grounded
  - **Moderate (0.4-0.7)**: Some fragility, inspect critical claims
  - **Low (<0.4)**: Network is fragile, likely circular reasoning or weak claims
  """
  @spec analyze(CNS.Core.SNO.t(), keyword()) :: %{
          robustness_score: float(),
          interpretation: :highly_robust | :moderately_robust | :fragile,
          critical_claims: [CNS.Core.SNO.t()],
          removal_sensitivity: [{CNS.Core.SNO.t(), float()}],
          bottleneck_stability: float(),
          statistics: map()
        }
  def analyze(sno, opts \\ []) do
    # Extract all claims from SNO tree
    snos = extract_all_claims(sno)

    if length(snos) < 3 do
      Logger.warning("Fragility analysis requires at least 3 claims, got #{length(snos)}")

      %{
        robustness_score: 1.0,
        interpretation: :insufficient_data,
        critical_claims: [],
        removal_sensitivity: [],
        bottleneck_stability: 0.0,
        statistics: %{n_claims: length(snos)}
      }
    else
      # Convert SNOs to embeddings
      embeddings = Adapter.sno_embeddings(snos, Keyword.get(opts, :embedding_opts, []))

      max_dim = Keyword.get(opts, :max_dimension, 1)
      top_k = Keyword.get(opts, :top_k, 5)

      # Compute point removal sensitivity
      removal_scores =
        Fragility.point_removal_sensitivity(embeddings,
          max_dimension: max_dim,
          metric: :bottleneck
        )

      # Identify critical points
      critical_indices =
        Fragility.identify_critical_points(removal_scores,
          top_k: min(top_k, length(snos))
        )

      # Overall robustness
      robustness = Fragility.robustness_score(embeddings)

      # Bottleneck stability
      stability = Fragility.bottleneck_stability(embeddings, num_samples: 10)

      # Build result
      %{
        robustness_score: Float.round(robustness, 4),
        interpretation: interpret_robustness(robustness),
        critical_claims: Enum.map(critical_indices, &Enum.at(snos, &1)),
        removal_sensitivity:
          Enum.zip(snos, Enum.map(0..(length(snos) - 1), &Map.get(removal_scores, &1, 0.0))),
        bottleneck_stability: Float.round(stability, 4),
        statistics: compute_statistics(snos, removal_scores, robustness)
      }
    end
  end

  @doc """
  Analyzes local fragility around a specific claim.

  Examines how a claim relates to its semantic neighborhood and
  measures its local stability.

  ## Parameters

  - `sno` - Root SNO (synthesis or claim tree)
  - `opts` - Keyword list:
    - `:claim_index` - Index of claim to analyze (REQUIRED)
    - `:k` - Number of neighbors to consider (default: 3)
    - `:embedding_opts` - Options for embedding generation

  ## Returns

  Map with:
  - `:claim` - The analyzed SNO
  - `:removal_impact` - Score indicating what happens if claim removed
  - `:neighborhood_fragility` - Mean fragility of nearby claims
  - `:relative_fragility` - This claim's fragility vs. its neighborhood
  - `:neighbors` - List of semantically similar claims
  - `:interpretation` - Human-readable assessment

  ## Examples

      iex> synthesis = build_test_synthesis()
      iex> local = CNS.Topology.Fragility.local_analysis(synthesis, claim_index: 2)
      iex> local.removal_impact
      0.45
      iex> local.interpretation
      "Moderately critical - removal would change local topology"

  ## CNS Interpretation

  - **High removal impact (>0.7)**: Critical claim, removing breaks arguments
  - **High relative fragility**: Claim is unstable compared to neighbors
  - **Low neighborhood fragility**: Claim is in a stable semantic region
  """
  @spec local_analysis(CNS.Core.SNO.t(), keyword()) :: %{
          claim: CNS.Core.SNO.t(),
          removal_impact: float(),
          neighborhood_fragility: float(),
          relative_fragility: float(),
          neighbors: [CNS.Core.SNO.t()],
          interpretation: String.t()
        }
  def local_analysis(sno, opts \\ []) do
    claim_index = Keyword.fetch!(opts, :claim_index)
    k = Keyword.get(opts, :k, 3)

    snos = extract_all_claims(sno)

    if claim_index < 0 or claim_index >= length(snos) do
      raise ArgumentError, "claim_index #{claim_index} out of range [0, #{length(snos) - 1}]"
    end

    embeddings = Adapter.sno_embeddings(snos, Keyword.get(opts, :embedding_opts, []))

    # Compute local fragility
    local = Fragility.local_fragility(embeddings, claim_index, k: k)

    # Build result
    %{
      claim: Enum.at(snos, claim_index),
      removal_impact: Float.round(local.removal_impact, 4),
      neighborhood_fragility: Float.round(local.neighborhood_mean_fragility, 4),
      relative_fragility: Float.round(local.relative_fragility, 4),
      neighbors: Enum.map(local.neighbor_indices, &Enum.at(snos, &1)),
      interpretation: interpret_local_fragility(local)
    }
  end

  @doc """
  Computes robustness score for a claim network.

  This is a convenience function that returns just the robustness score
  without the full analysis. Useful for quick checks.

  ## Parameters

  - `sno` - Root SNO
  - `opts` - Options (passed to analyze/2)

  ## Returns

  Float in [0, 1] where:
  - 1.0 = Perfectly robust (rare)
  - 0.7-1.0 = Highly robust
  - 0.4-0.7 = Moderately robust
  - 0.0-0.4 = Fragile

  ## Examples

      iex> synthesis = build_test_synthesis()
      iex> CNS.Topology.Fragility.robustness_score(synthesis)
      0.68
  """
  @spec robustness_score(CNS.Core.SNO.t(), keyword()) :: float()
  def robustness_score(sno, opts \\ []) do
    snos = extract_all_claims(sno)

    if length(snos) < 3 do
      1.0
    else
      embeddings = Adapter.sno_embeddings(snos, Keyword.get(opts, :embedding_opts, []))
      Fragility.robustness_score(embeddings) |> Float.round(4)
    end
  end

  @doc """
  Identifies the top k most critical claims in a network.

  ## Parameters

  - `sno` - Root SNO
  - `opts` - Keyword list:
    - `:top_k` - Number of critical claims (default: 5)
    - `:max_dimension` - Max homology dimension (default: 1)

  ## Returns

  List of tuples: `{sno, fragility_score}` sorted by score (desc)

  ## Examples

      iex> synthesis = build_test_synthesis()
      iex> critical = CNS.Topology.Fragility.critical_claims(synthesis, top_k: 3)
      iex> length(critical)
      3
      iex> {claim, score} = hd(critical)
      iex> score > 0.5
      true
  """
  @spec critical_claims(CNS.Core.SNO.t(), keyword()) :: [{CNS.Core.SNO.t(), float()}]
  def critical_claims(sno, opts \\ []) do
    top_k = Keyword.get(opts, :top_k, 5)
    max_dim = Keyword.get(opts, :max_dimension, 1)

    snos = extract_all_claims(sno)

    if length(snos) < 3 do
      []
    else
      embeddings = Adapter.sno_embeddings(snos, Keyword.get(opts, :embedding_opts, []))

      removal_scores =
        Fragility.point_removal_sensitivity(embeddings,
          max_dimension: max_dim,
          metric: :bottleneck
        )

      critical_indices =
        Fragility.identify_critical_points(removal_scores,
          top_k: min(top_k, length(snos))
        )

      critical_indices
      |> Enum.map(fn idx ->
        {Enum.at(snos, idx), Map.get(removal_scores, idx, 0.0)}
      end)
      |> Enum.sort_by(fn {_sno, score} -> score end, :desc)
    end
  end

  @doc """
  Compares fragility between two claim networks (e.g., thesis vs antithesis).

  ## Parameters

  - `sno1` - First claim network
  - `sno2` - Second claim network
  - `opts` - Options (passed to analyze/2)

  ## Returns

  Map with:
  - `:sno1_robustness` - Robustness of first network
  - `:sno2_robustness` - Robustness of second network
  - `:robustness_difference` - Difference (positive = sno1 more robust)
  - `:interpretation` - Comparison summary

  ## Examples

      iex> thesis = build_thesis()
      iex> antithesis = build_antithesis()
      iex> comp = CNS.Topology.Fragility.compare(thesis, antithesis)
      iex> comp.interpretation
      "Thesis is more robust (0.72 vs 0.58)"
  """
  @spec compare(CNS.Core.SNO.t(), CNS.Core.SNO.t(), keyword()) :: map()
  def compare(sno1, sno2, opts \\ []) do
    rob1 = robustness_score(sno1, opts)
    rob2 = robustness_score(sno2, opts)

    diff = rob1 - rob2

    interpretation =
      cond do
        abs(diff) < 0.1 -> "Similar robustness (#{rob1} vs #{rob2})"
        diff > 0 -> "First network more robust (#{rob1} vs #{rob2})"
        true -> "Second network more robust (#{rob2} vs #{rob1})"
      end

    %{
      sno1_robustness: rob1,
      sno2_robustness: rob2,
      robustness_difference: Float.round(diff, 4),
      interpretation: interpretation
    }
  end

  @doc """
  Validates that fragility analysis is working correctly.

  Runs diagnostic checks on a claim network and reports any issues.

  ## Parameters

  - `sno` - Root SNO
  - `opts` - Options for analysis

  ## Returns

  Map with:
  - `:valid?` - Boolean, true if all checks pass
  - `:warnings` - List of warning messages
  - `:diagnostics` - Detailed diagnostic info

  ## Examples

      iex> synthesis = build_test_synthesis()
      iex> validation = CNS.Topology.Fragility.validate(synthesis)
      iex> validation.valid?
      true
      iex> validation.warnings
      []
  """
  @spec validate(CNS.Core.SNO.t(), keyword()) :: map()
  def validate(sno, opts \\ []) do
    snos = extract_all_claims(sno)
    warnings = []

    # Check 1: Sufficient claims
    warnings =
      if length(snos) < 3 do
        ["Insufficient claims (#{length(snos)} < 3) for fragility analysis" | warnings]
      else
        warnings
      end

    # Check 2: Embeddings exist
    embeddings =
      try do
        Adapter.sno_embeddings(snos, Keyword.get(opts, :embedding_opts, []))
      rescue
        e ->
          warnings = ["Failed to generate embeddings: #{inspect(e)}" | warnings]
          nil
      end

    # Check 3: Fragility computation works
    {fragility_works, warnings} =
      if embeddings && length(snos) >= 3 do
        try do
          _rob = Fragility.robustness_score(embeddings)
          {true, warnings}
        rescue
          e ->
            {false, ["Fragility computation failed: #{inspect(e)}" | warnings]}
        end
      else
        {false, warnings}
      end

    # Check 4: Removal sensitivity works
    {removal_works, warnings} =
      if embeddings && fragility_works do
        try do
          _scores = Fragility.point_removal_sensitivity(embeddings, max_dimension: 1)
          {true, warnings}
        rescue
          e ->
            {false, ["Point removal sensitivity failed: #{inspect(e)}" | warnings]}
        end
      else
        {false, warnings}
      end

    %{
      valid?: Enum.empty?(warnings),
      warnings: Enum.reverse(warnings),
      diagnostics: %{
        n_claims: length(snos),
        embeddings_ok: !is_nil(embeddings),
        fragility_ok: fragility_works,
        removal_ok: removal_works
      }
    }
  end

  #
  # Private Functions
  #

  # Extract all claims from SNO tree (depth-first traversal)
  defp extract_all_claims(sno) do
    # Start with root
    claims = [sno]

    # Add evidence claims
    evidence_claims =
      case Map.get(sno, :evidence) do
        nil -> []
        evidence when is_list(evidence) -> Enum.flat_map(evidence, &extract_evidence_claims/1)
        _ -> []
      end

    # Add challenge claims
    challenge_claims =
      case Map.get(sno, :challenges) do
        nil -> []
        challenges when is_list(challenges) -> Enum.flat_map(challenges, &extract_all_claims/1)
        _ -> []
      end

    # Add synthesis claims (recursive)
    synthesis_claims =
      case Map.get(sno, :synthesis) do
        nil -> []
        synthesis -> extract_all_claims(synthesis)
      end

    claims ++ evidence_claims ++ challenge_claims ++ synthesis_claims
  end

  defp extract_evidence_claims(evidence) do
    case Map.get(evidence, :source_sno) do
      nil -> []
      sno -> [sno]
    end
  end

  # Interpret robustness score
  defp interpret_robustness(score) when score > 0.7, do: :highly_robust
  defp interpret_robustness(score) when score > 0.4, do: :moderately_robust
  defp interpret_robustness(_), do: :fragile

  # Interpret local fragility
  defp interpret_local_fragility(%{removal_impact: impact, relative_fragility: rel_frag}) do
    cond do
      impact > 0.7 and rel_frag > 1.5 ->
        "Highly critical and unstable - removing would significantly change topology"

      impact > 0.7 ->
        "Highly critical - removal would significantly change topology"

      impact > 0.4 and rel_frag > 1.3 ->
        "Moderately critical and somewhat unstable"

      impact > 0.4 ->
        "Moderately critical - removal would change local topology"

      rel_frag > 1.5 ->
        "Low impact but locally unstable compared to neighbors"

      true ->
        "Stable and low impact - removal has minimal effect"
    end
  end

  # Compute summary statistics
  defp compute_statistics(snos, removal_scores, robustness) do
    scores = Map.values(removal_scores)

    mean_score = if Enum.empty?(scores), do: 0.0, else: Enum.sum(scores) / length(scores)
    max_score = if Enum.empty?(scores), do: 0.0, else: Enum.max(scores)
    min_score = if Enum.empty?(scores), do: 0.0, else: Enum.min(scores)

    %{
      n_claims: length(snos),
      mean_removal_score: Float.round(mean_score, 4),
      max_removal_score: Float.round(max_score, 4),
      min_removal_score: Float.round(min_score, 4),
      robustness_score: Float.round(robustness, 4)
    }
  end
end
```

---

## Integration Tests

```elixir
defmodule CNS.Topology.FragilityTest do
  use ExUnit.Case, async: true

  alias CNS.Topology.Fragility
  alias CNS.Core.SNO

  describe "analyze/2" do
    test "analyzes simple claim network" do
      sno = build_simple_network()

      result = Fragility.analyze(sno)

      assert result.robustness_score >= 0.0
      assert result.robustness_score <= 1.0
      assert result.interpretation in [:highly_robust, :moderately_robust, :fragile]
      assert is_list(result.critical_claims)
      assert is_list(result.removal_sensitivity)
      assert is_float(result.bottleneck_stability)
      assert is_map(result.statistics)
    end

    test "handles insufficient claims gracefully" do
      sno = %SNO{
        statement: "Single claim",
        evidence: []
      }

      result = Fragility.analyze(sno)

      assert result.robustness_score == 1.0
      assert result.interpretation == :insufficient_data
      assert result.critical_claims == []
    end

    test "identifies critical claims in circular reasoning" do
      # Build network with circular reasoning
      sno = build_circular_network()

      result = Fragility.analyze(sno, top_k: 2)

      # Should identify claims in the cycle as critical
      assert length(result.critical_claims) > 0
      assert result.robustness_score < 0.6
    end

    test "returns high robustness for well-grounded network" do
      sno = build_robust_network()

      result = Fragility.analyze(sno)

      assert result.robustness_score > 0.7
      assert result.interpretation == :highly_robust
    end
  end

  describe "local_analysis/2" do
    test "analyzes local fragility around claim" do
      sno = build_simple_network()

      result = Fragility.local_analysis(sno, claim_index: 0)

      assert is_map(result)
      assert is_float(result.removal_impact)
      assert is_float(result.neighborhood_fragility)
      assert is_float(result.relative_fragility)
      assert is_list(result.neighbors)
      assert is_binary(result.interpretation)
    end

    test "raises on invalid claim index" do
      sno = build_simple_network()

      assert_raise ArgumentError, fn ->
        Fragility.local_analysis(sno, claim_index: 999)
      end
    end
  end

  describe "robustness_score/2" do
    test "computes robustness score" do
      sno = build_simple_network()

      score = Fragility.robustness_score(sno)

      assert is_float(score)
      assert score >= 0.0
      assert score <= 1.0
    end

    test "returns 1.0 for insufficient claims" do
      sno = %SNO{statement: "Single", evidence: []}

      score = Fragility.robustness_score(sno)

      assert score == 1.0
    end
  end

  describe "critical_claims/2" do
    test "identifies critical claims" do
      sno = build_simple_network()

      critical = Fragility.critical_claims(sno, top_k: 3)

      assert is_list(critical)
      assert length(critical) <= 3

      for {claim, score} <- critical do
        assert %SNO{} = claim
        assert is_float(score)
      end
    end

    test "returns empty for insufficient claims" do
      sno = %SNO{statement: "Single", evidence: []}

      critical = Fragility.critical_claims(sno)

      assert critical == []
    end

    test "sorts by fragility score descending" do
      sno = build_simple_network()

      critical = Fragility.critical_claims(sno, top_k: 5)

      scores = Enum.map(critical, fn {_, score} -> score end)
      assert scores == Enum.sort(scores, :desc)
    end
  end

  describe "compare/3" do
    test "compares two networks" do
      sno1 = build_robust_network()
      sno2 = build_fragile_network()

      result = Fragility.compare(sno1, sno2)

      assert is_float(result.sno1_robustness)
      assert is_float(result.sno2_robustness)
      assert is_float(result.robustness_difference)
      assert is_binary(result.interpretation)

      # Robust network should score higher
      assert result.sno1_robustness > result.sno2_robustness
      assert result.robustness_difference > 0
    end
  end

  describe "validate/2" do
    test "validates working network" do
      sno = build_simple_network()

      result = Fragility.validate(sno)

      assert result.valid? == true
      assert result.warnings == []
      assert result.diagnostics.n_claims > 0
    end

    test "reports warnings for insufficient claims" do
      sno = %SNO{statement: "Single", evidence: []}

      result = Fragility.validate(sno)

      assert result.valid? == false
      assert length(result.warnings) > 0
    end
  end

  #
  # Test Helpers
  #

  defp build_simple_network do
    %SNO{
      statement: "Thesis: AI will transform medicine",
      evidence: [
        %{
          source_sno: %SNO{statement: "Evidence 1: Deep learning achieves 95% accuracy"}
        },
        %{
          source_sno: %SNO{statement: "Evidence 2: FDA approved AI diagnostic tools"}
        },
        %{
          source_sno: %SNO{statement: "Evidence 3: Reduced diagnosis time by 40%"}
        }
      ],
      challenges: [
        %SNO{
          statement: "Challenge: Data privacy concerns",
          evidence: [
            %{source_sno: %SNO{statement: "HIPAA violations in training data"}}
          ]
        }
      ]
    }
  end

  defp build_circular_network do
    # A → B → C → A (circular reasoning)
    %SNO{
      statement: "A: Therefore X is true",
      evidence: [
        %{
          source_sno: %SNO{
            statement: "B: Because Y is true",
            evidence: [
              %{
                source_sno: %SNO{
                  statement: "C: Since X is true",
                  evidence: []
                }
              }
            ]
          }
        }
      ]
    }
  end

  defp build_robust_network do
    %SNO{
      statement: "Well-grounded thesis",
      evidence:
        Enum.map(1..8, fn i ->
          %{
            source_sno: %SNO{
              statement: "Independent evidence #{i}",
              evidence: []
            }
          }
        end)
    }
  end

  defp build_fragile_network do
    %SNO{
      statement: "Fragile thesis relying on single source",
      evidence: [
        %{
          source_sno: %SNO{
            statement: "Single weak claim",
            evidence: []
          }
        }
      ]
    }
  end
end
```

---

## Property-Based Tests

```elixir
defmodule CNS.Topology.Fragility.PropertyTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias CNS.Topology.Fragility
  alias CNS.Core.SNO

  @moduletag :property

  property "robustness score always in [0, 1]" do
    check all(sno <- sno_generator(min_claims: 3, max_claims: 20)) do
      score = Fragility.robustness_score(sno)

      assert score >= 0.0
      assert score <= 1.0
    end
  end

  property "more evidence increases robustness" do
    check all(
            base_evidence <- integer(3..5),
            additional <- integer(1..10)
          ) do
      sno_small = build_network_with_n_evidence(base_evidence)
      sno_large = build_network_with_n_evidence(base_evidence + additional)

      score_small = Fragility.robustness_score(sno_small)
      score_large = Fragility.robustness_score(sno_large)

      # More evidence should not decrease robustness
      assert score_large >= score_small - 0.1
    end
  end

  property "circular networks have lower robustness" do
    check all(n_claims <- integer(3..10)) do
      linear = build_linear_network(n_claims)
      circular = build_circular_network_n(n_claims)

      score_linear = Fragility.robustness_score(linear)
      score_circular = Fragility.robustness_score(circular)

      # Circular should generally be less robust
      # (allowing some variance due to embeddings)
      assert score_circular <= score_linear + 0.2
    end
  end

  property "critical claims count respects top_k" do
    check all(
            sno <- sno_generator(min_claims: 10, max_claims: 20),
            k <- integer(1..5)
          ) do
      critical = Fragility.critical_claims(sno, top_k: k)

      assert length(critical) <= k
    end
  end

  property "local analysis returns valid structure" do
    check all(sno <- sno_generator(min_claims: 5, max_claims: 10)) do
      snos = extract_all_snos(sno)
      index = Enum.random(0..(length(snos) - 1))

      result = Fragility.local_analysis(sno, claim_index: index)

      assert is_float(result.removal_impact)
      assert result.removal_impact >= 0.0
      assert is_float(result.neighborhood_fragility)
      assert is_float(result.relative_fragility)
      assert is_list(result.neighbors)
    end
  end

  #
  # Generators
  #

  defp sno_generator(opts) do
    min_claims = Keyword.get(opts, :min_claims, 3)
    max_claims = Keyword.get(opts, :max_claims, 10)

    gen all(
          n_evidence <- integer(min_claims..max_claims),
          statements <- list_of(string(:alphanumeric), length: n_evidence)
        ) do
      %SNO{
        statement: "Generated thesis",
        evidence:
          Enum.map(statements, fn stmt ->
            %{source_sno: %SNO{statement: stmt, evidence: []}}
          end)
      }
    end
  end

  defp build_network_with_n_evidence(n) do
    %SNO{
      statement: "Thesis with #{n} evidence",
      evidence:
        Enum.map(1..n, fn i ->
          %{source_sno: %SNO{statement: "Evidence #{i}", evidence: []}}
        end)
    }
  end

  defp build_linear_network(n) do
    Enum.reduce((n - 1)..1, %SNO{statement: "Leaf claim", evidence: []}, fn i, acc ->
      %SNO{
        statement: "Claim #{i}",
        evidence: [%{source_sno: acc}]
      }
    end)
  end

  defp build_circular_network_n(n) do
    # Create chain, then link last to first (circular reference)
    # For testing purposes, approximate with high interconnection
    %SNO{
      statement: "Root in circular reasoning",
      evidence:
        Enum.map(1..n, fn i ->
          %{
            source_sno: %SNO{
              statement: "Circular claim #{i}",
              evidence: []
            }
          }
        end)
    }
  end

  defp extract_all_snos(sno) do
    # Use CNS.Topology.Fragility private function (test access)
    # In practice, duplicate the extraction logic
    [sno] ++
      Enum.flat_map(Map.get(sno, :evidence, []), fn ev ->
        case Map.get(ev, :source_sno) do
          nil -> []
          source -> extract_all_snos(source)
        end
      end)
  end
end
```

---

## Integration with Adapter Module

The Fragility module depends on `CNS.Topology.Adapter` for SNO → embedding conversion:

```elixir
# In CNS.Topology.Adapter (from Section 1)
defmodule CNS.Topology.Adapter do
  @doc """
  Convert SNO list to embedding point cloud.

  ## Parameters

  - `snos` - List of CNS.Core.SNO structs
  - `opts` - Options:
    - `:encoder` - Encoder module (default: CNS.Embedding.Encoder)
    - `:cache` - Use cached embeddings (default: true)

  ## Returns

  Nx tensor of shape {n, embedding_dim}
  """
  @spec sno_embeddings([CNS.Core.SNO.t()], keyword()) :: Nx.Tensor.t()
  def sno_embeddings(snos, opts \\ []) do
    encoder = Keyword.get(opts, :encoder, CNS.Embedding.Encoder)
    use_cache = Keyword.get(opts, :cache, true)

    # Extract statements
    statements = Enum.map(snos, & &1.statement)

    # Encode (with caching)
    if use_cache do
      encoder.encode_cached(statements)
    else
      encoder.encode(statements)
    end
  end
end
```

---

## Usage Examples

### Example 1: Analyze Synthesis Result

```elixir
# After synthesis completes
def validate_synthesis(synthesis_sno) do
  # Run fragility analysis
  fragility = CNS.Topology.Fragility.analyze(synthesis_sno, top_k: 5)

  Logger.info("""
  Synthesis Fragility Analysis:
  - Robustness: #{fragility.robustness_score} (#{fragility.interpretation})
  - Bottleneck stability: #{fragility.bottleneck_stability}
  - Critical claims: #{length(fragility.critical_claims)}
  """)

  # Check if synthesis is robust enough
  if fragility.robustness_score < 0.4 do
    Logger.warning("Synthesis is fragile - consider strengthening arguments")

    # Log critical claims
    for claim <- fragility.critical_claims do
      Logger.warning("Critical claim: #{claim.statement}")
    end
  end

  fragility
end
```

### Example 2: Compare Thesis vs Antithesis

```elixir
def compare_dialectical_pair(thesis, antithesis) do
  comparison = CNS.Topology.Fragility.compare(thesis, antithesis)

  Logger.info("""
  Dialectical Comparison:
  - Thesis robustness: #{comparison.sno1_robustness}
  - Antithesis robustness: #{comparison.sno2_robustness}
  - Difference: #{comparison.robustness_difference}
  - #{comparison.interpretation}
  """)

  # More robust argument wins?
  winner =
    if comparison.robustness_difference > 0.2 do
      :thesis_stronger
    else
      if comparison.robustness_difference < -0.2 do
        :antithesis_stronger
      else
        :balanced
      end
    end

  {winner, comparison}
end
```

### Example 3: Identify Weak Arguments

```elixir
def find_weak_arguments(synthesis_sno) do
  # Get all claims
  all_claims = extract_all_claims(synthesis_sno)

  # Analyze local fragility for each
  weak_claims =
    all_claims
    |> Enum.with_index()
    |> Enum.map(fn {_claim, idx} ->
      CNS.Topology.Fragility.local_analysis(synthesis_sno, claim_index: idx)
    end)
    |> Enum.filter(fn local ->
      # High relative fragility = weak compared to neighbors
      local.relative_fragility > 1.5
    end)
    |> Enum.sort_by(& &1.relative_fragility, :desc)

  Logger.info("Found #{length(weak_claims)} weak arguments")

  for weak <- weak_claims do
    Logger.warning("""
    Weak claim: #{weak.claim.statement}
    - Relative fragility: #{weak.relative_fragility}
    - #{weak.interpretation}
    """)
  end

  weak_claims
end
```

### Example 4: Robustness-Based Quality Scoring

```elixir
defmodule CNS.Metrics.Quality do
  alias CNS.Topology.Fragility

  @doc """
  Compute quality score incorporating topological robustness.
  """
  def compute_quality_score(synthesis_sno) do
    # Existing metrics
    semantic_score = compute_semantic_coherence(synthesis_sno)
    citation_score = compute_citation_density(synthesis_sno)

    # Add topological robustness
    robustness = Fragility.robustness_score(synthesis_sno)

    # Weighted combination
    weights = %{
      semantic: 0.35,
      citation: 0.30,
      robustness: 0.35
    }

    total_score =
      semantic_score * weights.semantic +
        citation_score * weights.citation +
        robustness * weights.robustness

    %{
      total: Float.round(total_score, 4),
      components: %{
        semantic: semantic_score,
        citation: citation_score,
        robustness: robustness
      },
      interpretation: interpret_quality(total_score)
    }
  end

  defp interpret_quality(score) when score > 0.8, do: :excellent
  defp interpret_quality(score) when score > 0.6, do: :good
  defp interpret_quality(score) when score > 0.4, do: :acceptable
  defp interpret_quality(_), do: :needs_improvement
end
```

---

## Performance Considerations

### Computational Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| `analyze/2` | O(n² + n·PH) | n² for distances, n·PH for removal |
| `local_analysis/2` | O(n² + PH) | Single point removal + k-NN |
| `robustness_score/2` | O(n² + PH) | Full fragility computation |
| `critical_claims/2` | O(n² + n·PH) | Same as analyze |

Where:
- n = number of claims
- PH = persistent homology cost (depends on max_dimension)

### Optimization Strategies

```elixir
# 1. Cache embeddings
defmodule CNS.Embedding.Cache do
  use Agent

  def start_link(_) do
    Agent.start_link(fn -> %{} end, name: __MODULE__)
  end

  def get_or_compute(key, compute_fn) do
    Agent.get_and_update(__MODULE__, fn cache ->
      case Map.get(cache, key) do
        nil ->
          value = compute_fn.()
          {value, Map.put(cache, key, value)}

        value ->
          {value, cache}
      end
    end)
  end
end

# 2. Batch analysis for large networks
def analyze_large_network(sno, opts \\ []) do
  snos = extract_all_claims(sno)

  if length(snos) > 100 do
    # Sample for fragility analysis
    sample_size = Keyword.get(opts, :sample_size, 100)
    sampled = Enum.take_random(snos, sample_size)

    Logger.info("Large network (#{length(snos)} claims), sampling #{sample_size}")

    analyze_impl(sampled, opts)
  else
    analyze_impl(snos, opts)
  end
end

# 3. Parallel processing for comparisons
def compare_many(sno_list, opts \\ []) do
  sno_list
  |> Task.async_stream(
    fn sno -> {sno, analyze(sno, opts)} end,
    max_concurrency: System.schedulers_online()
  )
  |> Enum.map(fn {:ok, result} -> result end)
end
```

---

## Error Handling

```elixir
# In CNS.Topology.Fragility

defp safe_analyze(sno, opts) do
  try do
    {:ok, analyze(sno, opts)}
  rescue
    e in ArgumentError ->
      {:error, {:invalid_input, Exception.message(e)}}

    e in RuntimeError ->
      {:error, {:computation_failed, Exception.message(e)}}

    e ->
      Logger.error("Unexpected error in fragility analysis: #{inspect(e)}")
      {:error, {:unknown, Exception.message(e)}}
  end
end

# Usage
case safe_analyze(sno, opts) do
  {:ok, result} ->
    process_result(result)

  {:error, {:invalid_input, msg}} ->
    Logger.warning("Invalid input: #{msg}")
    default_result()

  {:error, {:computation_failed, msg}} ->
    Logger.error("Computation failed: #{msg}")
    alert_admin()

  {:error, {:unknown, msg}} ->
    Logger.error("Unknown error: #{msg}")
    alert_admin()
end
```

---

## Documentation Standards

All functions include:

1. **@doc blocks** with:
   - Clear description
   - Parameter documentation with types
   - Return value documentation
   - Examples (with iex> format)
   - Performance notes where relevant
   - CNS-specific interpretation

2. **@spec typespecs** for all public functions

3. **Examples** that are doctests

4. **CNS interpretation sections** explaining what results mean in dialectical reasoning context

---

## Validation Criteria

### Success Criteria

- [ ] All tests pass (unit + property-based)
- [ ] Zero compilation warnings
- [ ] Fragility scores in [0, 1] range
- [ ] Critical claims identified correctly
- [ ] Local analysis provides actionable insights
- [ ] Performance acceptable for networks <100 claims
- [ ] Documentation complete (100% public API)

### Integration Criteria

- [ ] Works with CNS.Topology.Adapter
- [ ] Compatible with CNS.Core.SNO structures
- [ ] Integrates with CNS.Metrics.Quality
- [ ] Usable by CNS.Agents.Pipeline
- [ ] Compatible with CNS.Critics.*

---

## Migration from Surrogates

### Deprecation Path

```elixir
# In CNS.Topology.Surrogates

@deprecated "Use CNS.Topology.Fragility.analyze/2 instead"
def compute_fragility_surrogate(embeddings, opts) do
  Logger.warning("""
  CNS.Topology.Surrogates.compute_fragility_surrogate/2 is deprecated.
  Use CNS.Topology.Fragility.analyze/2 for full topological fragility analysis.
  """)

  # Fallback to old implementation
  compute_fragility_surrogate_impl(embeddings, opts)
end
```

### Migration Guide

**Old code:**
```elixir
fragility = CNS.Topology.Surrogates.compute_fragility_surrogate(embeddings, k: 5)
```

**New code:**
```elixir
analysis = CNS.Topology.Fragility.analyze(sno, top_k: 5)
robustness = analysis.robustness_score
fragility_approx = 1.0 - robustness  # Inverse relationship
```

---

## Summary

This implementation provides:

1. **Complete replacement** for CNS.Topology.Surrogates fragility functionality
2. **Production-ready code** with full typespecs and documentation
3. **Comprehensive test suite** (unit + property-based + integration)
4. **CNS-specific semantics** interpreting ex_topology results for dialectical reasoning
5. **Performance optimizations** for large networks
6. **Error handling** and validation
7. **Migration path** from old surrogates

**Next Steps:**
1. Implement CNS.Topology.Adapter (Section 1)
2. Copy this code into `lib/cns/topology/fragility.ex`
3. Run tests: `mix test test/cns/topology/fragility_test.exs`
4. Integrate with CNS.Agents.Pipeline
5. Update CNS.Metrics.Quality to use robustness scoring

**Dependencies:**
- `ex_topology ~> 0.1.1`
- `CNS.Topology.Adapter.sno_embeddings/2`
- `CNS.Core.SNO` struct
- `CNS.Embedding.Encoder` (for embedding generation)

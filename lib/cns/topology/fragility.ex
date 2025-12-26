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
      synthesis = %CNS.SNO{...}
      analysis = CNS.Topology.Fragility.analyze(synthesis)

      analysis.robustness_score
      # => 0.73 (moderately robust)

      analysis.critical_claims
      # => [%CNS.SNO{claim: "Core premise"}, ...]

      # Analyze specific claim
      local = CNS.Topology.Fragility.local_analysis(synthesis, claim_index: 3)
      local.removal_impact
      # => 0.82 (high impact - removing this claim destabilizes network)
  """

  # ExTopology integration for topological fragility analysis
  alias CNS.SNO
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

  - `sno` - A CNS.SNO struct (synthesis or claim tree)
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
  @spec analyze(SNO.t(), keyword()) :: %{
          robustness_score: float(),
          interpretation: :highly_robust | :moderately_robust | :fragile | :insufficient_data,
          critical_claims: [SNO.t()],
          removal_sensitivity: [{SNO.t(), float()}],
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
        ExTopology.Fragility.point_removal_sensitivity(embeddings,
          max_dimension: max_dim,
          metric: :bottleneck
        )

      # Identify critical points
      critical_indices =
        ExTopology.Fragility.identify_critical_points(removal_scores,
          top_k: min(top_k, length(snos))
        )

      # Overall robustness
      robustness = ExTopology.Fragility.robustness_score(embeddings)

      # Bottleneck stability
      stability = ExTopology.Fragility.bottleneck_stability(embeddings, num_samples: 10)

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
  @spec local_analysis(SNO.t(), keyword()) :: %{
          claim: SNO.t(),
          removal_impact: float(),
          neighborhood_fragility: float(),
          relative_fragility: float(),
          neighbors: [SNO.t()],
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
    local = ExTopology.Fragility.local_fragility(embeddings, claim_index, k: k)

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
  @spec robustness_score(SNO.t(), keyword()) :: float()
  def robustness_score(sno, opts \\ []) do
    snos = extract_all_claims(sno)

    if length(snos) < 3 do
      1.0
    else
      embeddings = Adapter.sno_embeddings(snos, Keyword.get(opts, :embedding_opts, []))
      ExTopology.Fragility.robustness_score(embeddings) |> Float.round(4)
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
  @spec critical_claims(SNO.t(), keyword()) :: [{SNO.t(), float()}]
  def critical_claims(sno, opts \\ []) do
    top_k = Keyword.get(opts, :top_k, 5)
    max_dim = Keyword.get(opts, :max_dimension, 1)

    snos = extract_all_claims(sno)

    if length(snos) < 3 do
      []
    else
      embeddings = Adapter.sno_embeddings(snos, Keyword.get(opts, :embedding_opts, []))

      removal_scores =
        ExTopology.Fragility.point_removal_sensitivity(embeddings,
          max_dimension: max_dim,
          metric: :bottleneck
        )

      critical_indices =
        ExTopology.Fragility.identify_critical_points(removal_scores,
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
  @spec compare(SNO.t(), SNO.t(), keyword()) :: map()
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
  @spec validate(SNO.t(), keyword()) :: map()
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
    {embeddings, warnings} =
      try do
        emb = Adapter.sno_embeddings(snos, Keyword.get(opts, :embedding_opts, []))
        {emb, warnings}
      rescue
        e ->
          {nil, ["Failed to generate embeddings: #{inspect(e)}" | warnings]}
      end

    # Check 3: Fragility computation works
    {fragility_works, warnings} =
      if embeddings && length(snos) >= 3 do
        try do
          _rob = ExTopology.Fragility.robustness_score(embeddings)
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
          _scores = ExTopology.Fragility.point_removal_sensitivity(embeddings, max_dimension: 1)
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

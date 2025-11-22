# Critic Pipeline Architecture

## Version
- Date: 2025-11-21
- Status: Design Document
- Author: Architecture Team

## Overview

The critic pipeline evaluates SNO quality through five specialized critics, each implemented as GenServers that leverage Tinkex for ML inference.

## Critic Behaviour

```elixir
defmodule CNS.Critic do
  @moduledoc """
  Behaviour for CNS critics.
  """

  @type evaluation_result :: %{
    score: float(),
    issues: [issue()],
    explanation: String.t(),
    suggestions: [String.t()],
    metadata: map()
  }

  @type issue :: %{
    severity: :critical | :major | :minor,
    type: atom(),
    description: String.t(),
    location: String.t() | nil
  }

  @callback evaluate(sno :: CNS.SNO.t(), opts :: keyword()) ::
    {:ok, evaluation_result()} | {:error, term()}

  @callback name() :: atom()

  @callback weight() :: float()

  @callback threshold() :: float()

  @doc """
  Aggregates results from multiple critics.
  """
  @spec aggregate([{atom(), evaluation_result()}]) :: map()
  def aggregate(results) do
    total_weight = Enum.reduce(results, 0, fn {critic, _}, acc ->
      acc + critic_module(critic).weight()
    end)

    weighted_score = Enum.reduce(results, 0, fn {critic, result}, acc ->
      weight = critic_module(critic).weight()
      acc + (result.score * weight)
    end)

    overall_score = if total_weight > 0, do: weighted_score / total_weight, else: 0

    all_issues = Enum.flat_map(results, fn {_, result} -> result.issues end)
    |> Enum.sort_by(& &1.severity, :desc)

    critical_issues = Enum.filter(all_issues, & &1.severity == :critical)

    passed = overall_score >= 0.6 and critical_issues == []

    %{
      overall_score: overall_score,
      individual_scores: Map.new(results, fn {critic, r} -> {critic, r.score} end),
      issues: all_issues,
      critical_issues: critical_issues,
      passed: passed
    }
  end

  defp critic_module(:logic), do: CNS.Critics.Logic
  defp critic_module(:grounding), do: CNS.Critics.Grounding
  defp critic_module(:novelty), do: CNS.Critics.Novelty
  defp critic_module(:causal), do: CNS.Critics.Causal
  defp critic_module(:bias), do: CNS.Critics.Bias
end
```

## Critic Supervisor

```elixir
defmodule CNS.CriticSupervisor do
  @moduledoc """
  Supervises all critic GenServers.
  """

  use Supervisor

  def start_link(opts) do
    Supervisor.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @impl true
  def init(_opts) do
    children = [
      {CNS.Critics.Logic, []},
      {CNS.Critics.Grounding, []},
      {CNS.Critics.Novelty, []},
      {CNS.Critics.Causal, []},
      {CNS.Critics.Bias, []}
    ]

    Supervisor.init(children, strategy: :one_for_one)
  end
end
```

## Logic Critic

```elixir
defmodule CNS.Critics.Logic do
  @moduledoc """
  Evaluates logical consistency of SNOs.

  Detects:
  - Contradictions between claims
  - Non-sequiturs in synthesis
  - Invalid inference patterns
  - Circular reasoning
  """

  use GenServer

  @behaviour CNS.Critic

  defstruct [:config, :sampling_client, :model_registry]

  @impl CNS.Critic
  def name, do: :logic

  @impl CNS.Critic
  def weight, do: 0.25

  @impl CNS.Critic
  def threshold, do: 0.7

  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Evaluates logical consistency of an SNO.
  """
  @impl CNS.Critic
  def evaluate(sno, opts \\ []) do
    GenServer.call(__MODULE__, {:evaluate, sno, opts}, 30_000)
  end

  @impl true
  def init(opts) do
    config = Keyword.get(opts, :config, default_config())
    {:ok, %__MODULE__{config: config}}
  end

  @impl true
  def handle_call({:evaluate, sno, opts}, _from, state) do
    result = perform_evaluation(sno, opts, state)
    {:reply, result, state}
  end

  defp perform_evaluation(sno, opts, state) do
    # 1. Graph-based analysis
    graph_issues = analyze_graph_structure(sno)

    # 2. NLI-based contradiction detection
    nli_issues = detect_contradictions(sno, state)

    # 3. LLM-based logical analysis
    llm_result = llm_logical_analysis(sno, opts, state)

    # Combine results
    all_issues = graph_issues ++ nli_issues ++ llm_result.issues

    score = compute_score(all_issues, llm_result.score)

    {:ok, %{
      score: score,
      issues: all_issues,
      explanation: llm_result.explanation,
      suggestions: generate_suggestions(all_issues),
      metadata: %{
        graph_analysis: graph_issues,
        nli_analysis: nli_issues,
        llm_analysis: llm_result
      }
    }}
  end

  defp analyze_graph_structure(sno) do
    issues = []

    # Check for cycles (circular reasoning)
    if CNS.SNO.Graph.has_invalid_cycles?(sno.graph) do
      [%{
        severity: :major,
        type: :circular_reasoning,
        description: "Graph contains circular reasoning patterns",
        location: nil
      } | issues]
    else
      issues
    end

    # Check for disconnected evidence
    disconnected = find_disconnected_evidence(sno)
    if disconnected != [] do
      [%{
        severity: :minor,
        type: :unused_evidence,
        description: "Evidence not connected to any claim: #{inspect(disconnected)}",
        location: nil
      } | issues]
    else
      issues
    end
  end

  defp detect_contradictions(sno, state) do
    # Use NLI model via Tinkex to detect contradictions
    claims = [
      {"thesis", sno.thesis},
      {"antithesis", sno.antithesis},
      {"synthesis", sno.synthesis}
    ]

    # Check pairwise for contradictions not captured by dialectical structure
    pairs = for {id1, c1} <- claims, {id2, c2} <- claims, id1 < id2 do
      {id1, id2, c1, c2}
    end

    Enum.flat_map(pairs, fn {id1, id2, claim1, claim2} ->
      case nli_entailment(claim1, claim2, state) do
        {:contradiction, confidence} when confidence > 0.8 ->
          # Only flag if unexpected contradiction
          if unexpected_contradiction?(id1, id2) do
            [%{
              severity: :major,
              type: :contradiction,
              description: "Contradiction between #{id1} and #{id2}",
              location: "#{id1} <-> #{id2}"
            }]
          else
            []
          end

        _ ->
          []
      end
    end)
  end

  defp llm_logical_analysis(sno, opts, _state) do
    prompt = format_logic_prompt(sno)

    # Use ensemble if configured
    result = if opts[:use_ensemble] do
      Crucible.Ensemble.Critics.evaluate(opts[:logic_pool], sno, :logic)
    else
      # Direct LLM call
      {:ok, response} = generate_analysis(prompt, opts)
      parse_logic_response(response)
    end

    result
  end

  defp format_logic_prompt(sno) do
    """
    Analyze the logical consistency of this dialectical synthesis.

    THESIS: #{sno.thesis}

    ANTITHESIS: #{sno.antithesis}

    SYNTHESIS: #{sno.synthesis || "(not yet generated)"}

    EVIDENCE:
    #{format_evidence_for_prompt(sno.evidence)}

    Evaluate for:
    1. Internal contradictions not resolved by synthesis
    2. Non-sequiturs or invalid inferences
    3. Unsupported logical leaps
    4. Fallacies (straw man, false dichotomy, etc.)

    Provide:
    - A score from 0.0 to 1.0
    - List of specific issues found
    - Explanation of reasoning
    """
  end

  defp compute_score(issues, llm_score) do
    # Reduce score based on issues
    penalty = Enum.reduce(issues, 0, fn issue, acc ->
      case issue.severity do
        :critical -> acc + 0.3
        :major -> acc + 0.15
        :minor -> acc + 0.05
      end
    end)

    max(0.0, llm_score - penalty)
  end

  defp generate_suggestions(issues) do
    Enum.flat_map(issues, fn issue ->
      case issue.type do
        :circular_reasoning ->
          ["Restructure argument to remove circular dependencies"]
        :contradiction ->
          ["Revise synthesis to explicitly address the contradiction at #{issue.location}"]
        :unused_evidence ->
          ["Either cite the unused evidence or remove it"]
        _ ->
          []
      end
    end)
  end

  defp default_config do
    %{
      temperature: 0.3,
      max_tokens: 1024
    }
  end

  defp find_disconnected_evidence(sno) do
    Map.keys(sno.evidence)
    |> Enum.filter(fn ev_id ->
      edges = CNS.SNO.Graph.edges_from(sno.graph, ev_id)
      edges == []
    end)
  end

  defp nli_entailment(_claim1, _claim2, _state) do
    # Placeholder - would call NLI model via Tinkex
    {:neutral, 0.5}
  end

  defp unexpected_contradiction?(id1, id2) do
    # Thesis-antithesis contradiction is expected
    not (id1 == "thesis" and id2 == "antithesis")
  end

  defp generate_analysis(_prompt, _opts) do
    # Placeholder - would call Tinkex
    {:ok, %{score: 0.85, explanation: "", issues: []}}
  end

  defp parse_logic_response(_response) do
    %{score: 0.85, explanation: "", issues: []}
  end

  defp format_evidence_for_prompt(evidence_map) do
    evidence_map
    |> Enum.map(fn {id, ev} -> "[#{id}] #{ev.content}" end)
    |> Enum.join("\n")
  end
end
```

## Grounding Critic

```elixir
defmodule CNS.Critics.Grounding do
  @moduledoc """
  Verifies that claims are properly grounded in evidence.

  Checks:
  - Citation accuracy
  - Evidence relevance
  - Claim support strength
  """

  use GenServer

  @behaviour CNS.Critic

  @impl CNS.Critic
  def name, do: :grounding

  @impl CNS.Critic
  def weight, do: 0.30

  @impl CNS.Critic
  def threshold, do: 0.8

  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @impl CNS.Critic
  def evaluate(sno, opts \\ []) do
    GenServer.call(__MODULE__, {:evaluate, sno, opts}, 30_000)
  end

  @impl true
  def init(opts) do
    {:ok, %{config: Keyword.get(opts, :config, %{})}}
  end

  @impl true
  def handle_call({:evaluate, sno, opts}, _from, state) do
    result = perform_evaluation(sno, opts, state)
    {:reply, result, state}
  end

  defp perform_evaluation(sno, opts, state) do
    # Skip if no synthesis yet
    if is_nil(sno.synthesis) do
      {:ok, %{
        score: 1.0,
        issues: [],
        explanation: "No synthesis to evaluate",
        suggestions: [],
        metadata: %{}
      }}
    else
      # 1. Check citation validity
      citation_issues = check_citations(sno)

      # 2. NLI entailment for each citation
      entailment_issues = check_entailment(sno, state)

      # 3. Coverage analysis
      coverage = analyze_coverage(sno)

      all_issues = citation_issues ++ entailment_issues

      score = compute_grounding_score(all_issues, coverage)

      {:ok, %{
        score: score,
        issues: all_issues,
        explanation: format_explanation(all_issues, coverage),
        suggestions: generate_suggestions(all_issues),
        metadata: %{
          citation_validity: 1.0 - (length(citation_issues) * 0.1),
          entailment_score: 1.0 - (length(entailment_issues) * 0.1),
          coverage: coverage
        }
      }}
    end
  end

  defp check_citations(sno) do
    # Get all citations from synthesis
    citations = extract_citations(sno.synthesis)

    # Check each citation exists
    invalid = Enum.filter(citations, fn cite_id ->
      not Map.has_key?(sno.evidence, cite_id)
    end)

    Enum.map(invalid, fn cite_id ->
      %{
        severity: :critical,
        type: :invalid_citation,
        description: "Citation [#{cite_id}] not found in evidence pool",
        location: cite_id
      }
    end)
  end

  defp check_entailment(sno, _state) do
    # For each citation, verify the evidence supports the claim
    citations = extract_citations(sno.synthesis)

    Enum.flat_map(citations, fn cite_id ->
      case Map.get(sno.evidence, cite_id) do
        nil -> []  # Already caught by citation check
        evidence ->
          # Extract the specific claim being supported
          claim_context = extract_claim_context(sno.synthesis, cite_id)

          # Check NLI entailment
          case nli_check(evidence.content, claim_context) do
            {:entailment, _} -> []
            {:neutral, _} ->
              [%{
                severity: :minor,
                type: :weak_support,
                description: "Evidence [#{cite_id}] only weakly supports the claim",
                location: cite_id
              }]
            {:contradiction, _} ->
              [%{
                severity: :major,
                type: :contradicting_citation,
                description: "Evidence [#{cite_id}] contradicts the claim it supports",
                location: cite_id
              }]
          end
      end
    end)
  end

  defp analyze_coverage(sno) do
    total_evidence = map_size(sno.evidence)
    citations = extract_citations(sno.synthesis) |> Enum.uniq()
    cited_count = length(citations)

    %{
      total_evidence: total_evidence,
      cited_count: cited_count,
      coverage_ratio: if(total_evidence > 0, do: cited_count / total_evidence, else: 1.0),
      uncited: Map.keys(sno.evidence) -- citations
    }
  end

  defp compute_grounding_score(issues, coverage) do
    base_score = 1.0

    # Penalize for issues
    issue_penalty = Enum.reduce(issues, 0, fn issue, acc ->
      case issue.severity do
        :critical -> acc + 0.25
        :major -> acc + 0.1
        :minor -> acc + 0.03
      end
    end)

    # Bonus for good coverage
    coverage_bonus = coverage.coverage_ratio * 0.1

    max(0.0, min(1.0, base_score - issue_penalty + coverage_bonus))
  end

  defp format_explanation(issues, coverage) do
    """
    Grounding Analysis:
    - #{length(issues)} issues found
    - Evidence coverage: #{Float.round(coverage.coverage_ratio * 100, 1)}%
    - Uncited evidence: #{inspect(coverage.uncited)}
    """
  end

  defp generate_suggestions(issues) do
    Enum.flat_map(issues, fn issue ->
      case issue.type do
        :invalid_citation ->
          ["Remove invalid citation [#{issue.location}] or add corresponding evidence"]
        :weak_support ->
          ["Strengthen connection between [#{issue.location}] and claim, or cite additional evidence"]
        :contradicting_citation ->
          ["Review citation [#{issue.location}] - it appears to contradict the claim"]
        _ ->
          []
      end
    end)
  end

  defp extract_citations(text) when is_binary(text) do
    Regex.scan(~r/\[([eE]\d+)\]/, text)
    |> Enum.map(fn [_, id] -> String.downcase(id) end)
  end
  defp extract_citations(_), do: []

  defp extract_claim_context(text, cite_id) do
    # Extract the sentence containing the citation
    sentences = String.split(text, ~r/[.!?]/)
    Enum.find(sentences, fn s ->
      String.contains?(String.downcase(s), "[#{cite_id}]")
    end) || text
  end

  defp nli_check(_evidence, _claim) do
    # Placeholder - would call NLI model via Tinkex
    {:entailment, 0.9}
  end
end
```

## Novelty Critic

```elixir
defmodule CNS.Critics.Novelty do
  @moduledoc """
  Assesses novelty and information gain in synthesis.
  """

  use GenServer

  @behaviour CNS.Critic

  @impl CNS.Critic
  def name, do: :novelty

  @impl CNS.Critic
  def weight, do: 0.15

  @impl CNS.Critic
  def threshold, do: 0.3

  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @impl CNS.Critic
  def evaluate(sno, opts \\ []) do
    GenServer.call(__MODULE__, {:evaluate, sno, opts}, 30_000)
  end

  @impl true
  def init(opts) do
    {:ok, %{config: Keyword.get(opts, :config, %{})}}
  end

  @impl true
  def handle_call({:evaluate, sno, opts}, _from, state) do
    if is_nil(sno.synthesis) do
      {:reply, {:ok, %{score: 0.0, issues: [], explanation: "No synthesis", suggestions: [], metadata: %{}}}, state}
    else
      result = evaluate_novelty(sno, opts)
      {:reply, result, state}
    end
  end

  defp evaluate_novelty(sno, _opts) do
    # 1. Semantic similarity to source texts
    thesis_sim = semantic_similarity(sno.synthesis, sno.thesis)
    antithesis_sim = semantic_similarity(sno.synthesis, sno.antithesis)

    # Synthesis should be different from both
    max_sim = max(thesis_sim, antithesis_sim)

    # 2. Information content analysis
    info_gain = compute_information_gain(sno)

    # 3. Novel insights detection
    novel_concepts = detect_novel_concepts(sno)

    issues = if max_sim > 0.9 do
      [%{
        severity: :major,
        type: :low_novelty,
        description: "Synthesis too similar to source texts",
        location: nil
      }]
    else
      []
    end

    score = compute_novelty_score(max_sim, info_gain, novel_concepts)

    {:ok, %{
      score: score,
      issues: issues,
      explanation: """
      Novelty Analysis:
      - Max similarity to sources: #{Float.round(max_sim * 100, 1)}%
      - Information gain: #{Float.round(info_gain * 100, 1)}%
      - Novel concepts: #{length(novel_concepts)}
      """,
      suggestions: if(score < 0.3, do: ["Add novel insights beyond restating sources"], else: []),
      metadata: %{
        thesis_similarity: thesis_sim,
        antithesis_similarity: antithesis_sim,
        information_gain: info_gain,
        novel_concepts: novel_concepts
      }
    }}
  end

  defp semantic_similarity(_text1, _text2) do
    # Would use embedding comparison via Tinkex
    0.6
  end

  defp compute_information_gain(_sno) do
    # Would compute based on embedding space coverage
    0.5
  end

  defp detect_novel_concepts(_sno) do
    # Would use NLP to find concepts in synthesis not in sources
    ["reconciliation", "conditional-efficacy"]
  end

  defp compute_novelty_score(max_sim, info_gain, novel_concepts) do
    dissimilarity = 1 - max_sim
    concept_bonus = min(0.2, length(novel_concepts) * 0.05)

    (dissimilarity * 0.5) + (info_gain * 0.3) + concept_bonus
  end
end
```

## Causal and Bias Critics

```elixir
defmodule CNS.Critics.Causal do
  @moduledoc """
  Validates causal claims and reasoning.
  """

  use GenServer
  @behaviour CNS.Critic

  @impl CNS.Critic
  def name, do: :causal

  @impl CNS.Critic
  def weight, do: 0.20

  @impl CNS.Critic
  def threshold, do: 0.6

  def start_link(opts), do: GenServer.start_link(__MODULE__, opts, name: __MODULE__)

  @impl CNS.Critic
  def evaluate(sno, opts \\ []) do
    GenServer.call(__MODULE__, {:evaluate, sno, opts}, 30_000)
  end

  @impl true
  def init(opts), do: {:ok, %{config: Keyword.get(opts, :config, %{})}}

  @impl true
  def handle_call({:evaluate, sno, opts}, _from, state) do
    result = evaluate_causal(sno, opts)
    {:reply, result, state}
  end

  defp evaluate_causal(sno, _opts) do
    # Extract causal claims
    causal_claims = extract_causal_claims(sno)

    # Validate each
    issues = Enum.flat_map(causal_claims, &validate_causal_claim(&1, sno))

    score = 1.0 - (length(issues) * 0.15)

    {:ok, %{
      score: max(0.0, score),
      issues: issues,
      explanation: "Found #{length(causal_claims)} causal claims, #{length(issues)} issues",
      suggestions: [],
      metadata: %{causal_claims: causal_claims}
    }}
  end

  defp extract_causal_claims(_sno) do
    # Would use NLP to find "causes", "leads to", "results in", etc.
    []
  end

  defp validate_causal_claim(_claim, _sno) do
    # Would check for evidence supporting causation
    []
  end
end

defmodule CNS.Critics.Bias do
  @moduledoc """
  Detects systematic biases in synthesis.
  """

  use GenServer
  @behaviour CNS.Critic

  @impl CNS.Critic
  def name, do: :bias

  @impl CNS.Critic
  def weight, do: 0.10

  @impl CNS.Critic
  def threshold, do: 0.5

  def start_link(opts), do: GenServer.start_link(__MODULE__, opts, name: __MODULE__)

  @impl CNS.Critic
  def evaluate(sno, opts \\ []) do
    GenServer.call(__MODULE__, {:evaluate, sno, opts}, 30_000)
  end

  @impl true
  def init(opts), do: {:ok, %{config: Keyword.get(opts, :config, %{})}}

  @impl true
  def handle_call({:evaluate, sno, opts}, _from, state) do
    result = evaluate_bias(sno, opts)
    {:reply, result, state}
  end

  defp evaluate_bias(sno, _opts) do
    # Check dialectical balance
    balance = compute_balance(sno)

    # Check for loaded language
    loaded_terms = detect_loaded_language(sno.synthesis)

    issues = []
    issues = if balance < 0.3 or balance > 0.7 do
      [%{
        severity: :major,
        type: :imbalanced,
        description: "Synthesis favors one side (balance: #{Float.round(balance, 2)})",
        location: nil
      } | issues]
    else
      issues
    end

    issues = if length(loaded_terms) > 2 do
      [%{
        severity: :minor,
        type: :loaded_language,
        description: "Uses loaded terms: #{inspect(loaded_terms)}",
        location: nil
      } | issues]
    else
      issues
    end

    # Score based on balance and issues
    score = compute_bias_score(balance, issues)

    {:ok, %{
      score: score,
      issues: issues,
      explanation: "Balance score: #{Float.round(balance, 2)}",
      suggestions: generate_bias_suggestions(issues),
      metadata: %{balance: balance, loaded_terms: loaded_terms}
    }}
  end

  defp compute_balance(_sno) do
    # Would analyze how much synthesis incorporates each side
    0.5
  end

  defp detect_loaded_language(_text) do
    # Would scan for emotionally charged terms
    []
  end

  defp compute_bias_score(balance, issues) do
    balance_score = 1 - abs(0.5 - balance) * 2
    penalty = length(issues) * 0.1
    max(0.0, balance_score - penalty)
  end

  defp generate_bias_suggestions(issues) do
    Enum.flat_map(issues, fn issue ->
      case issue.type do
        :imbalanced -> ["Revise synthesis to more equally acknowledge both perspectives"]
        :loaded_language -> ["Use more neutral language"]
        _ -> []
      end
    end)
  end
end
```

## Pipeline Orchestration

```elixir
defmodule CNS.Critics.Pipeline do
  @moduledoc """
  Orchestrates evaluation across all critics.
  """

  @critics [:logic, :grounding, :novelty, :causal, :bias]

  @doc """
  Evaluates SNO with all critics in parallel.
  """
  def evaluate_all(sno, opts \\ []) do
    critics = Keyword.get(opts, :critics, @critics)

    tasks = Enum.map(critics, fn critic ->
      Task.async(fn ->
        module = critic_module(critic)
        {critic, module.evaluate(sno, opts)}
      end)
    end)

    results = Task.await_many(tasks, 30_000)
    |> Enum.map(fn {critic, {:ok, result}} -> {critic, result} end)

    {:ok, CNS.Critic.aggregate(results)}
  end

  @doc """
  Evaluates with ensemble critics.
  """
  def evaluate_with_ensemble(sno, session, opts \\ []) do
    critics = Keyword.get(opts, :critics, @critics)

    # Create ensemble pools for each critic
    pools = Enum.map(critics, fn critic ->
      {:ok, pool} = Crucible.Ensemble.Critics.create_critic_ensemble(session, critic)
      {critic, pool}
    end)

    Crucible.Ensemble.Critics.evaluate_all(Map.new(pools), sno, opts)
  end

  defp critic_module(:logic), do: CNS.Critics.Logic
  defp critic_module(:grounding), do: CNS.Critics.Grounding
  defp critic_module(:novelty), do: CNS.Critics.Novelty
  defp critic_module(:causal), do: CNS.Critics.Causal
  defp critic_module(:bias), do: CNS.Critics.Bias
end
```

## Usage Example

```elixir
# Evaluate with all critics
{:ok, results} = CNS.Critics.Pipeline.evaluate_all(sno)

IO.inspect(results)
# %{
#   overall_score: 0.82,
#   individual_scores: %{
#     logic: 0.9,
#     grounding: 0.85,
#     novelty: 0.6,
#     causal: 0.88,
#     bias: 0.75
#   },
#   issues: [...],
#   critical_issues: [],
#   passed: true
# }

# Evaluate with ensemble models
{:ok, results} = CNS.Critics.Pipeline.evaluate_with_ensemble(sno, session)
```

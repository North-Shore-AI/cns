defmodule CNS.Critics.Logic do
  @moduledoc """
  Logic Critic for evaluating logical consistency of SNOs.

  Checks for:
  - Circular reasoning (cycles in evidence chains)
  - Contradictions between claims
  - Logical entailment validity
  - Argument structure coherence

  ## Examples

      iex> {:ok, logic} = CNS.Critics.Logic.start_link([])
      iex> sno = CNS.SNO.new("Consistent claim")
      iex> {:ok, result} = CNS.Critics.Logic.evaluate(logic, sno)
      iex> result.score > 0
      true
  """

  use GenServer

  @behaviour CNS.Critics.Critic

  alias CNS.{Evidence, SNO}

  # Client API

  @doc """
  Start a Logic critic GenServer.
  """
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Evaluate an SNO for logical consistency via GenServer.
  """
  @spec call(GenServer.server(), SNO.t(), keyword()) ::
          {:ok, CNS.Critics.Critic.evaluation_result()} | {:error, term()}
  def call(server, %SNO{} = sno, opts \\ []) do
    GenServer.call(server, {:evaluate, sno, opts})
  end

  # Behaviour callbacks

  @impl CNS.Critics.Critic
  def name, do: :logic

  @impl CNS.Critics.Critic
  def weight, do: 0.3

  @impl CNS.Critics.Critic
  def evaluate(%SNO{} = sno) do
    # Synchronous evaluation without GenServer
    do_evaluate(sno, %{})
  end

  # GenServer callbacks

  @impl GenServer
  def init(opts) do
    state = %{
      evaluations: 0,
      total_score: 0.0,
      opts: opts
    }

    {:ok, state}
  end

  @impl GenServer
  def handle_call({:evaluate, sno, _opts}, _from, state) do
    result = do_evaluate(sno, state)

    new_state =
      case result do
        {:ok, %{score: score}} ->
          %{
            state
            | evaluations: state.evaluations + 1,
              total_score: state.total_score + score
          }

        _ ->
          state
      end

    {:reply, result, new_state}
  end

  @impl GenServer
  def handle_call(:stats, _from, state) do
    avg_score =
      if state.evaluations > 0 do
        state.total_score / state.evaluations
      else
        0.0
      end

    stats = %{
      evaluations: state.evaluations,
      average_score: Float.round(avg_score, 4)
    }

    {:reply, {:ok, stats}, state}
  end

  # Private evaluation functions

  defp do_evaluate(%SNO{} = sno, _state) do
    issues = []

    # Check for circular reasoning
    {cycle_score, cycle_issues} = check_circular_reasoning(sno)
    issues = issues ++ cycle_issues

    # Check for contradictions
    {contradiction_score, contradiction_issues} = check_contradictions(sno)
    issues = issues ++ contradiction_issues

    # Check logical entailment
    {entailment_score, entailment_issues} = check_entailment(sno)
    issues = issues ++ entailment_issues

    # Check argument structure
    {structure_score, structure_issues} = check_argument_structure(sno)
    issues = issues ++ structure_issues

    # Calculate final score (weighted average)
    score =
      (cycle_score * 0.3 +
         contradiction_score * 0.3 +
         entailment_score * 0.2 +
         structure_score * 0.2)
      |> Float.round(4)

    result = %{
      score: score,
      issues: issues,
      details: %{
        cycle_score: cycle_score,
        contradiction_score: contradiction_score,
        entailment_score: entailment_score,
        structure_score: structure_score,
        cycles_found: length(Enum.filter(cycle_issues, &String.contains?(&1, "circular"))),
        contradictions_found:
          length(Enum.filter(contradiction_issues, &String.contains?(&1, "contradiction")))
      }
    }

    {:ok, result}
  rescue
    e -> {:error, Exception.message(e)}
  end

  defp check_circular_reasoning(%SNO{evidence: evidence}) do
    # Build evidence dependency graph
    edges = build_evidence_edges(evidence)

    # Detect cycles using DFS
    cycles = detect_cycles(edges)

    if Enum.empty?(cycles) do
      {1.0, []}
    else
      # Penalize based on number of cycles
      penalty = min(1.0, length(cycles) * 0.2)
      issues = Enum.map(cycles, fn cycle -> "circular_reasoning: #{inspect(cycle)}" end)
      {max(0.0, 1.0 - penalty), issues}
    end
  end

  defp build_evidence_edges(evidence) do
    # Build edges from evidence sources to their citations
    Enum.reduce(evidence, %{}, fn %Evidence{id: id, source: source}, acc ->
      # Check if source references another evidence
      refs =
        evidence
        |> Enum.filter(fn e -> String.contains?(source || "", e.id || "") end)
        |> Enum.map(& &1.id)

      Map.put(acc, id, refs)
    end)
  end

  defp detect_cycles(edges) do
    # Simple cycle detection
    nodes = Map.keys(edges)

    Enum.reduce(nodes, [], fn node, cycles ->
      if has_cycle?(edges, node, node, %{}) do
        [node | cycles]
      else
        cycles
      end
    end)
  end

  @spec has_cycle?(map(), any(), any(), map()) :: boolean()
  defp has_cycle?(edges, start, current, visited) when is_map(visited) do
    neighbors = Map.get(edges, current, [])

    cond do
      start in neighbors and map_size(visited) > 0 ->
        true

      Map.has_key?(visited, current) ->
        false

      true ->
        new_visited = Map.put(visited, current, true)

        Enum.any?(neighbors, fn neighbor ->
          has_cycle?(edges, start, neighbor, new_visited)
        end)
    end
  end

  defp check_contradictions(%SNO{claim: claim, children: children}) do
    # Check for contradictory statements
    all_claims = [claim | Enum.map(children, & &1.claim)]

    contradictions =
      for {c1, i} <- Enum.with_index(all_claims),
          {c2, j} <- Enum.with_index(all_claims),
          i < j,
          are_contradictory?(c1, c2) do
        {c1, c2}
      end

    if Enum.empty?(contradictions) do
      {1.0, []}
    else
      penalty = min(1.0, length(contradictions) * 0.3)

      issues =
        Enum.map(contradictions, fn {c1, c2} ->
          "contradiction: '#{truncate(c1)}' vs '#{truncate(c2)}'"
        end)

      {max(0.0, 1.0 - penalty), issues}
    end
  end

  defp are_contradictory?(claim1, claim2) do
    # Simple heuristic: check for negation patterns
    negation_patterns = [
      {~r/\b(is|are|was|were)\b/i,
       ~r/\b(is not|are not|isn't|aren't|was not|wasn't|were not|weren't)\b/i},
      {~r/\bcan\b/i, ~r/\bcannot|can't\b/i},
      {~r/\bwill\b/i, ~r/\bwill not|won't\b/i},
      {~r/\btrue\b/i, ~r/\bfalse\b/i},
      {~r/\byes\b/i, ~r/\bno\b/i}
    ]

    # Check if one claim negates the other
    Enum.any?(negation_patterns, fn {pos, neg} ->
      (Regex.match?(pos, claim1) and Regex.match?(neg, claim2)) or
        (Regex.match?(neg, claim1) and Regex.match?(pos, claim2))
    end)
  end

  defp check_entailment(%SNO{claim: _claim, evidence: []}),
    do: {0.5, ["no_evidence: claim has no supporting evidence"]}

  defp check_entailment(%SNO{claim: claim, evidence: evidence}) do
    support_scores = Enum.map(evidence, & &1.validity)
    avg_support = Enum.sum(support_scores) / length(support_scores)

    claim_words = extract_words(claim)
    avg_relevance = calculate_avg_relevance(evidence, claim_words)

    score = (avg_support * 0.6 + avg_relevance * 0.4) |> Float.round(4)

    issues =
      if score < 0.5,
        do: ["weak_entailment: evidence weakly supports claim (#{Float.round(score, 2)})"],
        else: []

    {score, issues}
  end

  defp calculate_avg_relevance(evidence, claim_words) do
    relevance_scores = Enum.map(evidence, &calculate_relevance(&1, claim_words))

    if Enum.empty?(relevance_scores),
      do: 0.0,
      else: Enum.sum(relevance_scores) / length(relevance_scores)
  end

  defp calculate_relevance(evidence, claim_words) do
    evidence_words = extract_words(evidence.content || "")
    overlap = MapSet.intersection(claim_words, evidence_words) |> MapSet.size()
    union = MapSet.union(claim_words, evidence_words) |> MapSet.size()
    if union > 0, do: overlap / union, else: 0.0
  end

  defp check_argument_structure(%SNO{claim: claim, evidence: evidence, children: children}) do
    issues = []

    # Check claim length (too short or too long)
    word_count = claim |> String.split() |> length()

    {length_penalty, length_issues} =
      cond do
        word_count < 3 -> {0.2, ["claim_too_short: claim has only #{word_count} words"]}
        word_count > 100 -> {0.1, ["claim_too_long: claim has #{word_count} words"]}
        true -> {0.0, []}
      end

    issues = issues ++ length_issues

    # Check evidence ratio
    evidence_count = length(evidence)

    {evidence_penalty, evidence_issues} =
      cond do
        evidence_count == 0 ->
          {0.3, ["no_evidence: claim has no evidence"]}

        evidence_count > 10 ->
          {0.1, ["excessive_evidence: #{evidence_count} pieces may indicate over-citation"]}

        true ->
          {0.0, []}
      end

    issues = issues ++ evidence_issues

    # Check hierarchy depth
    max_depth = calculate_max_depth(children, 0)

    {depth_penalty, depth_issues} =
      if max_depth > 5 do
        {0.2, ["excessive_depth: argument has #{max_depth} levels of nesting"]}
      else
        {0.0, []}
      end

    issues = issues ++ depth_issues

    score = max(0.0, 1.0 - length_penalty - evidence_penalty - depth_penalty) |> Float.round(4)
    {score, issues}
  end

  defp calculate_max_depth([], current), do: current

  defp calculate_max_depth(children, current) do
    children
    |> Enum.map(fn child -> calculate_max_depth(child.children, current + 1) end)
    |> Enum.max(fn -> current end)
  end

  defp extract_words(text) do
    text
    |> String.downcase()
    |> String.replace(~r/[^\w\s]/, "")
    |> String.split(~r/\s+/, trim: true)
    |> MapSet.new()
  end

  defp truncate(text, max \\ 50) do
    if String.length(text) > max do
      String.slice(text, 0, max) <> "..."
    else
      text
    end
  end
end

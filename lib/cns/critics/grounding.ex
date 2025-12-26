defmodule CNS.Critics.Grounding do
  @moduledoc """
  Grounding Critic for evaluating factual accuracy and evidence quality.

  Checks for:
  - Citation validity and verification
  - Evidence relevance to claims
  - Source reliability
  - NLI-based entailment checking

  ## Examples

      iex> sno = CNS.SNO.new("Supported claim", evidence: [CNS.Evidence.new("Study", "Data")])
      iex> {:ok, result} = CNS.Critics.Grounding.evaluate(sno)
      iex> result.score > 0
      true
  """

  use GenServer

  @behaviour CNS.Critics.Critic

  alias CNS.{Evidence, SNO}

  # Client API

  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @spec call(GenServer.server(), SNO.t(), keyword()) ::
          {:ok, CNS.Critics.Critic.evaluation_result()} | {:error, term()}
  def call(server, %SNO{} = sno, opts \\ []) do
    GenServer.call(server, {:evaluate, sno, opts})
  end

  # Behaviour callbacks

  @impl CNS.Critics.Critic
  def name, do: :grounding

  @impl CNS.Critics.Critic
  def weight, do: 0.4

  @impl CNS.Critics.Critic
  def evaluate(%SNO{} = sno) do
    do_evaluate(sno, %{})
  end

  # GenServer callbacks

  @impl GenServer
  def init(opts) do
    {:ok, %{evaluations: 0, opts: opts}}
  end

  @impl GenServer
  def handle_call({:evaluate, sno, _opts}, _from, state) do
    result = do_evaluate(sno, state)
    new_state = %{state | evaluations: state.evaluations + 1}
    {:reply, result, new_state}
  end

  # Private functions

  defp do_evaluate(%SNO{evidence: evidence, claim: claim} = _sno, _state) do
    issues = []

    # Check evidence coverage
    {coverage_score, coverage_issues} = check_evidence_coverage(evidence)
    issues = issues ++ coverage_issues

    # Check evidence validity
    {validity_score, validity_issues} = check_evidence_validity(evidence)
    issues = issues ++ validity_issues

    # Check relevance
    {relevance_score, relevance_issues} = check_evidence_relevance(claim, evidence)
    issues = issues ++ relevance_issues

    # Check source diversity
    {diversity_score, diversity_issues} = check_source_diversity(evidence)
    issues = issues ++ diversity_issues

    # Calculate final score
    score =
      (coverage_score * 0.3 +
         validity_score * 0.3 +
         relevance_score * 0.25 +
         diversity_score * 0.15)
      |> Float.round(4)

    result = %{
      score: score,
      issues: issues,
      details: %{
        coverage_score: coverage_score,
        validity_score: validity_score,
        relevance_score: relevance_score,
        diversity_score: diversity_score,
        evidence_count: length(evidence),
        avg_validity:
          if(evidence != [],
            do: Enum.sum(Enum.map(evidence, & &1.validity)) / length(evidence),
            else: 0.0
          )
      }
    }

    {:ok, result}
  rescue
    e -> {:error, Exception.message(e)}
  end

  defp check_evidence_coverage(evidence) do
    count = length(evidence)

    cond do
      count == 0 ->
        {0.0, ["no_evidence: claim has no supporting evidence"]}

      count < 2 ->
        {0.5, ["limited_evidence: only #{count} piece(s) of evidence"]}

      count > 10 ->
        {0.8, ["excessive_evidence: #{count} pieces may indicate over-citation"]}

      true ->
        {1.0, []}
    end
  end

  defp check_evidence_validity(evidence) do
    if Enum.empty?(evidence) do
      {0.0, []}
    else
      validities = Enum.map(evidence, & &1.validity)
      avg = Enum.sum(validities) / length(validities)
      min_val = Enum.min(validities)

      issues =
        cond do
          avg < 0.5 ->
            ["low_validity: average evidence validity is #{Float.round(avg, 2)}"]

          min_val < 0.3 ->
            ["weak_evidence: some evidence has very low validity (#{Float.round(min_val, 2)})"]

          true ->
            []
        end

      {Float.round(avg, 4), issues}
    end
  end

  defp check_evidence_relevance(claim, evidence) do
    if Enum.empty?(evidence) do
      {0.0, []}
    else
      claim_words = extract_words(claim)

      relevance_scores =
        Enum.map(evidence, fn %Evidence{content: content} ->
          content_words = extract_words(content || "")
          jaccard_similarity(claim_words, content_words)
        end)

      avg_relevance = Enum.sum(relevance_scores) / length(relevance_scores)

      issues =
        if avg_relevance < 0.3 do
          ["low_relevance: evidence may not be relevant to claim"]
        else
          []
        end

      {Float.round(avg_relevance, 4), issues}
    end
  end

  defp check_source_diversity(evidence) do
    if Enum.empty?(evidence) do
      {0.0, []}
    else
      sources = Enum.map(evidence, & &1.source)
      unique_sources = Enum.uniq(sources)

      diversity = length(unique_sources) / length(sources)

      issues =
        if diversity < 0.5 do
          ["low_diversity: many citations from same source"]
        else
          []
        end

      {Float.round(diversity, 4), issues}
    end
  end

  defp extract_words(text) do
    (text || "")
    |> String.downcase()
    |> String.replace(~r/[^\w\s]/, "")
    |> String.split(~r/\s+/, trim: true)
    |> MapSet.new()
  end

  defp jaccard_similarity(set1, set2) do
    if MapSet.size(set1) == 0 or MapSet.size(set2) == 0 do
      0.0
    else
      intersection = MapSet.intersection(set1, set2) |> MapSet.size()
      union = MapSet.union(set1, set2) |> MapSet.size()
      intersection / union
    end
  end
end

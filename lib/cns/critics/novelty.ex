defmodule CNS.Critics.Novelty do
  @moduledoc """
  Novelty Critic for evaluating originality and parsimony.

  Checks for:
  - Originality of claims (not just restating evidence)
  - Information density
  - Parsimony (avoiding unnecessary complexity)
  - Non-trivial synthesis

  ## Examples

      iex> sno = CNS.SNO.new("Novel synthesis combining multiple perspectives")
      iex> {:ok, result} = CNS.Critics.Novelty.evaluate(sno)
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
  def name, do: :novelty

  @impl CNS.Critics.Critic
  def weight, do: 0.15

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

  defp do_evaluate(%SNO{claim: claim, evidence: evidence} = sno, _state) do
    issues = []

    # Check originality (not just restating evidence)
    {originality_score, originality_issues} = check_originality(claim, evidence)
    issues = issues ++ originality_issues

    # Check information density
    {density_score, density_issues} = check_information_density(claim)
    issues = issues ++ density_issues

    # Check parsimony
    {parsimony_score, parsimony_issues} = check_parsimony(sno)
    issues = issues ++ parsimony_issues

    # Check for non-trivial content
    {nontrivial_score, nontrivial_issues} = check_nontrivial(claim)
    issues = issues ++ nontrivial_issues

    # Calculate final score
    score =
      (originality_score * 0.35 +
         density_score * 0.25 +
         parsimony_score * 0.2 +
         nontrivial_score * 0.2)
      |> Float.round(4)

    result = %{
      score: score,
      issues: issues,
      details: %{
        originality_score: originality_score,
        density_score: density_score,
        parsimony_score: parsimony_score,
        nontrivial_score: nontrivial_score
      }
    }

    {:ok, result}
  rescue
    e -> {:error, Exception.message(e)}
  end

  defp check_originality(claim, evidence) do
    if Enum.empty?(evidence) do
      # No evidence to copy from
      {0.8, []}
    else
      claim_words = extract_words(claim)

      # Check overlap with evidence
      max_overlap =
        evidence
        |> Enum.map(fn %Evidence{content: content} ->
          ev_words = extract_words(content || "")
          jaccard_similarity(claim_words, ev_words)
        end)
        |> Enum.max(fn -> 0.0 end)

      if max_overlap > 0.7 do
        {1.0 - max_overlap, ["low_originality: claim too similar to evidence"]}
      else
        {min(1.0, 1.0 - max_overlap + 0.3), []}
      end
    end
  end

  defp check_information_density(claim) do
    words = String.split(claim, ~r/\s+/, trim: true)
    word_count = length(words)

    # Remove stop words to get content words
    stop_words =
      ~w(the a an is are was were be been being have has had do does did will would could should may might must shall can and or but if then else when where why how what which who whom whose)

    content_words = Enum.reject(words, &(String.downcase(&1) in stop_words))
    content_ratio = if word_count > 0, do: length(content_words) / word_count, else: 0.0

    cond do
      content_ratio < 0.3 ->
        {content_ratio, ["low_density: too many filler words"]}

      word_count < 5 ->
        {0.3, ["too_short: claim lacks sufficient content"]}

      true ->
        {Float.round(content_ratio, 4), []}
    end
  end

  defp check_parsimony(%SNO{claim: claim, children: children}) do
    word_count = claim |> String.split(~r/\s+/, trim: true) |> length()
    depth = calculate_depth(children, 0)

    complexity_penalty =
      cond do
        word_count > 150 -> 0.3
        word_count > 100 -> 0.1
        true -> 0.0
      end

    depth_penalty = if depth > 3, do: 0.2, else: 0.0

    score = max(0.0, 1.0 - complexity_penalty - depth_penalty)

    issues =
      []
      |> then(fn i ->
        if complexity_penalty > 0,
          do: ["verbose: claim is too long (#{word_count} words)" | i],
          else: i
      end)
      |> then(fn i ->
        if depth_penalty > 0, do: ["deep_nesting: #{depth} levels of hierarchy" | i], else: i
      end)

    {Float.round(score, 4), issues}
  end

  defp check_nontrivial(claim) do
    # Check for trivial patterns
    trivial_patterns = [
      ~r/^(yes|no|true|false|maybe)\.?$/i,
      ~r/^it depends\.?$/i,
      ~r/^this is (good|bad|true|false)\.?$/i
    ]

    is_trivial = Enum.any?(trivial_patterns, &Regex.match?(&1, String.trim(claim)))

    if is_trivial do
      {0.0, ["trivial: claim is too simple or generic"]}
    else
      {1.0, []}
    end
  end

  defp calculate_depth([], current), do: current

  defp calculate_depth(children, current) do
    children
    |> Enum.map(fn child -> calculate_depth(child.children, current + 1) end)
    |> Enum.max(fn -> current end)
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

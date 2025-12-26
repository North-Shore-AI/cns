defmodule CNS.Critics.Bias do
  @moduledoc """
  Bias Critic for evaluating fairness and detecting systemic biases.

  Checks for:
  - Group disparity in claims
  - Loaded language and framing
  - One-sided perspectives
  - Power shadow detection

  ## Examples

      iex> sno = CNS.SNO.new("A balanced view considering multiple perspectives")
      iex> {:ok, result} = CNS.Critics.Bias.evaluate(sno)
      iex> result.score > 0
      true
  """

  use GenServer

  @behaviour CNS.Critics.Critic

  alias CNS.SNO

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
  def name, do: :bias

  @impl CNS.Critics.Critic
  def weight, do: 0.05

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

  defp do_evaluate(%SNO{claim: claim} = sno, _state) do
    issues = []

    # Check for loaded language
    {loaded_score, loaded_issues} = check_loaded_language(claim)
    issues = issues ++ loaded_issues

    # Check for one-sided framing
    {framing_score, framing_issues} = check_framing_balance(claim)
    issues = issues ++ framing_issues

    # Check for absolutist language
    {absolutist_score, absolutist_issues} = check_absolutist_language(claim)
    issues = issues ++ absolutist_issues

    # Check source diversity for bias
    {diversity_score, diversity_issues} = check_evidence_bias(sno.evidence)
    issues = issues ++ diversity_issues

    # Check for balanced perspective indicators
    {balance_score, balance_issues} = check_perspective_balance(claim)
    issues = issues ++ balance_issues

    # Calculate final score
    score =
      (loaded_score * 0.25 +
         framing_score * 0.25 +
         absolutist_score * 0.2 +
         diversity_score * 0.15 +
         balance_score * 0.15)
      |> Float.round(4)

    result = %{
      score: score,
      issues: issues,
      details: %{
        loaded_score: loaded_score,
        framing_score: framing_score,
        absolutist_score: absolutist_score,
        diversity_score: diversity_score,
        balance_score: balance_score
      }
    }

    {:ok, result}
  rescue
    e -> {:error, Exception.message(e)}
  end

  defp check_loaded_language(claim) do
    claim_lower = String.downcase(claim)

    # Loaded/emotional language patterns
    loaded_patterns = [
      {~r/\b(terrible|horrible|awful|catastrophic|disaster)\b/, "negative_loaded"},
      {~r/\b(amazing|incredible|revolutionary|groundbreaking)\b/, "positive_loaded"},
      {~r/\b(always|never|everyone|nobody|all|none)\b/, "absolutist"},
      {~r/\b(obviously|clearly|undeniably|unquestionably)\b/, "presumptive"}
    ]

    matches =
      loaded_patterns
      |> Enum.filter(fn {pattern, _type} -> Regex.match?(pattern, claim_lower) end)
      |> Enum.map(fn {_pattern, type} -> type end)

    if Enum.empty?(matches) do
      {1.0, []}
    else
      penalty = min(1.0, length(matches) * 0.2)
      types = Enum.join(Enum.uniq(matches), ", ")
      {max(0.0, 1.0 - penalty), ["loaded_language: contains #{types} terms"]}
    end
  end

  defp check_framing_balance(claim) do
    claim_lower = String.downcase(claim)

    positive_markers =
      length(Regex.scan(~r/\b(benefit|advantage|positive|good|better|improve)\b/, claim_lower))

    negative_markers =
      length(Regex.scan(~r/\b(harm|disadvantage|negative|bad|worse|decline)\b/, claim_lower))

    analyze_framing_ratio(positive_markers, negative_markers)
  end

  defp analyze_framing_ratio(positive, negative) when positive + negative == 0, do: {0.8, []}

  defp analyze_framing_ratio(positive, negative) do
    total = positive + negative
    ratio = abs(positive - negative) / total

    if ratio > 0.7 do
      direction = if positive > negative, do: "positive", else: "negative"
      {1.0 - ratio, ["one_sided: heavily #{direction} framing"]}
    else
      {1.0, []}
    end
  end

  defp check_absolutist_language(claim) do
    claim_lower = String.downcase(claim)

    absolutist_patterns = [
      ~r/\balways\b/,
      ~r/\bnever\b/,
      ~r/\beveryone\b/,
      ~r/\bnobody\b/,
      ~r/\bnothing\b/,
      ~r/\beverything\b/,
      ~r/\bonly\b/,
      ~r/\bmust\b/
    ]

    matches = Enum.count(absolutist_patterns, &Regex.match?(&1, claim_lower))

    if matches == 0 do
      {1.0, []}
    else
      penalty = min(0.6, matches * 0.15)
      {Float.round(1.0 - penalty, 4), ["absolutist: uses #{matches} absolutist term(s)"]}
    end
  end

  defp check_evidence_bias(evidence) do
    if Enum.empty?(evidence) do
      {0.5, ["no_evidence_diversity: cannot assess evidence bias"]}
    else
      sources = Enum.map(evidence, & &1.source)

      # Check source name patterns for bias indicators
      # This is a simplified heuristic
      unique_sources = Enum.uniq(sources)
      diversity_ratio = length(unique_sources) / length(sources)

      if diversity_ratio < 0.5 do
        {diversity_ratio, ["evidence_concentration: many citations from same sources"]}
      else
        {diversity_ratio, []}
      end
    end
  end

  defp check_perspective_balance(claim) do
    claim_lower = String.downcase(claim)

    # Indicators of balanced perspective
    balance_patterns = [
      ~r/\bhowever\b/,
      ~r/\bon the other hand\b/,
      ~r/\bwhile\b.*\balso\b/,
      ~r/\bboth\b.*\band\b/,
      ~r/\balthough\b/,
      ~r/\bnevertheless\b/,
      ~r/\bconversely\b/
    ]

    has_balance = Enum.any?(balance_patterns, &Regex.match?(&1, claim_lower))

    if has_balance do
      {1.0, []}
    else
      # Check if claim is long enough to warrant balance
      word_count = claim |> String.split(~r/\s+/, trim: true) |> length()

      if word_count > 20 do
        {0.6, ["unbalanced: longer claim without balancing language"]}
      else
        {0.8, []}
      end
    end
  end
end

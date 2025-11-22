defmodule CNS.Critics.Causal do
  @moduledoc """
  Causal Critic for evaluating causal validity of claims.

  Checks for:
  - Correlation vs causation confusion
  - Proper causal language use
  - Temporal ordering
  - Confounding factors

  ## Examples

      iex> sno = CNS.SNO.new("X causes Y due to mechanism Z")
      iex> {:ok, result} = CNS.Critics.Causal.evaluate(sno)
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
  def name, do: :causal

  @impl CNS.Critics.Critic
  def weight, do: 0.1

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

    # Check for causal language
    {causal_type, type_issues} = classify_causal_type(claim)
    issues = issues ++ type_issues

    # Check for correlation-causation confusion
    {confusion_score, confusion_issues} = check_correlation_confusion(claim)
    issues = issues ++ confusion_issues

    # Check for mechanism explanation
    {mechanism_score, mechanism_issues} = check_mechanism(claim)
    issues = issues ++ mechanism_issues

    # Check temporal ordering
    {temporal_score, temporal_issues} = check_temporal_ordering(claim)
    issues = issues ++ temporal_issues

    # Check confidence in causal claims
    {confidence_score, confidence_issues} = check_causal_confidence(claim, sno.confidence)
    issues = issues ++ confidence_issues

    # Calculate final score
    score =
      (confusion_score * 0.3 +
         mechanism_score * 0.25 +
         temporal_score * 0.2 +
         confidence_score * 0.25)
      |> Float.round(4)

    result = %{
      score: score,
      issues: issues,
      details: %{
        causal_type: causal_type,
        confusion_score: confusion_score,
        mechanism_score: mechanism_score,
        temporal_score: temporal_score,
        confidence_score: confidence_score
      }
    }

    {:ok, result}
  rescue
    e -> {:error, Exception.message(e)}
  end

  defp classify_causal_type(claim) do
    claim_lower = String.downcase(claim)

    cond do
      has_causal_language?(claim_lower) ->
        {:causal, []}

      has_correlational_language?(claim_lower) ->
        {:correlational, []}

      true ->
        {:descriptive, []}
    end
  end

  defp has_causal_language?(text) do
    causal_patterns = [
      ~r/\bcauses?\b/,
      ~r/\bresults? in\b/,
      ~r/\bleads? to\b/,
      ~r/\bproduces?\b/,
      ~r/\bgenerates?\b/,
      ~r/\bbecause\b/,
      ~r/\bdue to\b/,
      ~r/\btherefore\b/,
      ~r/\bconsequently\b/,
      ~r/\beffect of\b/,
      ~r/\bimpact of\b/
    ]

    Enum.any?(causal_patterns, &Regex.match?(&1, text))
  end

  defp has_correlational_language?(text) do
    correlational_patterns = [
      ~r/\bcorrelat(ed?|ion)\b/,
      ~r/\bassociat(ed?|ion)\b/,
      ~r/\brelat(ed?|ionship)\b/,
      ~r/\blink(ed)?\b/,
      ~r/\bconnect(ed|ion)\b/
    ]

    Enum.any?(correlational_patterns, &Regex.match?(&1, text))
  end

  defp check_correlation_confusion(claim) do
    claim_lower = String.downcase(claim)

    # Red flags: correlational evidence with causal conclusions
    has_correlation = has_correlational_language?(claim_lower)
    has_causation = has_causal_language?(claim_lower)

    # Check for hedging language
    has_hedging = Regex.match?(~r/\b(may|might|could|possibly|suggests?)\b/, claim_lower)

    cond do
      has_correlation and has_causation and not has_hedging ->
        {0.3, ["correlation_causation: may confuse correlation with causation"]}

      has_causation and not has_hedging ->
        {0.7, ["strong_causal: causal claim without hedging language"]}

      true ->
        {1.0, []}
    end
  end

  defp check_mechanism(claim) do
    claim_lower = String.downcase(claim)

    # Check for mechanism explanation patterns
    mechanism_patterns = [
      ~r/\bthrough\b/,
      ~r/\bvia\b/,
      ~r/\bby means of\b/,
      ~r/\bmechanism\b/,
      ~r/\bprocess\b/,
      ~r/\bpathway\b/
    ]

    has_mechanism = Enum.any?(mechanism_patterns, &Regex.match?(&1, claim_lower))
    has_causation = has_causal_language?(claim_lower)

    cond do
      has_causation and has_mechanism ->
        {1.0, []}

      has_causation and not has_mechanism ->
        {0.5, ["no_mechanism: causal claim without mechanism explanation"]}

      true ->
        {0.8, []}
    end
  end

  defp check_temporal_ordering(claim) do
    claim_lower = String.downcase(claim)

    # Check for temporal language
    temporal_patterns = [
      ~r/\bbefore\b/,
      ~r/\bafter\b/,
      ~r/\bfirst\b.*\bthen\b/,
      ~r/\bsubsequently\b/,
      ~r/\bprecedes?\b/,
      ~r/\bfollows?\b/
    ]

    has_temporal = Enum.any?(temporal_patterns, &Regex.match?(&1, claim_lower))
    has_causation = has_causal_language?(claim_lower)

    cond do
      has_causation and has_temporal ->
        {1.0, []}

      has_causation and not has_temporal ->
        {0.6, ["no_temporal: causal claim without temporal ordering"]}

      true ->
        {0.8, []}
    end
  end

  defp check_causal_confidence(claim, confidence) do
    has_causation = has_causal_language?(String.downcase(claim))

    cond do
      has_causation and confidence > 0.9 ->
        {0.5, ["overconfident_causation: high confidence (#{confidence}) on causal claim"]}

      has_causation and confidence < 0.5 ->
        {0.7, ["uncertain_causation: low confidence (#{confidence}) undermines causal claim"]}

      true ->
        {1.0, []}
    end
  end
end

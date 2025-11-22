# Synthesis Engine Design

## Version
- Date: 2025-11-21
- Status: Design Document
- Author: Architecture Team

## Overview

The synthesis engine generates dialectical reconciliations using LLM-based generation with constrained decoding to ensure evidence preservation and logical coherence.

## Module Structure

```
lib/cns/synthesis/
  engine.ex                   # Main synthesis GenServer pool
  constraints.ex              # Constrained decoding rules
  evidence_linker.ex          # Evidence attribution
  iterative.ex                # Iterative refinement
  templates.ex                # Prompt templates
```

## Engine Architecture

```elixir
defmodule CNS.Synthesis.Engine do
  @moduledoc """
  Main synthesis engine that generates dialectical reconciliations.

  Uses Tinkex for LLM generation with constrained decoding to ensure
  evidence preservation and logical coherence.
  """

  use GenServer

  alias CNS.Synthesis.{Constraints, EvidenceLinker, Templates}
  alias Crucible.Tinkex

  defstruct [
    :config,
    :sampling_client,
    :constraint_checker,
    :max_iterations
  ]

  @default_config %{
    temperature: 0.7,
    top_p: 0.95,
    max_tokens: 1024,
    max_iterations: 5,
    min_critic_score: 0.6
  }

  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts)
  end

  @doc """
  Generates a synthesis for the given SNO.
  """
  def synthesize(engine, sno, opts \\ []) do
    GenServer.call(engine, {:synthesize, sno, opts}, 60_000)
  end

  @impl true
  def init(opts) do
    config = Map.merge(@default_config, Keyword.get(opts, :config, %{}))

    {:ok, %__MODULE__{
      config: config,
      max_iterations: config.max_iterations
    }}
  end

  @impl true
  def handle_call({:synthesize, sno, opts}, _from, state) do
    result = perform_synthesis(sno, opts, state)
    {:reply, result, state}
  end

  defp perform_synthesis(sno, opts, state) do
    strategy = Keyword.get(opts, :strategy, :iterative)

    case strategy do
      :single_shot ->
        single_shot_synthesis(sno, opts, state)

      :iterative ->
        iterative_synthesis(sno, opts, state)

      :ensemble ->
        ensemble_synthesis(sno, opts, state)
    end
  end

  defp single_shot_synthesis(sno, opts, state) do
    # Generate prompt
    prompt = Templates.synthesis_prompt(sno, opts)

    # Build constraints
    constraints = Constraints.build(sno, opts)

    # Generate with constraints
    sampling_params = build_sampling_params(state.config, opts)

    case generate_with_constraints(prompt, constraints, sampling_params, opts) do
      {:ok, text} ->
        # Link evidence citations
        {:ok, synthesis, citations} = EvidenceLinker.link(text, sno.evidence)

        # Update SNO
        new_sno = CNS.SNO.set_synthesis(sno, synthesis, citations: citations)

        {:ok, new_sno}

      {:error, _} = error ->
        error
    end
  end

  defp iterative_synthesis(sno, opts, state) do
    max_iterations = Keyword.get(opts, :max_iterations, state.max_iterations)
    min_score = Keyword.get(opts, :min_critic_score, state.config.min_critic_score)

    # Initial synthesis
    {:ok, sno} = single_shot_synthesis(sno, opts, state)

    # Iterative refinement
    refine_until_acceptable(sno, 1, max_iterations, min_score, opts, state)
  end

  defp refine_until_acceptable(sno, iteration, max_iterations, min_score, opts, state) do
    # Evaluate with critics
    {:ok, eval} = CNS.Critics.Pipeline.evaluate_all(sno)

    if eval.passed and eval.overall_score >= min_score do
      {:ok, sno}
    else
      if iteration >= max_iterations do
        # Return best effort
        {:ok, sno}
      else
        # Generate refinement prompt
        prompt = Templates.refinement_prompt(sno, eval, opts)
        constraints = Constraints.build(sno, opts)
        sampling_params = build_sampling_params(state.config, opts)

        case generate_with_constraints(prompt, constraints, sampling_params, opts) do
          {:ok, text} ->
            {:ok, synthesis, citations} = EvidenceLinker.link(text, sno.evidence)
            new_sno = CNS.SNO.set_synthesis(sno, synthesis, citations: citations)

            refine_until_acceptable(new_sno, iteration + 1, max_iterations, min_score, opts, state)

          {:error, _} ->
            {:ok, sno}  # Keep previous synthesis
        end
      end
    end
  end

  defp ensemble_synthesis(sno, opts, state) do
    # Use Crucible ensemble for synthesis
    pool = Keyword.fetch!(opts, :ensemble_pool)

    {:ok, result} = Crucible.Ensemble.Synthesis.synthesize(
      pool,
      sno.thesis,
      sno.antithesis,
      Map.values(sno.evidence),
      Keyword.merge(opts, [
        blend_strategy: Keyword.get(opts, :blend_strategy, :best_of_n),
        n: Keyword.get(opts, :n, 3)
      ])
    )

    {:ok, synthesis, citations} = EvidenceLinker.link(result, sno.evidence)
    new_sno = CNS.SNO.set_synthesis(sno, synthesis, citations: citations)

    {:ok, new_sno}
  end

  defp generate_with_constraints(prompt, constraints, sampling_params, opts) do
    # Get sampling client
    client = Keyword.fetch!(opts, :sampling_client)

    # Apply constraint-aware generation
    case Tinkex.SamplingClient.generate(client, prompt, sampling_params) do
      {:ok, response} ->
        # Post-process to enforce constraints
        text = extract_text(response)
        validated = Constraints.enforce(text, constraints)
        {:ok, validated}

      {:error, _} = error ->
        error
    end
  end

  defp build_sampling_params(config, opts) do
    Tinkex.sampling_params(
      temperature: Keyword.get(opts, :temperature, config.temperature),
      top_p: Keyword.get(opts, :top_p, config.top_p),
      max_tokens: Keyword.get(opts, :max_tokens, config.max_tokens),
      stop_sequences: Keyword.get(opts, :stop_sequences, [])
    )
  end

  defp extract_text(%{text: text}), do: text
  defp extract_text(response) when is_binary(response), do: response
end
```

## Constrained Decoding

```elixir
defmodule CNS.Synthesis.Constraints do
  @moduledoc """
  Defines and enforces constraints for synthesis generation.
  """

  @type constraint :: %{
    type: constraint_type(),
    config: map()
  }

  @type constraint_type ::
    :must_cite |
    :must_mention |
    :banned_phrases |
    :max_length |
    :structure |
    :evidence_ratio

  @doc """
  Builds constraints for synthesis generation.
  """
  @spec build(CNS.SNO.t(), keyword()) :: [constraint()]
  def build(sno, opts \\ []) do
    base_constraints = [
      # Must cite at least some evidence
      %{
        type: :evidence_ratio,
        config: %{
          min_ratio: Keyword.get(opts, :min_evidence_ratio, 0.5),
          evidence_ids: Map.keys(sno.evidence)
        }
      },

      # Must mention key terms from both sides
      %{
        type: :must_mention,
        config: %{
          from_thesis: extract_key_terms(sno.thesis),
          from_antithesis: extract_key_terms(sno.antithesis),
          min_each: 1
        }
      },

      # Structural requirements
      %{
        type: :structure,
        config: %{
          require_sections: Keyword.get(opts, :require_sections, false),
          min_sentences: 3,
          max_sentences: 10
        }
      }
    ]

    # Add custom constraints
    custom = Keyword.get(opts, :constraints, [])
    base_constraints ++ custom
  end

  @doc """
  Enforces constraints on generated text.
  """
  @spec enforce(String.t(), [constraint()]) :: String.t()
  def enforce(text, constraints) do
    Enum.reduce(constraints, text, fn constraint, acc ->
      enforce_constraint(acc, constraint)
    end)
  end

  @doc """
  Validates that text satisfies all constraints.
  """
  @spec validate(String.t(), [constraint()]) :: {:ok, []} | {:error, [String.t()]}
  def validate(text, constraints) do
    violations = Enum.flat_map(constraints, fn constraint ->
      check_constraint(text, constraint)
    end)

    if violations == [] do
      {:ok, []}
    else
      {:error, violations}
    end
  end

  defp enforce_constraint(text, %{type: :must_cite, config: config}) do
    if missing_citations?(text, config.evidence_ids) do
      # Add missing citations (placeholder - real implementation would be smarter)
      text
    else
      text
    end
  end

  defp enforce_constraint(text, %{type: :banned_phrases, config: config}) do
    Enum.reduce(config.phrases, text, fn phrase, acc ->
      String.replace(acc, phrase, config[:replacement] || "")
    end)
  end

  defp enforce_constraint(text, %{type: :max_length, config: config}) do
    if String.length(text) > config.max_chars do
      truncate_to_sentence(text, config.max_chars)
    else
      text
    end
  end

  defp enforce_constraint(text, _), do: text

  defp check_constraint(text, %{type: :evidence_ratio, config: config}) do
    cited = extract_citations(text)
    ratio = length(cited) / length(config.evidence_ids)

    if ratio < config.min_ratio do
      ["Evidence ratio #{Float.round(ratio, 2)} below minimum #{config.min_ratio}"]
    else
      []
    end
  end

  defp check_constraint(text, %{type: :must_mention, config: config}) do
    text_lower = String.downcase(text)

    thesis_mentions = Enum.count(config.from_thesis, fn term ->
      String.contains?(text_lower, String.downcase(term))
    end)

    antithesis_mentions = Enum.count(config.from_antithesis, fn term ->
      String.contains?(text_lower, String.downcase(term))
    end)

    violations = []
    violations = if thesis_mentions < config.min_each do
      ["Must mention at least #{config.min_each} thesis terms" | violations]
    else
      violations
    end

    if antithesis_mentions < config.min_each do
      ["Must mention at least #{config.min_each} antithesis terms" | violations]
    else
      violations
    end
  end

  defp check_constraint(text, %{type: :structure, config: config}) do
    sentences = String.split(text, ~r/[.!?]/) |> Enum.filter(&(String.trim(&1) != ""))

    violations = []
    violations = if length(sentences) < config.min_sentences do
      ["Too few sentences (#{length(sentences)} < #{config.min_sentences})" | violations]
    else
      violations
    end

    if length(sentences) > config.max_sentences do
      ["Too many sentences (#{length(sentences)} > #{config.max_sentences})" | violations]
    else
      violations
    end
  end

  defp check_constraint(_, _), do: []

  defp extract_key_terms(text) do
    # Simple extraction - would use NLP in practice
    text
    |> String.downcase()
    |> String.split(~r/\W+/)
    |> Enum.filter(&(String.length(&1) > 4))
    |> Enum.uniq()
    |> Enum.take(5)
  end

  defp extract_citations(text) do
    Regex.scan(~r/\[([eE]\d+)\]/, text)
    |> Enum.map(fn [_, id] -> String.downcase(id) end)
    |> Enum.uniq()
  end

  defp missing_citations?(text, evidence_ids) do
    cited = extract_citations(text)
    length(cited) < length(evidence_ids) * 0.5
  end

  defp truncate_to_sentence(text, max_chars) do
    if String.length(text) <= max_chars do
      text
    else
      # Find last sentence break before max_chars
      truncated = String.slice(text, 0, max_chars)
      case Regex.run(~r/.*[.!?]/, truncated) do
        [match] -> match
        nil -> truncated <> "..."
      end
    end
  end
end
```

## Evidence Linker

```elixir
defmodule CNS.Synthesis.EvidenceLinker do
  @moduledoc """
  Links generated text to evidence through citation insertion.
  """

  @doc """
  Links evidence citations in generated text.

  Returns the text with citations and a list of cited evidence IDs.
  """
  @spec link(String.t(), map()) :: {:ok, String.t(), [String.t()]}
  def link(text, evidence_map) do
    # 1. Find existing citations
    existing = extract_existing_citations(text)

    # 2. Find claims that need citations
    uncited_claims = find_uncited_claims(text, existing)

    # 3. Match claims to evidence
    matches = match_claims_to_evidence(uncited_claims, evidence_map)

    # 4. Insert citations
    linked_text = insert_citations(text, matches)

    # 5. Collect all citations
    all_citations = (existing ++ Enum.map(matches, & &1.evidence_id))
    |> Enum.uniq()

    {:ok, linked_text, all_citations}
  end

  defp extract_existing_citations(text) do
    Regex.scan(~r/\[([eE]\d+)\]/, text)
    |> Enum.map(fn [_, id] -> String.downcase(id) end)
    |> Enum.uniq()
  end

  defp find_uncited_claims(text, existing_citations) do
    # Split into sentences
    sentences = String.split(text, ~r/(?<=[.!?])\s+/)

    # Find sentences without citations that make claims
    Enum.with_index(sentences)
    |> Enum.filter(fn {sentence, _idx} ->
      no_citation = not Regex.match?(~r/\[[eE]\d+\]/, sentence)
      makes_claim = makes_factual_claim?(sentence)
      no_citation and makes_claim
    end)
    |> Enum.map(fn {sentence, idx} -> %{sentence: sentence, index: idx} end)
  end

  defp makes_factual_claim?(sentence) do
    # Check for factual claim indicators
    indicators = [
      ~r/\b(is|are|was|were|has|have|shows?|demonstrates?|indicates?|proves?)\b/i,
      ~r/\b(studies?|research|evidence|data)\b/i,
      ~r/\b\d+%|\d+\s*(patients?|subjects?|participants?)/i
    ]

    Enum.any?(indicators, &Regex.match?(&1, sentence))
  end

  defp match_claims_to_evidence(uncited_claims, evidence_map) do
    Enum.flat_map(uncited_claims, fn claim ->
      # Find best matching evidence
      best_match = evidence_map
      |> Enum.map(fn {id, ev} ->
        score = compute_relevance(claim.sentence, ev.content)
        {id, score}
      end)
      |> Enum.filter(fn {_, score} -> score > 0.5 end)
      |> Enum.max_by(fn {_, score} -> score end, fn -> nil end)

      case best_match do
        nil -> []
        {evidence_id, _score} ->
          [%{
            sentence_index: claim.index,
            sentence: claim.sentence,
            evidence_id: evidence_id
          }]
      end
    end)
  end

  defp compute_relevance(claim, evidence) do
    # Simple word overlap - would use embeddings in practice
    claim_words = tokenize(claim)
    evidence_words = tokenize(evidence)

    overlap = MapSet.intersection(
      MapSet.new(claim_words),
      MapSet.new(evidence_words)
    )

    if length(claim_words) == 0 do
      0.0
    else
      MapSet.size(overlap) / length(claim_words)
    end
  end

  defp tokenize(text) do
    text
    |> String.downcase()
    |> String.split(~r/\W+/)
    |> Enum.filter(&(String.length(&1) > 3))
  end

  defp insert_citations(text, matches) do
    sentences = String.split(text, ~r/(?<=[.!?])\s+/)

    updated_sentences = sentences
    |> Enum.with_index()
    |> Enum.map(fn {sentence, idx} ->
      case Enum.find(matches, fn m -> m.sentence_index == idx end) do
        nil -> sentence
        match ->
          # Insert citation before sentence-ending punctuation
          citation = " [#{match.evidence_id}]"
          Regex.replace(~r/([.!?])$/, sentence, "#{citation}\\1")
      end
    end)

    Enum.join(updated_sentences, " ")
  end
end
```

## Prompt Templates

```elixir
defmodule CNS.Synthesis.Templates do
  @moduledoc """
  Prompt templates for synthesis generation.
  """

  @doc """
  Main synthesis prompt.
  """
  def synthesis_prompt(sno, opts \\ []) do
    style = Keyword.get(opts, :style, :balanced)

    """
    #{preamble(style)}

    ## Input

    ### Thesis
    #{sno.thesis}

    ### Antithesis
    #{sno.antithesis}

    ### Evidence
    #{format_evidence(sno.evidence)}

    ## Instructions

    Generate a synthesis that:

    1. **Acknowledges both perspectives**: Begin by recognizing valid points from both thesis and antithesis.

    2. **Cites evidence**: Support claims with evidence citations using [E1], [E2], etc.

    3. **Reconciles the conflict**: Provide a higher-level perspective that explains how both views can be partially correct.

    4. **Maintains logical coherence**: Ensure no contradictions between stated claims.

    5. **Adds insight**: Go beyond merely restating the inputs - provide genuine reconciliation.

    #{style_instructions(style)}

    ## Synthesis
    """
  end

  @doc """
  Refinement prompt for iterative improvement.
  """
  def refinement_prompt(sno, eval, _opts \\ []) do
    """
    Your previous synthesis received the following evaluation:

    Score: #{Float.round(eval.overall_score, 2)}

    Issues found:
    #{format_issues(eval.issues)}

    Suggestions:
    #{format_suggestions(eval)}

    ## Original Synthesis
    #{sno.synthesis}

    ## Evidence Available
    #{format_evidence(sno.evidence)}

    ## Instructions

    Please revise the synthesis to address the identified issues:
    #{specific_revision_instructions(eval.issues)}

    ## Revised Synthesis
    """
  end

  defp preamble(:balanced) do
    """
    You are a synthesis engine that creates balanced dialectical reconciliations.
    Your goal is to find truth in opposing viewpoints and create a higher-level understanding.
    """
  end

  defp preamble(:scholarly) do
    """
    You are a scholarly synthesis engine that creates rigorous dialectical reconciliations.
    Your goal is to produce publication-quality analysis with proper evidence attribution.
    """
  end

  defp preamble(:accessible) do
    """
    You are a synthesis engine that creates clear, accessible reconciliations.
    Your goal is to make complex opposing viewpoints understandable to general readers.
    """
  end

  defp style_instructions(:balanced), do: "Write in a balanced, objective tone."
  defp style_instructions(:scholarly), do: "Write in formal academic style with detailed analysis."
  defp style_instructions(:accessible), do: "Write clearly for a general audience, avoiding jargon."

  defp format_evidence(evidence_map) do
    evidence_map
    |> Enum.sort_by(fn {id, _} -> id end)
    |> Enum.map(fn {id, ev} ->
      """
      [#{String.upcase(id)}] #{ev.content}
      Source: #{ev.source}
      """
    end)
    |> Enum.join("\n")
  end

  defp format_issues(issues) do
    issues
    |> Enum.map(fn issue ->
      "- [#{issue.severity}] #{issue.description}"
    end)
    |> Enum.join("\n")
  end

  defp format_suggestions(eval) do
    eval.individual_scores
    |> Enum.filter(fn {_, score} -> score < 0.7 end)
    |> Enum.map(fn {critic, score} ->
      "- Improve #{critic} (current: #{Float.round(score, 2)})"
    end)
    |> Enum.join("\n")
  end

  defp specific_revision_instructions(issues) do
    issue_types = Enum.map(issues, & &1.type) |> Enum.uniq()

    instructions = []

    instructions = if :invalid_citation in issue_types do
      ["- Remove invalid citations and only cite from the evidence pool" | instructions]
    else
      instructions
    end

    instructions = if :low_novelty in issue_types do
      ["- Add more original insight beyond restating the sources" | instructions]
    else
      instructions
    end

    instructions = if :imbalanced in issue_types do
      ["- Give more equal weight to both thesis and antithesis" | instructions]
    else
      instructions
    end

    instructions = if :weak_support in issue_types do
      ["- Ensure citations strongly support their associated claims" | instructions]
    else
      instructions
    end

    Enum.join(instructions, "\n")
  end
end
```

## Engine Pool

```elixir
defmodule CNS.Synthesis.EnginePool do
  @moduledoc """
  Pool of synthesis engines for concurrent processing.
  """

  use Supervisor

  def start_link(opts) do
    Supervisor.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @impl true
  def init(opts) do
    pool_size = Keyword.get(opts, :pool_size, 4)

    children = for i <- 1..pool_size do
      Supervisor.child_spec(
        {CNS.Synthesis.Engine, Keyword.put(opts, :id, i)},
        id: {CNS.Synthesis.Engine, i}
      )
    end

    Supervisor.init(children, strategy: :one_for_one)
  end

  @doc """
  Synthesizes using an available engine from the pool.
  """
  def synthesize(sno, opts \\ []) do
    # Get available engine (simple round-robin)
    engine = get_engine()
    CNS.Synthesis.Engine.synthesize(engine, sno, opts)
  end

  defp get_engine do
    children = Supervisor.which_children(__MODULE__)
    {_, pid, _, _} = Enum.random(children)
    pid
  end
end
```

## Usage Example

```elixir
# Create SNO
sno = CNS.SNO.new(
  thesis: "Remote work improves productivity",
  antithesis: "Remote work reduces collaboration and innovation",
  evidence: [
    %{id: "e1", content: "Survey of 10,000 workers found 67% report higher productivity at home", source: "Stanford Study"},
    %{id: "e2", content: "Patent filings decreased 8% at fully remote companies", source: "NBER"},
    %{id: "e3", content: "Hybrid workers report better work-life balance and similar collaboration", source: "McKinsey"}
  ]
)

# Configure sampling client
opts = [
  sampling_client: sampling_pid,
  strategy: :iterative,
  max_iterations: 3,
  min_critic_score: 0.7,
  style: :balanced
]

# Generate synthesis
{:ok, sno_with_synthesis} = CNS.Synthesis.EnginePool.synthesize(sno, opts)

IO.puts(sno_with_synthesis.synthesis)
# While remote work demonstrably increases individual productivity [E1],
# it can negatively impact collaborative innovation [E2]. The optimal
# approach appears to be hybrid arrangements that balance focused work
# with in-person collaboration [E3], allowing organizations to capture
# productivity gains while maintaining innovation capacity.
```

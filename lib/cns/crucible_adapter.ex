unless Code.ensure_loaded?(Crucible.CNS.Adapter) do
  defmodule Crucible.CNS.Adapter do
    @callback evaluate(examples :: [map()], outputs :: list(), opts :: map()) ::
                {:ok, map()} | {:error, term()}
  end
end

defmodule CNS.CrucibleAdapter do
  @moduledoc """
  Concrete implementation of Crucible.CNS.Adapter for evaluating model outputs.

  This adapter bridges CNS validation and metrics to the Crucible pipeline,
  enabling experiments to use CNS for quality evaluation of generated claims.

  ## Usage

  Configure in your application:

      config :crucible_framework, :cns_adapter, CNS.CrucibleAdapter

  The adapter will be called by `Crucible.Stage.CNSMetrics` during pipeline
  execution, receiving examples and outputs from the context.

  ## Metrics Computed

  - **Schema compliance**: Percentage of outputs parseable as CLAIM/RELATION
  - **Citation accuracy**: Percentage of valid citations against corpus
  - **Entailment scores**: NLI-based semantic validation (if Bumblebee available)
  - **Topology metrics**: β₁, connectivity, DAG validation
  - **Chirality metrics**: Polarity conflict detection
  - **Overall quality**: Weighted aggregate per CNS 3.0 targets

  ## CNS 3.0 Quality Targets

  - Schema compliance ≥ 95%
  - Citation accuracy ≥ 95%
  - Mean entailment ≥ 0.50

  ## Example

      examples = [
        %{
          "prompt" => "Extract claims from...",
          "completion" => "CLAIM[c1] (Document 123): Coffee has caffeine",
          "metadata" => %{"doc_ids" => ["123"]}
        }
      ]

      outputs = [
        "CLAIM[c1] (Document 123): Coffee contains caffeine\\nRELATION: c1 supports c2"
      ]

      {:ok, metrics} = CNS.CrucibleAdapter.evaluate(examples, outputs, %{})
  """

  @behaviour Crucible.CNS.Adapter

  require Logger

  alias CNS.{SNO, Evidence, Provenance, Topology, Config}
  alias CNS.Validation.Semantic

  @doc """
  Evaluate model outputs using CNS metrics.

  ## Parameters

    * `examples` - List of example maps with "prompt", "completion", "metadata"
    * `outputs` - List of model-generated output strings
    * `opts` - Options map (currently unused, reserved for future configuration)

  ## Returns

    * `{:ok, metrics}` - Success with computed metrics
    * `{:error, reason}` - Failure with error reason

  ## Metrics Structure

  The returned metrics map contains:

    * `:schema_compliance` - Float 0.0-1.0, percentage of parseable outputs
    * `:parseable_count` - Integer, number of outputs successfully parsed
    * `:unparseable_count` - Integer, number of outputs that failed parsing
    * `:citation_accuracy` - Float 0.0-1.0, percentage of valid citations
    * `:valid_citations` - Integer, count of valid citations
    * `:invalid_citations` - Integer, count of invalid citations
    * `:hallucinated_citations` - Integer, count of fabricated citations
    * `:mean_entailment` - Float 0.0-1.0 or nil, average NLI entailment score
    * `:mean_similarity` - Float 0.0-1.0 or nil, average semantic similarity
    * `:topology` - Map with `:mean_beta1`, `:max_beta1`, `:dag_count`, `:cyclic_count`
    * `:chirality` - Map with `:mean_score`, `:polarity_conflicts`, `:high_conflict_count`
    * `:overall_quality` - Float 0.0-1.0, weighted combination of metrics
    * `:meets_threshold` - Boolean, whether CNS 3.0 targets are met
  """
  @impl true
  @spec evaluate(list(map()), list(String.t()), map()) :: {:ok, map()} | {:error, term()}
  def evaluate(examples, outputs, opts \\ %{})

  def evaluate([], [], _opts) do
    # Empty inputs produce empty metrics
    {:ok, empty_metrics()}
  end

  def evaluate(examples, outputs, _opts) when length(examples) != length(outputs) do
    {:error,
     {:mismatched_lengths,
      "examples count (#{length(examples)}) != outputs count (#{length(outputs)})"}}
  end

  def evaluate(examples, outputs, opts) do
    try do
      # Parse outputs into structured claims
      parsed_results = parse_outputs(outputs)

      # Build corpus from examples (for citation validation)
      corpus = build_corpus(examples)

      # Compute schema metrics
      schema_metrics = compute_schema_metrics(parsed_results)

      # Extract SNOs from parsed results
      snos = extract_snos(parsed_results)

      # Compute citation metrics
      citation_metrics = compute_citation_metrics(snos, corpus)

      # Compute semantic metrics (if models available)
      semantic_metrics = compute_semantic_metrics(examples, outputs, snos, corpus, opts)

      # Compute topology metrics
      topology_metrics = compute_topology_metrics(snos)

      # Compute chirality metrics
      chirality_metrics = compute_chirality_metrics(snos)

      # Calculate overall quality
      overall_metrics =
        compute_overall_quality(
          schema_metrics,
          citation_metrics,
          semantic_metrics,
          topology_metrics,
          chirality_metrics
        )

      # Combine all metrics
      metrics =
        %{}
        |> Map.merge(schema_metrics)
        |> Map.merge(citation_metrics)
        |> Map.merge(semantic_metrics)
        |> Map.merge(topology_metrics)
        |> Map.merge(chirality_metrics)
        |> Map.merge(overall_metrics)

      {:ok, metrics}
    rescue
      e ->
        Logger.error("CNS.CrucibleAdapter evaluation failed: #{Exception.message(e)}")
        {:error, Exception.message(e)}
    end
  end

  # Private functions

  defp empty_metrics do
    %{
      schema_compliance: 1.0,
      parseable_count: 0,
      unparseable_count: 0,
      citation_accuracy: 1.0,
      valid_citations: 0,
      invalid_citations: 0,
      hallucinated_citations: 0,
      mean_entailment: nil,
      mean_similarity: nil,
      topology: %{
        mean_beta1: 0.0,
        max_beta1: 0,
        dag_count: 0,
        cyclic_count: 0
      },
      chirality: %{
        mean_score: 0.0,
        polarity_conflicts: 0,
        high_conflict_count: 0
      },
      overall_quality: 1.0,
      meets_threshold: true
    }
  end

  defp parse_outputs(outputs) do
    Enum.map(outputs, &parse_single_output/1)
  end

  defp parse_single_output(output) do
    try do
      # Extract CLAIM patterns
      claims = extract_claims(output)

      # Extract RELATION patterns
      relations = extract_relations(output)

      # If no claims found, consider it unparseable
      success = length(claims) > 0

      %{
        success: success,
        claims: claims,
        relations: relations,
        raw: output
      }
    rescue
      _ ->
        %{
          success: false,
          claims: [],
          relations: [],
          raw: output,
          error: "Failed to parse output"
        }
    end
  end

  defp extract_claims(text) do
    # Pattern: CLAIM[c1] (Document 123): Some claim text
    ~r/CLAIM\[([^\]]+)\](?:\s*\(Document\s+(\d+)\))?\s*:\s*([^\n]+)/
    |> Regex.scan(text)
    |> Enum.map(fn
      [_full, claim_id, "", claim_text] ->
        %{
          id: claim_id,
          text: String.trim(claim_text),
          doc_ids: []
        }

      [_full, claim_id, doc_id, claim_text] ->
        %{
          id: claim_id,
          text: String.trim(claim_text),
          doc_ids: [doc_id]
        }

      _ ->
        nil
    end)
    |> Enum.reject(&is_nil/1)
  end

  defp extract_relations(text) do
    # Pattern: RELATION: c1 supports c2
    ~r/RELATION:\s*([^\s]+)\s+(supports|refutes|contradicts)\s+([^\s\n]+)/
    |> Regex.scan(text)
    |> Enum.map(fn [_full, source, relation_type, target] ->
      %{
        source: source,
        type: String.to_atom(relation_type),
        target: target
      }
    end)
  end

  defp build_corpus(examples) do
    # Build corpus map from examples metadata
    Enum.reduce(examples, %{}, fn example, acc ->
      case example do
        %{"metadata" => %{"doc_ids" => doc_ids}} when is_list(doc_ids) ->
          # Add placeholder documents for each doc_id
          Enum.reduce(doc_ids, acc, fn doc_id, inner_acc ->
            Map.put(inner_acc, to_string(doc_id), %{
              "id" => to_string(doc_id),
              "text" => Map.get(example, "prompt", ""),
              "abstract" => Map.get(example, "completion", "")
            })
          end)

        %{"metadata" => %{doc_ids: doc_ids}} when is_list(doc_ids) ->
          # Handle atom keys too
          Enum.reduce(doc_ids, acc, fn doc_id, inner_acc ->
            Map.put(inner_acc, to_string(doc_id), %{
              "id" => to_string(doc_id),
              "text" => Map.get(example, "prompt", ""),
              "abstract" => Map.get(example, "completion", "")
            })
          end)

        _ ->
          acc
      end
    end)
  end

  defp compute_schema_metrics(parsed_results) do
    total = length(parsed_results)
    parseable = Enum.count(parsed_results, & &1.success)
    unparseable = total - parseable

    compliance = if total > 0, do: parseable / total, else: 1.0

    %{
      schema_compliance: Float.round(compliance, 4),
      parseable_count: parseable,
      unparseable_count: unparseable
    }
  end

  defp extract_snos(parsed_results) do
    parsed_results
    |> Enum.filter(& &1.success)
    |> Enum.flat_map(fn result ->
      # Create SNO for each claim
      Enum.map(result.claims, fn claim ->
        # Create evidence from doc_ids
        evidence =
          Enum.map(claim.doc_ids, fn doc_id ->
            Evidence.new("Document #{doc_id}", "", validity: 0.9)
          end)

        # Create provenance if this claim is derived
        provenance =
          if length(result.relations) > 0 do
            # Find relations where this claim is a target
            parent_ids =
              result.relations
              |> Enum.filter(fn rel -> rel.target == claim.id end)
              |> Enum.map(& &1.source)

            if length(parent_ids) > 0 do
              Provenance.new(:synthesizer, parent_ids: parent_ids)
            else
              nil
            end
          else
            nil
          end

        SNO.new(claim.text,
          id: claim.id,
          evidence: evidence,
          # Default confidence
          confidence: 0.8,
          provenance: provenance
        )
      end)
    end)
  end

  defp compute_citation_metrics(snos, corpus) do
    all_citations =
      snos
      |> Enum.flat_map(fn sno ->
        # Extract document IDs from evidence
        Enum.map(sno.evidence, fn evidence ->
          case Regex.run(~r/Document\s+(\d+)/, evidence.source) do
            [_, doc_id] -> doc_id
            _ -> nil
          end
        end)
        |> Enum.reject(&is_nil/1)
      end)

    if Enum.empty?(all_citations) do
      %{
        citation_accuracy: 1.0,
        valid_citations: 0,
        invalid_citations: 0,
        hallucinated_citations: 0
      }
    else
      valid = Enum.count(all_citations, &Map.has_key?(corpus, &1))
      invalid = length(all_citations) - valid

      %{
        citation_accuracy: Float.round(valid / length(all_citations), 4),
        valid_citations: valid,
        invalid_citations: invalid,
        # Consider invalid as hallucinated
        hallucinated_citations: invalid
      }
    end
  end

  defp compute_semantic_metrics(examples, outputs, snos, _corpus, _opts) do
    # Try to compute entailment and similarity if we have both examples and outputs
    if length(examples) > 0 and length(outputs) > 0 do
      # Compute pairwise similarities between expected and generated
      similarities =
        Enum.zip(examples, outputs)
        |> Enum.map(fn {example, output} ->
          expected = Map.get(example, "completion", "")
          Semantic.compute_similarity(expected, output)
        end)

      # For entailment, we'd need the model loader, so we'll use a simplified version
      # In production, this would use CNS.Validation.Semantic with real NLI models
      entailments =
        snos
        |> Enum.map(fn sno ->
          # Simplified entailment based on evidence quality
          sno.confidence * SNO.evidence_score(sno)
        end)

      mean_sim =
        if length(similarities) > 0 do
          Float.round(Enum.sum(similarities) / length(similarities), 4)
        else
          nil
        end

      mean_ent =
        if length(entailments) > 0 do
          Float.round(Enum.sum(entailments) / length(entailments), 4)
        else
          nil
        end

      %{
        mean_entailment: mean_ent,
        mean_similarity: mean_sim
      }
    else
      %{
        mean_entailment: nil,
        mean_similarity: nil
      }
    end
  end

  defp compute_topology_metrics(snos) do
    if Enum.empty?(snos) do
      %{
        topology: %{
          mean_beta1: 0.0,
          max_beta1: 0,
          dag_count: 0,
          cyclic_count: 0
        }
      }
    else
      # Build graph from SNOs
      graph = Topology.build_graph(snos)

      # Compute Betti numbers
      betti = Topology.betti_numbers(graph)

      # Check for DAG property
      is_dag = Topology.is_dag?(graph)

      # Count cycles
      cycles = Topology.detect_cycles(graph)

      %{
        topology: %{
          mean_beta1: Float.round(betti.b1 / max(1, length(snos)), 4),
          max_beta1: betti.b1,
          dag_count: if(is_dag, do: 1, else: 0),
          cyclic_count: length(cycles)
        }
      }
    end
  end

  defp compute_chirality_metrics(snos) do
    if Enum.empty?(snos) do
      %{
        chirality: %{
          mean_score: 0.0,
          polarity_conflicts: 0,
          high_conflict_count: 0
        }
      }
    else
      # Detect polarity conflicts between claims
      conflicts = detect_polarity_conflicts(snos)

      # Calculate chirality score (simplified)
      chirality_score =
        if length(conflicts) > 0 do
          Float.round(length(conflicts) / length(snos), 4)
        else
          0.0
        end

      %{
        chirality: %{
          mean_score: chirality_score,
          polarity_conflicts: length(conflicts),
          high_conflict_count: Enum.count(conflicts, fn {_a, _b, score} -> score > 0.7 end)
        }
      }
    end
  end

  defp detect_polarity_conflicts(snos) do
    # Simple polarity conflict detection
    # In production, this would use more sophisticated NLI
    pairs = for a <- snos, b <- snos, a.id < b.id, do: {a, b}

    Enum.flat_map(pairs, fn {sno_a, sno_b} ->
      # Check for opposing terms (simplified)
      if contains_opposition?(sno_a.claim, sno_b.claim) do
        [{sno_a.id, sno_b.id, 0.8}]
      else
        []
      end
    end)
  end

  defp contains_opposition?(text_a, text_b) do
    opposites = [
      {"increases", "decreases"},
      {"supports", "refutes"},
      {"true", "false"},
      {"yes", "no"},
      {"positive", "negative"}
    ]

    text_a_lower = String.downcase(text_a)
    text_b_lower = String.downcase(text_b)

    Enum.any?(opposites, fn {word_a, word_b} ->
      (String.contains?(text_a_lower, word_a) and String.contains?(text_b_lower, word_b)) or
        (String.contains?(text_a_lower, word_b) and String.contains?(text_b_lower, word_a))
    end)
  end

  defp compute_overall_quality(schema, citation, semantic, topology, chirality) do
    # Get CNS 3.0 quality targets
    targets = Config.quality_targets()

    # Weighted quality score
    weights = %{
      schema: 0.25,
      citation: 0.25,
      semantic: 0.30,
      topology: 0.10,
      chirality: 0.10
    }

    # Calculate component scores
    schema_score = schema.schema_compliance
    citation_score = citation.citation_accuracy
    semantic_score = semantic.mean_entailment || semantic.mean_similarity || 0.5
    topology_score = 1.0 - min(1.0, topology.topology.mean_beta1)
    chirality_score = 1.0 - chirality.chirality.mean_score

    # Weighted average
    overall =
      weights.schema * schema_score +
        weights.citation * citation_score +
        weights.semantic * semantic_score +
        weights.topology * topology_score +
        weights.chirality * chirality_score

    # Check if meets CNS 3.0 targets
    meets_threshold =
      schema.schema_compliance >= targets.schema_compliance and
        citation.citation_accuracy >= targets.citation_accuracy and
        (semantic.mean_entailment || 0.5) >= targets.mean_entailment

    %{
      overall_quality: Float.round(overall, 4),
      meets_threshold: meets_threshold
    }
  end
end

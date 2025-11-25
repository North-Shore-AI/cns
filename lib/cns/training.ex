defmodule CNS.TrainingLegacy do
  @moduledoc """
  DEPRECATED: This module uses legacy Crucible contracts.

  Use the new CNS.Training module instead (in lib/cns/training/).

  This module will be removed in v0.3.0.

  Legacy training integration for CNS with Tinkex LoRA fine-tuning.

  Provides utilities for:
  - Dataset preparation for dialectical training
  - LoRA configuration for CNS-specific training
  - Training loop management
  - Checkpoint handling

  ## Example

      # Prepare training data
      {:ok, dataset} = CNS.Training.prepare_dataset(snos, format: :dialectical)

      # Configure LoRA
      config = CNS.Training.lora_config(rank: 16, target: :synthesizer)

      # Train (requires Tinkex)
      {:ok, adapter} = CNS.Training.train(dataset, config)
  """

  @deprecated "Use CNS.Training (in lib/cns/training/) instead"

  alias CNS.{SNO, Evidence}

  @type dataset :: %{
          train: [map()],
          validation: [map()],
          test: [map()]
        }

  @type lora_config :: %{
          base_model: String.t(),
          rank: pos_integer(),
          alpha: pos_integer(),
          target_modules: [String.t()],
          dropout: float(),
          learning_rate: float(),
          epochs: pos_integer(),
          batch_size: pos_integer(),
          target: atom()
        }

  @doc """
  Prepare dataset for CNS training.

  Converts SNOs into training examples for fine-tuning.

  ## Options

  * `:format` - Output format (:dialectical, :qa, :nli) (default: :dialectical)
  * `:split` - Train/val/test split ratios (default: [0.8, 0.1, 0.1])
  * `:include_evidence` - Include evidence in examples (default: true)

  ## Examples

      iex> snos = [CNS.SNO.new("Claim 1"), CNS.SNO.new("Claim 2")]
      iex> {:ok, dataset} = CNS.Training.prepare_dataset(snos)
      iex> Map.has_key?(dataset, :train)
      true
  """
  @spec prepare_dataset([SNO.t()], keyword()) :: {:ok, dataset()} | {:error, term()}
  def prepare_dataset(snos, opts \\ []) when is_list(snos) do
    format = Keyword.get(opts, :format, :dialectical)
    [train_ratio, val_ratio, _test_ratio] = Keyword.get(opts, :split, [0.8, 0.1, 0.1])
    include_evidence = Keyword.get(opts, :include_evidence, true)

    # Convert SNOs to training examples
    examples = Enum.map(snos, &sno_to_example(&1, format, include_evidence))

    # Shuffle and split
    shuffled = Enum.shuffle(examples)
    n = length(shuffled)
    # Ensure at least 1 training example if we have any examples
    train_n = if n > 0, do: max(1, trunc(n * train_ratio)), else: 0
    val_n = trunc(n * val_ratio)

    {train, rest} = Enum.split(shuffled, train_n)
    {validation, test} = Enum.split(rest, val_n)

    dataset = %{
      train: train,
      validation: validation,
      test: test,
      metadata: %{
        total: n,
        format: format,
        include_evidence: include_evidence
      }
    }

    {:ok, dataset}
  rescue
    e -> {:error, Exception.message(e)}
  end

  @doc """
  Create LoRA configuration for CNS training.

  ## Options

  * `:rank` - LoRA rank (default: 16)
  * `:alpha` - LoRA alpha (default: 32)
  * `:target` - Training target (:proposer, :antagonist, :synthesizer) (default: :synthesizer)
  * `:base_model` - Base model to fine-tune (default: "mistral-7b")
  * `:learning_rate` - Learning rate (default: 2.0e-4)
  * `:epochs` - Number of training epochs (default: 3)

  ## Examples

      iex> config = CNS.Training.lora_config(rank: 8, target: :synthesizer)
      iex> config.rank
      8
  """
  @spec lora_config(keyword()) :: lora_config()
  def lora_config(opts \\ []) do
    target = Keyword.get(opts, :target, :synthesizer)

    %{
      base_model: Keyword.get(opts, :base_model, "mistral-7b"),
      rank: Keyword.get(opts, :rank, 16),
      alpha: Keyword.get(opts, :alpha, 32),
      target_modules: target_modules_for(target),
      dropout: Keyword.get(opts, :dropout, 0.1),
      learning_rate: Keyword.get(opts, :learning_rate, 2.0e-4),
      epochs: Keyword.get(opts, :epochs, 3),
      batch_size: Keyword.get(opts, :batch_size, 8),
      target: target
    }
  end

  @doc """
  Train a LoRA adapter (requires Tinkex).

  This is a placeholder that would integrate with Tinkex for actual training.

  ## Examples

      iex> {:ok, dataset} = CNS.Training.prepare_dataset([CNS.SNO.new("Test")])
      iex> config = CNS.Training.lora_config()
      iex> result = CNS.Training.train(dataset, config)
      iex> match?({:ok, _} | {:error, :tinkex_not_available}, result)
      true
  """
  @spec train(dataset(), lora_config()) :: {:ok, map()} | {:error, term()}
  def train(dataset, config) do
    # Check if Tinkex is available
    if Code.ensure_loaded?(Tinkex.LoRA) do
      do_train(dataset, config)
    else
      # Return mock result for testing without Tinkex
      {:error, :tinkex_not_available}
    end
  end

  @doc """
  Save training checkpoint.

  ## Examples

      iex> state = %{epoch: 1, loss: 0.5, model_state: %{}}
      iex> {:ok, path} = CNS.Training.save_checkpoint(state, "/tmp/cns_checkpoint")
      iex> is_binary(path)
      true
  """
  @spec save_checkpoint(map(), String.t()) :: {:ok, String.t()} | {:error, term()}
  def save_checkpoint(state, path) when is_map(state) do
    try do
      # Serialize state to binary
      binary = :erlang.term_to_binary(state)

      # Ensure directory exists
      dir = Path.dirname(path)
      File.mkdir_p!(dir)

      # Write checkpoint
      File.write!(path, binary)

      {:ok, path}
    rescue
      e -> {:error, Exception.message(e)}
    end
  end

  @doc """
  Load training checkpoint.

  ## Examples

      iex> state = %{epoch: 1, loss: 0.5}
      iex> {:ok, path} = CNS.Training.save_checkpoint(state, "/tmp/cns_test_checkpoint")
      iex> {:ok, loaded} = CNS.Training.load_checkpoint(path)
      iex> loaded.epoch
      1
  """
  @spec load_checkpoint(String.t()) :: {:ok, map()} | {:error, term()}
  def load_checkpoint(path) do
    try do
      binary = File.read!(path)
      state = :erlang.binary_to_term(binary)
      {:ok, state}
    rescue
      e -> {:error, Exception.message(e)}
    end
  end

  @doc """
  Create training examples from thesis/antithesis/synthesis triplets.

  ## Examples

      iex> thesis = CNS.SNO.new("Coffee is good")
      iex> antithesis = CNS.SNO.new("Coffee has downsides")
      iex> synthesis = CNS.SNO.new("Coffee has benefits and risks")
      iex> example = CNS.Training.triplet_to_example(thesis, antithesis, synthesis)
      iex> Map.has_key?(example, :input)
      true
  """
  @spec triplet_to_example(SNO.t(), SNO.t(), SNO.t()) :: map()
  def triplet_to_example(%SNO{} = thesis, %SNO{} = antithesis, %SNO{} = synthesis) do
    %{
      input: format_triplet_input(thesis, antithesis),
      output: synthesis.claim,
      metadata: %{
        thesis_id: thesis.id,
        antithesis_id: antithesis.id,
        synthesis_id: synthesis.id,
        thesis_confidence: thesis.confidence,
        antithesis_confidence: antithesis.confidence,
        synthesis_confidence: synthesis.confidence
      }
    }
  end

  @doc """
  Evaluate trained model on test dataset.

  ## Examples

      iex> test_data = [%{input: "Test", output: "Expected"}]
      iex> predictions = [%{output: "Expected"}]
      iex> metrics = CNS.Training.evaluate(test_data, predictions)
      iex> Map.has_key?(metrics, :accuracy)
      true
  """
  @spec evaluate([map()], [map()]) :: map()
  def evaluate(test_data, predictions) when is_list(test_data) and is_list(predictions) do
    if length(test_data) != length(predictions) do
      raise ArgumentError, "Test data and predictions must have same length"
    end

    pairs = Enum.zip(test_data, predictions)

    # Calculate exact match accuracy
    exact_matches =
      Enum.count(pairs, fn {expected, actual} ->
        Map.get(expected, :output) == Map.get(actual, :output)
      end)

    accuracy = if length(pairs) > 0, do: exact_matches / length(pairs), else: 0.0

    # Calculate average confidence if available
    confidences =
      predictions
      |> Enum.map(&Map.get(&1, :confidence, 0.5))
      |> Enum.filter(&is_number/1)

    avg_confidence =
      if length(confidences) > 0,
        do: Enum.sum(confidences) / length(confidences),
        else: 0.0

    %{
      accuracy: Float.round(accuracy, 4),
      exact_matches: exact_matches,
      total: length(pairs),
      avg_confidence: Float.round(avg_confidence, 4)
    }
  end

  @doc """
  Generate training report.
  """
  @spec training_report(map()) :: String.t()
  def training_report(results) when is_map(results) do
    """
    CNS Training Report
    ===================

    Dataset:
      - Train: #{get_in(results, [:dataset, :train]) |> length()} examples
      - Validation: #{get_in(results, [:dataset, :validation]) |> length()} examples
      - Test: #{get_in(results, [:dataset, :test]) |> length()} examples

    Training:
      - Epochs: #{Map.get(results, :epochs, "N/A")}
      - Final Loss: #{Map.get(results, :final_loss, "N/A")}
      - Best Epoch: #{Map.get(results, :best_epoch, "N/A")}

    Evaluation:
      - Accuracy: #{get_in(results, [:eval, :accuracy]) || "N/A"}
      - Avg Confidence: #{get_in(results, [:eval, :avg_confidence]) || "N/A"}

    Model:
      - Base: #{get_in(results, [:config, :base_model]) || "N/A"}
      - LoRA Rank: #{get_in(results, [:config, :rank]) || "N/A"}
    """
  end

  # Private functions

  defp sno_to_example(%SNO{} = sno, format, include_evidence) do
    base = %{
      claim: sno.claim,
      confidence: sno.confidence,
      id: sno.id
    }

    base =
      if include_evidence and not Enum.empty?(sno.evidence) do
        evidence_text =
          sno.evidence
          |> Enum.map(&format_evidence/1)
          |> Enum.join("\n")

        Map.put(base, :evidence, evidence_text)
      else
        base
      end

    case format do
      :dialectical ->
        Map.merge(base, %{
          input: "Analyze and synthesize: #{sno.claim}",
          output: sno.claim,
          type: :dialectical
        })

      :qa ->
        Map.merge(base, %{
          question: "What is the evidence for: #{sno.claim}?",
          answer: format_evidence_answer(sno.evidence),
          type: :qa
        })

      :nli ->
        Map.merge(base, %{
          premise: sno.claim,
          hypothesis: "This claim is well-supported.",
          label: if(sno.confidence > 0.7, do: :entailment, else: :neutral),
          type: :nli
        })

      _ ->
        base
    end
  end

  defp format_evidence(%Evidence{} = evidence) do
    "[#{evidence.source}] (validity: #{Float.round(evidence.validity, 2)})"
  end

  defp format_evidence_answer([]), do: "No evidence available."

  defp format_evidence_answer(evidence) do
    evidence
    |> Enum.map(&"- #{&1.source}: #{&1.content}")
    |> Enum.join("\n")
  end

  defp target_modules_for(:proposer), do: ["q_proj", "v_proj", "k_proj"]
  defp target_modules_for(:antagonist), do: ["q_proj", "v_proj", "o_proj"]
  defp target_modules_for(:synthesizer), do: ["q_proj", "v_proj", "k_proj", "o_proj"]
  defp target_modules_for(_), do: ["q_proj", "v_proj"]

  defp format_triplet_input(%SNO{} = thesis, %SNO{} = antithesis) do
    """
    Thesis: #{thesis.claim}
    Confidence: #{thesis.confidence}

    Antithesis: #{antithesis.claim}
    Confidence: #{antithesis.confidence}

    Synthesize these perspectives:
    """
  end

  defp do_train(dataset, config) do
    # This would integrate with Tinkex.LoRA.train/1
    # For now, return a mock successful result
    {:ok,
     %{
       adapter_path: "/tmp/cns_adapter_#{System.unique_integer([:positive])}",
       config: config,
       dataset: dataset,
       epochs: config.epochs,
       final_loss: 0.1,
       best_epoch: config.epochs
     }}
  end
end

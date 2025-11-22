# CNS Experiment Harness

## Version
- Date: 2025-11-21
- Status: Design Document
- Author: Architecture Team

## Overview

This document describes how to run CNS experiments using crucible_harness for orchestration, crucible_datasets for data loading, crucible_bench for statistical analysis, and crucible_telemetry for metrics collection.

## Experiment Architecture

```
+------------------------------------------------------------------+
|                    CNS Experiment Harness                         |
+------------------------------------------------------------------+
|                                                                    |
|  +-------------------+    +-------------------+                    |
|  | Crucible.Harness  |    | CNS.Experiment    |                    |
|  | - DSL             |    | - Dataset loading |                    |
|  | - Orchestration   |    | - Metric defs     |                    |
|  | - Checkpointing   |    | - Evaluation      |                    |
|  +-------------------+    +-------------------+                    |
|                                                                    |
+------------------------------------------------------------------+
                               |
                               v
+------------------------------------------------------------------+
|                    Data & Analysis                                |
+------------------------------------------------------------------+
|  crucible_datasets  |  crucible_bench  |  crucible_telemetry      |
+------------------------------------------------------------------+
```

## Module Structure

```
lib/cns/experiment/
  harness.ex                  # Harness integration
  datasets.ex                 # Dataset loading and preprocessing
  metrics.ex                  # CNS-specific metrics
  evaluation.ex               # Batch evaluation
  reporter.ex                 # Result reporting
```

## Experiment Configuration

```elixir
defmodule CNS.Experiment.Config do
  @moduledoc """
  Configuration for CNS experiments.
  """

  @type t :: %__MODULE__{
    name: String.t(),
    description: String.t(),
    dataset: atom(),
    dataset_opts: keyword(),
    training: training_config(),
    synthesis: synthesis_config(),
    critics: critics_config(),
    evaluation: evaluation_config(),
    telemetry: telemetry_config()
  }

  @type training_config :: %{
    enabled: boolean(),
    base_model: String.t(),
    lora_rank: pos_integer(),
    epochs: pos_integer(),
    batch_size: pos_integer(),
    learning_rate: float(),
    loss_fn: atom() | tuple()
  }

  @type synthesis_config :: %{
    strategy: :single_shot | :iterative | :ensemble,
    max_iterations: pos_integer(),
    min_critic_score: float(),
    ensemble_size: pos_integer()
  }

  @type critics_config :: %{
    enabled: [atom()],
    weights: %{atom() => float()},
    thresholds: %{atom() => float()}
  }

  @type evaluation_config :: %{
    metrics: [atom()],
    statistical_tests: [atom()],
    significance_level: float()
  }

  @type telemetry_config :: %{
    backend: :ets | :postgres,
    export_format: :csv | :json | :parquet
  }

  defstruct [
    :name,
    :description,
    :dataset,
    dataset_opts: [],
    training: %{enabled: false},
    synthesis: %{
      strategy: :iterative,
      max_iterations: 5,
      min_critic_score: 0.6
    },
    critics: %{
      enabled: [:logic, :grounding, :novelty, :causal, :bias],
      weights: %{logic: 0.25, grounding: 0.30, novelty: 0.15, causal: 0.20, bias: 0.10},
      thresholds: %{logic: 0.7, grounding: 0.8, novelty: 0.3, causal: 0.6, bias: 0.5}
    },
    evaluation: %{
      metrics: [:accuracy, :f1, :critic_score, :topology_validity],
      statistical_tests: [:t_test, :mann_whitney],
      significance_level: 0.05
    },
    telemetry: %{
      backend: :ets,
      export_format: :csv
    }
  ]

  def new(opts \\ []), do: struct(__MODULE__, opts)
end
```

## Dataset Loading

```elixir
defmodule CNS.Experiment.Datasets do
  @moduledoc """
  Dataset loading and preprocessing for CNS experiments.
  """

  alias Crucible.Datasets

  @doc """
  Loads a dataset and converts to CNS format.
  """
  @spec load(atom(), keyword()) :: {:ok, [map()]} | {:error, term()}
  def load(dataset_name, opts \\ []) do
    split = Keyword.get(opts, :split, :train)
    limit = Keyword.get(opts, :limit)

    with {:ok, raw_data} <- Datasets.load(dataset_name, split: split) do
      processed = raw_data
      |> Enum.map(&transform_to_cns(dataset_name, &1))
      |> maybe_limit(limit)

      {:ok, processed}
    end
  end

  @doc """
  Streams dataset for memory-efficient processing.
  """
  def stream(dataset_name, opts \\ []) do
    batch_size = Keyword.get(opts, :batch_size, 32)

    dataset_name
    |> Datasets.stream(opts)
    |> Stream.map(&transform_to_cns(dataset_name, &1))
    |> Stream.chunk_every(batch_size)
  end

  # Dataset-specific transformations

  defp transform_to_cns(:scifact, example) do
    claim = example["claim"]
    label = example["label"]

    # Extract thesis/antithesis from claim based on label
    {thesis, antithesis} = case label do
      "SUPPORTS" ->
        {claim, negate_claim(claim)}
      "REFUTES" ->
        {negate_claim(claim), claim}
      "NOT_ENOUGH_INFO" ->
        {claim, "Insufficient evidence to evaluate: #{claim}"}
    end

    %{
      id: example["id"],
      thesis: thesis,
      antithesis: antithesis,
      evidence: Enum.map(example["evidence"], &format_scifact_evidence/1),
      label: label,
      metadata: %{
        dataset: :scifact,
        claim_id: example["id"]
      }
    }
  end

  defp transform_to_cns(:fever, example) do
    claim = example["claim"]
    label = example["label"]

    {thesis, antithesis} = case label do
      "SUPPORTS" ->
        {claim, negate_claim(claim)}
      "REFUTES" ->
        {negate_claim(claim), claim}
      "NOT ENOUGH INFO" ->
        {claim, "Cannot verify: #{claim}"}
    end

    %{
      id: example["id"],
      thesis: thesis,
      antithesis: antithesis,
      evidence: format_fever_evidence(example["evidence_wiki"]),
      label: label,
      metadata: %{
        dataset: :fever,
        verifiable: label != "NOT ENOUGH INFO"
      }
    }
  end

  defp transform_to_cns(:custom, example) do
    # Direct pass-through for custom datasets
    %{
      id: example["id"] || generate_id(),
      thesis: example["thesis"],
      antithesis: example["antithesis"],
      evidence: example["evidence"] || [],
      label: example["label"],
      metadata: example["metadata"] || %{}
    }
  end

  defp format_scifact_evidence(evidence) do
    %{
      id: "e#{evidence["sentence_id"]}",
      content: evidence["text"],
      source: evidence["doc_id"],
      source_type: :study
    }
  end

  defp format_fever_evidence(evidence_list) when is_list(evidence_list) do
    evidence_list
    |> Enum.with_index()
    |> Enum.map(fn {text, idx} ->
      %{
        id: "e#{idx + 1}",
        content: text,
        source: "Wikipedia",
        source_type: :document
      }
    end)
  end

  defp negate_claim(claim) do
    # Simple negation - would use NLI model in practice
    "It is not the case that: #{claim}"
  end

  defp maybe_limit(data, nil), do: data
  defp maybe_limit(data, limit), do: Enum.take(data, limit)

  defp generate_id do
    :crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)
  end
end
```

## CNS Metrics

```elixir
defmodule CNS.Experiment.Metrics do
  @moduledoc """
  CNS-specific metrics for experiment evaluation.
  """

  @type metric_result :: %{
    name: atom(),
    value: float(),
    details: map()
  }

  @doc """
  Computes all configured metrics for experiment results.
  """
  @spec compute(list(), keyword()) :: [metric_result()]
  def compute(results, opts \\ []) do
    metrics = Keyword.get(opts, :metrics, [:accuracy, :f1, :critic_score, :topology_validity])

    Enum.map(metrics, fn metric ->
      {metric, compute_metric(metric, results)}
    end)
  end

  @doc """
  Registers custom CNS metrics with crucible_bench.
  """
  def register_metrics do
    Crucible.Bench.register_metric(:critic_score, &compute_critic_score/1)
    Crucible.Bench.register_metric(:topology_validity, &compute_topology_validity/1)
    Crucible.Bench.register_metric(:evidence_coverage, &compute_evidence_coverage/1)
    Crucible.Bench.register_metric(:dialectical_balance, &compute_dialectical_balance/1)
    Crucible.Bench.register_metric(:citation_accuracy, &compute_citation_accuracy/1)
  end

  defp compute_metric(:accuracy, results) do
    correct = Enum.count(results, fn r ->
      r.evaluation.passed and r.prediction == r.label
    end)
    %{
      name: :accuracy,
      value: correct / length(results),
      details: %{correct: correct, total: length(results)}
    }
  end

  defp compute_metric(:f1, results) do
    # Compute macro F1 across labels
    labels = Enum.map(results, & &1.label) |> Enum.uniq()

    f1_scores = Enum.map(labels, fn label ->
      tp = Enum.count(results, fn r -> r.prediction == label and r.label == label end)
      fp = Enum.count(results, fn r -> r.prediction == label and r.label != label end)
      fn_ = Enum.count(results, fn r -> r.prediction != label and r.label == label end)

      precision = if tp + fp > 0, do: tp / (tp + fp), else: 0
      recall = if tp + fn_ > 0, do: tp / (tp + fn_), else: 0

      if precision + recall > 0 do
        2 * precision * recall / (precision + recall)
      else
        0
      end
    end)

    macro_f1 = Enum.sum(f1_scores) / length(f1_scores)
    %{name: :f1, value: macro_f1, details: %{per_label: Enum.zip(labels, f1_scores)}}
  end

  defp compute_metric(:critic_score, results) do
    scores = Enum.map(results, fn r -> r.evaluation.overall_score end)
    mean = Enum.sum(scores) / length(scores)
    %{
      name: :critic_score,
      value: mean,
      details: %{
        min: Enum.min(scores),
        max: Enum.max(scores),
        std: std_dev(scores)
      }
    }
  end

  defp compute_metric(:topology_validity, results) do
    valid = Enum.count(results, fn r ->
      r.topology.betti_1 == 0 and r.topology.connectivity >= 0.9
    end)
    %{
      name: :topology_validity,
      value: valid / length(results),
      details: %{valid: valid, total: length(results)}
    }
  end

  defp compute_metric(metric, results) do
    # Delegate to registered metric
    Crucible.Bench.compute_metric(metric, results)
  end

  # Additional metric implementations

  defp compute_critic_score(results) do
    scores = Enum.map(results, & &1.evaluation.overall_score)
    Enum.sum(scores) / length(scores)
  end

  defp compute_topology_validity(results) do
    valid = Enum.count(results, fn r -> r.topology.betti_1 == 0 end)
    valid / length(results)
  end

  defp compute_evidence_coverage(results) do
    coverages = Enum.map(results, & &1.evaluation.metadata[:coverage_ratio])
    Enum.sum(coverages) / length(coverages)
  end

  defp compute_dialectical_balance(results) do
    balances = Enum.map(results, & &1.evaluation.individual_scores[:bias])
    Enum.sum(balances) / length(balances)
  end

  defp compute_citation_accuracy(results) do
    accuracies = Enum.map(results, fn r ->
      grounding = r.evaluation.individual_scores[:grounding]
      grounding
    end)
    Enum.sum(accuracies) / length(accuracies)
  end

  defp std_dev(values) do
    mean = Enum.sum(values) / length(values)
    variance = Enum.map(values, fn v -> (v - mean) * (v - mean) end)
    |> Enum.sum()
    |> Kernel./(length(values))
    :math.sqrt(variance)
  end
end
```

## Harness Integration

```elixir
defmodule CNS.Experiment.Harness do
  @moduledoc """
  Integrates CNS with crucible_harness for experiment orchestration.
  """

  alias CNS.Experiment.{Config, Datasets, Metrics, Evaluation}
  alias Crucible.{Harness, Lora, Bench, Telemetry}

  @doc """
  Runs a complete CNS experiment.
  """
  @spec run(Config.t(), keyword()) :: {:ok, map()} | {:error, term()}
  def run(%Config{} = config, opts \\ []) do
    # Register CNS metrics
    Metrics.register_metrics()

    # Create experiment
    {:ok, experiment} = create_experiment(config)

    # Run with harness orchestration
    Harness.run(experiment, fn ctx ->
      # Phase 1: Training (if enabled)
      ctx = if config.training.enabled do
        run_training_phase(ctx, config)
      else
        ctx
      end

      # Phase 2: Synthesis
      {:ok, results} = run_synthesis_phase(ctx, config)

      # Phase 3: Evaluation
      {:ok, evaluated} = run_evaluation_phase(results, config)

      # Phase 4: Analysis
      {:ok, analysis} = run_analysis_phase(evaluated, config)

      {:ok, %{
        results: evaluated,
        analysis: analysis,
        experiment_id: experiment.id
      }}
    end)
  end

  @doc """
  Runs a batch of experiments with different configurations.
  """
  @spec run_batch([Config.t()], keyword()) :: {:ok, [map()]} | {:error, term()}
  def run_batch(configs, opts \\ []) do
    parallel = Keyword.get(opts, :parallel, false)

    if parallel do
      tasks = Enum.map(configs, fn config ->
        Task.async(fn -> run(config, opts) end)
      end)
      results = Task.await_many(tasks, :infinity)
      {:ok, results}
    else
      results = Enum.map(configs, fn config ->
        {:ok, result} = run(config, opts)
        result
      end)
      {:ok, results}
    end
  end

  defp create_experiment(config) do
    Lora.create_experiment(
      name: config.name,
      description: config.description,
      tags: [:cns, config.dataset]
    )
  end

  defp run_training_phase(ctx, config) do
    # Load training dataset
    {:ok, train_data} = Datasets.load(config.dataset,
      split: :train,
      limit: config.dataset_opts[:train_limit]
    )

    # Configure training
    training_opts = [
      epochs: config.training.epochs,
      batch_size: config.training.batch_size,
      learning_rate: config.training.learning_rate,
      loss_fn: config.training.loss_fn,
      checkpoint_every: config.dataset_opts[:checkpoint_every] || 100
    ]

    # Run LoRA training
    {:ok, training_results} = Lora.TrainingLoop.run(
      ctx.session,
      train_data,
      training_opts
    )

    Map.put(ctx, :training_results, training_results)
  end

  defp run_synthesis_phase(ctx, config) do
    # Load test dataset
    {:ok, test_data} = Datasets.load(config.dataset,
      split: :test,
      limit: config.dataset_opts[:test_limit]
    )

    # Get sampling client
    sampling_client = ctx.session.sampling_client

    # Process each example
    results = Enum.map(test_data, fn example ->
      # Create SNO
      sno = CNS.SNO.new(
        thesis: example.thesis,
        antithesis: example.antithesis,
        evidence: example.evidence
      )

      # Generate synthesis
      synthesis_opts = [
        sampling_client: sampling_client,
        strategy: config.synthesis.strategy,
        max_iterations: config.synthesis.max_iterations,
        min_critic_score: config.synthesis.min_critic_score
      ]

      {:ok, sno_with_synthesis} = CNS.Synthesis.EnginePool.synthesize(sno, synthesis_opts)

      # Evaluate with critics
      {:ok, evaluation} = CNS.Critics.Pipeline.evaluate_all(sno_with_synthesis,
        critics: config.critics.enabled
      )

      # Get topology metrics
      {:ok, topology} = CNS.SNO.topology_metrics(sno_with_synthesis)

      # Emit telemetry
      emit_synthesis_telemetry(sno_with_synthesis, evaluation, topology)

      %{
        id: example.id,
        sno: sno_with_synthesis,
        evaluation: evaluation,
        topology: topology,
        label: example.label,
        prediction: infer_label(evaluation)
      }
    end)

    {:ok, results}
  end

  defp run_evaluation_phase(results, config) do
    # Compute metrics
    metrics = Metrics.compute(results, metrics: config.evaluation.metrics)

    # Add metrics to results
    evaluated = %{
      results: results,
      metrics: Map.new(metrics, fn {name, result} -> {name, result.value} end),
      metric_details: Map.new(metrics, fn {name, result} -> {name, result.details} end)
    }

    {:ok, evaluated}
  end

  defp run_analysis_phase(evaluated, config) do
    # Statistical analysis with crucible_bench
    {:ok, analysis} = Bench.analyze(evaluated.results,
      tests: config.evaluation.statistical_tests,
      alpha: config.evaluation.significance_level
    )

    # Generate summary
    summary = %{
      total_samples: length(evaluated.results),
      passed_synthesis: Enum.count(evaluated.results, & &1.evaluation.passed),
      metrics: evaluated.metrics,
      statistical_tests: analysis.tests,
      significant_findings: analysis.significant
    }

    {:ok, %{
      metrics: evaluated.metrics,
      metric_details: evaluated.metric_details,
      statistical_analysis: analysis,
      summary: summary
    }}
  end

  defp infer_label(evaluation) do
    # Infer label from synthesis evaluation
    if evaluation.passed do
      cond do
        evaluation.individual_scores[:grounding] > 0.8 -> "SUPPORTS"
        evaluation.individual_scores[:grounding] < 0.3 -> "REFUTES"
        true -> "NOT ENOUGH INFO"
      end
    else
      "NOT ENOUGH INFO"
    end
  end

  defp emit_synthesis_telemetry(sno, evaluation, topology) do
    Telemetry.emit([:cns, :synthesis, :complete], %{
      duration: 0,  # Would be measured
      iterations: 1
    }, %{
      sno_id: sno.id,
      overall_score: evaluation.overall_score,
      passed: evaluation.passed,
      betti_0: topology.betti_0,
      betti_1: topology.betti_1
    })
  end
end
```

## Report Generation

```elixir
defmodule CNS.Experiment.Reporter do
  @moduledoc """
  Generates reports for CNS experiments.
  """

  @doc """
  Generates a comprehensive experiment report.
  """
  @spec generate(map(), keyword()) :: {:ok, String.t()}
  def generate(results, opts \\ []) do
    format = Keyword.get(opts, :format, :markdown)

    report = case format do
      :markdown -> generate_markdown(results)
      :latex -> generate_latex(results)
      :html -> generate_html(results)
    end

    {:ok, report}
  end

  defp generate_markdown(results) do
    """
    # CNS Experiment Report

    ## Summary

    - **Total Samples**: #{results.analysis.summary.total_samples}
    - **Passed Synthesis**: #{results.analysis.summary.passed_synthesis}
    - **Pass Rate**: #{Float.round(results.analysis.summary.passed_synthesis / results.analysis.summary.total_samples * 100, 1)}%

    ## Metrics

    #{format_metrics_table(results.analysis.metrics)}

    ## Critic Performance

    #{format_critic_scores(results.results)}

    ## Statistical Analysis

    #{format_statistical_tests(results.analysis.statistical_analysis)}

    ## Key Findings

    #{format_findings(results.analysis)}

    ---
    Generated by CNS Experiment Harness
    """
  end

  defp format_metrics_table(metrics) do
    header = "| Metric | Value |\n|--------|-------|\n"
    rows = Enum.map(metrics, fn {name, value} ->
      "| #{name} | #{Float.round(value, 4)} |"
    end)
    |> Enum.join("\n")

    header <> rows
  end

  defp format_critic_scores(results) do
    # Aggregate critic scores
    critics = [:logic, :grounding, :novelty, :causal, :bias]

    scores = Enum.map(critics, fn critic ->
      critic_scores = Enum.map(results, fn r ->
        r.evaluation.individual_scores[critic]
      end)
      mean = Enum.sum(critic_scores) / length(critic_scores)
      {critic, mean}
    end)

    header = "| Critic | Mean Score |\n|--------|------------|\n"
    rows = Enum.map(scores, fn {critic, score} ->
      "| #{critic} | #{Float.round(score, 3)} |"
    end)
    |> Enum.join("\n")

    header <> rows
  end

  defp format_statistical_tests(analysis) do
    Enum.map(analysis.tests, fn test ->
      """
      ### #{test.name}
      - **Statistic**: #{Float.round(test.statistic, 4)}
      - **p-value**: #{Float.round(test.p_value, 4)}
      - **Significant**: #{test.significant}
      """
    end)
    |> Enum.join("\n")
  end

  defp format_findings(analysis) do
    analysis.summary.significant_findings
    |> Enum.map(&"- #{&1}")
    |> Enum.join("\n")
  end

  defp generate_latex(results) do
    # LaTeX report generation
    "% LaTeX report not implemented"
  end

  defp generate_html(results) do
    # HTML report generation
    "<html>HTML report not implemented</html>"
  end
end
```

## Complete Experiment Example

```elixir
# Define experiment configuration
config = CNS.Experiment.Config.new(
  name: "SciFact CNS Evaluation",
  description: "Evaluate CNS dialectical synthesis on SciFact claims",
  dataset: :scifact,
  dataset_opts: [
    train_limit: 1000,
    test_limit: 200,
    checkpoint_every: 100
  ],
  training: %{
    enabled: true,
    base_model: "llama-3-8b",
    lora_rank: 16,
    epochs: 3,
    batch_size: 8,
    learning_rate: 1.0e-4,
    loss_fn: {:composite, [:cross_entropy, :topological, :citation], [0.6, 0.2, 0.2]}
  },
  synthesis: %{
    strategy: :iterative,
    max_iterations: 3,
    min_critic_score: 0.65
  },
  critics: %{
    enabled: [:logic, :grounding, :novelty, :causal, :bias],
    weights: %{logic: 0.25, grounding: 0.30, novelty: 0.15, causal: 0.20, bias: 0.10},
    thresholds: %{logic: 0.7, grounding: 0.75, novelty: 0.3, causal: 0.6, bias: 0.5}
  },
  evaluation: %{
    metrics: [:accuracy, :f1, :critic_score, :topology_validity, :evidence_coverage],
    statistical_tests: [:t_test, :mann_whitney, :wilcoxon],
    significance_level: 0.05
  },
  telemetry: %{
    backend: :ets,
    export_format: :csv
  }
)

# Run experiment
{:ok, results} = CNS.Experiment.Harness.run(config)

# Generate report
{:ok, report} = CNS.Experiment.Reporter.generate(results, format: :markdown)
IO.puts(report)

# Export telemetry data
Crucible.Telemetry.export(
  experiment_id: results.experiment_id,
  format: :csv,
  path: "/tmp/cns_experiment_#{results.experiment_id}.csv"
)

# Compare with baseline
baseline_results = load_baseline_results()
{:ok, comparison} = Crucible.Bench.compare(
  baseline: baseline_results,
  treatment: results.results,
  metrics: [:accuracy, :critic_score],
  tests: [:t_test, :mann_whitney]
)

IO.inspect(comparison.summary)
# %{
#   accuracy: %{baseline: 0.78, treatment: 0.85, delta: 0.07, significant: true},
#   critic_score: %{baseline: 0.72, treatment: 0.81, delta: 0.09, significant: true}
# }
```

## Batch Experiments

```elixir
# Define multiple configurations for ablation study
configs = [
  # Baseline: no topology loss
  CNS.Experiment.Config.new(
    name: "Ablation: No Topology Loss",
    training: %{loss_fn: :cross_entropy},
    # ... other config
  ),

  # With topology loss
  CNS.Experiment.Config.new(
    name: "Ablation: With Topology Loss",
    training: %{loss_fn: {:composite, [:cross_entropy, :topological], [0.8, 0.2]}},
    # ... other config
  ),

  # Full loss
  CNS.Experiment.Config.new(
    name: "Full: All Losses",
    training: %{loss_fn: {:composite, [:cross_entropy, :topological, :citation], [0.6, 0.2, 0.2]}},
    # ... other config
  )
]

# Run batch
{:ok, batch_results} = CNS.Experiment.Harness.run_batch(configs, parallel: true)

# Compare all experiments
{:ok, comparison} = Crucible.Bench.compare_multiple(
  batch_results,
  metrics: [:accuracy, :critic_score, :topology_validity],
  post_hoc: :tukey
)

IO.inspect(comparison.rankings)
# [
#   {0, "Full: All Losses", %{accuracy: 0.87}},
#   {1, "Ablation: With Topology Loss", %{accuracy: 0.84}},
#   {2, "Ablation: No Topology Loss", %{accuracy: 0.78}}
# ]
```

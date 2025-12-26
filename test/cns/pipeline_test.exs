defmodule CNS.PipelineTest do
  use ExUnit.Case, async: true

  alias CNS.Agents.Pipeline
  alias CNS.{Config, SNO}

  describe "run/2" do
    test "runs full pipeline" do
      config = %Config{max_iterations: 2}
      assert {:ok, result} = Pipeline.run("Test question about effects", config)

      assert Map.has_key?(result, :final_synthesis)
      assert Map.has_key?(result, :iterations)
      assert Map.has_key?(result, :convergence_score)
      assert Map.has_key?(result, :evidence_chain)
    end

    test "respects max_iterations" do
      config = %Config{max_iterations: 1}
      assert {:ok, result} = Pipeline.run("Test input", config)

      assert result.iterations <= 1
    end

    test "returns metrics" do
      config = %Config{max_iterations: 2}
      assert {:ok, result} = Pipeline.run("Input text", config)

      assert Map.has_key?(result, :metrics)
      assert is_map(result.metrics)
    end
  end

  describe "configure/1" do
    test "creates config from options" do
      config = Pipeline.configure(max_iterations: 10, convergence_threshold: 0.9)

      assert config.max_iterations == 10
      assert config.convergence_threshold == 0.9
    end

    test "uses defaults for missing options" do
      config = Pipeline.configure()

      assert config.max_iterations == 5
      assert config.convergence_threshold == 0.85
    end
  end

  describe "converged?/2" do
    test "returns true when thresholds met" do
      evidence = [CNS.Evidence.new("S", "C", validity: 0.9)]
      synthesis = SNO.new("Result", confidence: 0.9, evidence: evidence)

      config = %Config{
        convergence_threshold: 0.85,
        coherence_threshold: 0.5,
        evidence_threshold: 0.5
      }

      assert Pipeline.converged?(synthesis, config)
    end

    test "returns false when below threshold" do
      synthesis = SNO.new("Result", confidence: 0.5)
      config = %Config{convergence_threshold: 0.85}

      refute Pipeline.converged?(synthesis, config)
    end
  end

  describe "iterate/2" do
    test "iterates with two claims" do
      claims = [
        SNO.new("Claim 1", id: "c1"),
        SNO.new("Claim 2", id: "c2")
      ]

      config = %Config{}

      assert {:ok, result} = Pipeline.iterate(claims, config)

      assert Map.has_key?(result, :synthesis)
      assert Map.has_key?(result, :challenges)
    end

    test "handles single claim" do
      claims = [SNO.new("Only claim", id: "c1")]
      config = %Config{}

      assert {:ok, result} = Pipeline.iterate(claims, config)
      assert result.synthesis.claim == "Only claim"
    end

    test "returns error for empty claims" do
      assert {:error, _} = Pipeline.iterate([], %Config{})
    end
  end

  describe "run_async/2" do
    test "returns a task" do
      config = %Config{max_iterations: 1}
      task = Pipeline.run_async("Test", config)

      assert %Task{} = task

      # Wait for result
      assert {:ok, result} = Task.await(task, 5000)
      assert Map.has_key?(result, :final_synthesis)
    end
  end

  describe "status/1" do
    test "returns pipeline status" do
      state = %{
        iteration: 2,
        converged: false,
        claims: [SNO.new("A"), SNO.new("B")],
        challenges: [],
        synthesis: nil
      }

      status = Pipeline.status(state)

      assert status.iteration == 2
      assert status.converged == false
      assert status.claim_count == 2
      assert status.challenge_count == 0
      assert status.has_synthesis == false
    end
  end
end

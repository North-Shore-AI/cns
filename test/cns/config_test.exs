defmodule CNS.ConfigTest do
  use ExUnit.Case, async: true

  alias CNS.Config

  describe "new/1" do
    test "creates config with defaults" do
      config = Config.new()

      assert config.max_iterations == 5
      assert config.convergence_threshold == 0.85
      assert config.coherence_threshold == 0.8
      assert config.evidence_threshold == 0.7
      assert config.telemetry_enabled == true
      assert config.timeout == 30_000
    end

    test "creates config with overrides" do
      config = Config.new(max_iterations: 10, convergence_threshold: 0.9)

      assert config.max_iterations == 10
      assert config.convergence_threshold == 0.9
    end
  end

  describe "validate/1" do
    test "validates correct config" do
      config = %Config{}
      assert {:ok, ^config} = Config.validate(config)
    end

    test "rejects non-positive max_iterations" do
      config = %Config{max_iterations: 0}
      assert {:error, errors} = Config.validate(config)
      assert "max_iterations must be a positive integer" in errors
    end

    test "rejects invalid thresholds" do
      config = %Config{convergence_threshold: 1.5}
      assert {:error, errors} = Config.validate(config)
      assert "convergence_threshold must be a number between 0.0 and 1.0" in errors
    end

    test "rejects non-positive timeout" do
      config = %Config{timeout: -1}
      assert {:error, errors} = Config.validate(config)
      assert "timeout must be a positive integer" in errors
    end
  end

  describe "merge/2" do
    test "merges with keyword list" do
      base = %Config{max_iterations: 5}
      merged = Config.merge(base, max_iterations: 10)

      assert merged.max_iterations == 10
    end

    test "merges with another config" do
      base = %Config{max_iterations: 5, convergence_threshold: 0.8}
      other = %Config{max_iterations: 10}

      merged = Config.merge(base, other)
      assert merged.max_iterations == 10
    end
  end

  describe "quality_targets/0" do
    test "returns quality target map" do
      targets = Config.quality_targets()

      assert targets.schema_compliance == 0.95
      assert targets.citation_accuracy == 0.95
      assert targets.mean_entailment == 0.50
      assert targets.min_confidence == 0.85
    end
  end

  describe "to_map/1 and from_map/1" do
    test "round-trips config" do
      config = Config.new(max_iterations: 7, convergence_threshold: 0.9)

      map = Config.to_map(config)
      assert {:ok, restored} = Config.from_map(map)

      assert restored.max_iterations == 7
      assert restored.convergence_threshold == 0.9
    end
  end
end

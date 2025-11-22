defmodule CNS.Training.EvaluationTest do
  use ExUnit.Case, async: true
  alias CNS.Training.Evaluation
  alias CNS.Training.Evaluation.{Metrics, EvalConfig, EvalResult}

  describe "Metrics struct" do
    test "can be constructed with all fields" do
      metrics = %Metrics{
        precision: 0.85,
        recall: 0.80,
        f1: 0.824,
        accuracy: 0.82
      }

      assert metrics.precision == 0.85
      assert metrics.f1 == 0.824
    end
  end

  describe "EvalConfig struct" do
    test "has sensible defaults" do
      config = %EvalConfig{}

      assert config.batch_size == 32
      assert config.max_samples == nil
    end
  end

  describe "compute_metrics/2" do
    test "computes perfect score for identical predictions" do
      predictions = ["A", "B", "C"]
      gold = ["A", "B", "C"]

      metrics = Evaluation.compute_metrics(predictions, gold)

      assert metrics.accuracy == 1.0
      assert metrics.precision == 1.0
      assert metrics.recall == 1.0
      assert metrics.f1 == 1.0
    end

    test "computes zero score for completely wrong predictions" do
      predictions = ["X", "Y", "Z"]
      gold = ["A", "B", "C"]

      metrics = Evaluation.compute_metrics(predictions, gold)

      assert metrics.accuracy == 0.0
    end

    test "computes partial accuracy" do
      predictions = ["A", "X", "C"]
      gold = ["A", "B", "C"]

      metrics = Evaluation.compute_metrics(predictions, gold)

      assert_in_delta metrics.accuracy, 2 / 3, 0.01
    end

    test "handles empty inputs" do
      metrics = Evaluation.compute_metrics([], [])

      assert metrics.accuracy == 0.0
      assert metrics.precision == 0.0
    end
  end

  describe "evaluate_claims/3" do
    test "evaluates claim extraction output" do
      predictions = [
        "CLAIM[c1]: Test claim\nRELATION: c2 supports c1"
      ]

      gold = [
        "CLAIM[c1]: Test claim\nRELATION: c2 supports c1"
      ]

      config = %EvalConfig{}
      result = Evaluation.evaluate_claims(predictions, gold, config)

      assert %EvalResult{} = result
      assert result.metrics.accuracy == 1.0
    end

    test "computes metrics for partial matches" do
      predictions = [
        "CLAIM[c1]: First claim",
        "CLAIM[c1]: Different claim"
      ]

      gold = [
        "CLAIM[c1]: First claim",
        "CLAIM[c1]: Second claim"
      ]

      config = %EvalConfig{}
      result = Evaluation.evaluate_claims(predictions, gold, config)

      assert result.metrics.accuracy == 0.5
    end
  end

  describe "compute_f1/2" do
    test "computes F1 from precision and recall" do
      f1 = Evaluation.compute_f1(0.8, 0.6)
      expected = 2 * 0.8 * 0.6 / (0.8 + 0.6)

      assert_in_delta f1, expected, 0.001
    end

    test "returns 0 when both are 0" do
      f1 = Evaluation.compute_f1(0.0, 0.0)

      assert f1 == 0.0
    end
  end

  describe "extract_claims_from_output/1" do
    test "extracts claim texts from formatted output" do
      output = "CLAIM[c1]: First claim\nCLAIM[c2]: Second claim"
      claims = Evaluation.extract_claims_from_output(output)

      assert length(claims) == 2
      assert "First claim" in claims
      assert "Second claim" in claims
    end

    test "returns empty list for no claims" do
      claims = Evaluation.extract_claims_from_output("No claims here")

      assert claims == []
    end
  end

  describe "extract_relations_from_output/1" do
    test "extracts relations from formatted output" do
      output = "RELATION: c2 supports c1\nRELATION: c3 refutes c1"
      relations = Evaluation.extract_relations_from_output(output)

      assert length(relations) == 2
    end

    test "normalizes relation format" do
      output = "RELATION: c2 supports c1"
      [relation] = Evaluation.extract_relations_from_output(output)

      assert relation == {"c2", "supports", "c1"}
    end
  end
end

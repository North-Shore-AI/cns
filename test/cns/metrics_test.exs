defmodule CNS.MetricsTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias CNS.{Challenge, Evidence, Metrics, SNO}

  describe "quality_score/1" do
    test "calculates overall quality" do
      evidence = [Evidence.new("S", "C", validity: 0.8)]
      sno = SNO.new("Claim", evidence: evidence, confidence: 0.9)

      metrics = Metrics.quality_score(sno)

      assert Map.has_key?(metrics, :overall)
      assert metrics.overall > 0
      assert metrics.confidence == 0.9
    end
  end

  describe "entailment/2" do
    test "calculates overlap score" do
      premise = SNO.new("Coffee contains caffeine", confidence: 0.8)
      conclusion = SNO.new("Caffeine is found in coffee", confidence: 0.7)

      score = Metrics.entailment(premise, conclusion)
      assert score > 0
    end

    test "returns 0 for unrelated claims" do
      premise = SNO.new("Apples are fruits", confidence: 0.8)
      conclusion = SNO.new("Cars have wheels", confidence: 0.8)

      score = Metrics.entailment(premise, conclusion)
      assert score < 0.5
    end
  end

  describe "citation_accuracy/1" do
    test "calculates ratio of valid citations" do
      snos = [
        SNO.new("A", evidence: [Evidence.new("S", "C", validity: 0.9)]),
        SNO.new("B", evidence: [Evidence.new("S", "C", validity: 0.5)])
      ]

      accuracy = Metrics.citation_accuracy(snos)
      assert accuracy == 0.5
    end

    test "returns 0 for no evidence" do
      snos = [SNO.new("A"), SNO.new("B")]
      assert Metrics.citation_accuracy(snos) == 0.0
    end
  end

  describe "pass_rate/2" do
    test "calculates pass rate" do
      snos = [
        SNO.new("A", confidence: 0.9),
        SNO.new("B", confidence: 0.4),
        SNO.new("C", confidence: 0.6)
      ]

      assert Metrics.pass_rate(snos, 0.5) == 0.6667
    end

    test "returns 0 for empty list" do
      assert Metrics.pass_rate([], 0.5) == 0.0
    end
  end

  describe "chirality/1" do
    test "calculates average chirality" do
      challenges = [
        Challenge.new("id", :contradiction, "Test", severity: :high, confidence: 0.8),
        Challenge.new("id", :scope, "Test", severity: :low, confidence: 0.5)
      ]

      score = Metrics.chirality(challenges)
      assert score > 0
    end

    test "returns 0 for empty challenges" do
      assert Metrics.chirality([]) == 0.0
    end
  end

  describe "fisher_rao_distance/2" do
    test "calculates distance between distributions" do
      dist1 = [0.2, 0.5, 0.3]
      dist2 = [0.3, 0.4, 0.3]

      distance = Metrics.fisher_rao_distance(dist1, dist2)
      assert distance >= 0.0
    end

    test "returns 0 for identical distributions" do
      dist = [0.25, 0.25, 0.25, 0.25]
      distance = Metrics.fisher_rao_distance(dist, dist)

      assert_in_delta distance, 0.0, 0.001
    end

    test "raises for different length distributions" do
      assert_raise ArgumentError, fn ->
        Metrics.fisher_rao_distance([0.5, 0.5], [0.33, 0.33, 0.34])
      end
    end
  end

  describe "schema_compliance/1" do
    test "returns 1.0 for valid SNOs" do
      snos = [
        SNO.new("Valid 1", confidence: 0.8),
        SNO.new("Valid 2", confidence: 0.7)
      ]

      assert Metrics.schema_compliance(snos) == 1.0
    end

    test "calculates compliance ratio" do
      snos = [
        SNO.new("Valid", confidence: 0.8),
        # Invalid
        %SNO{claim: "", confidence: 0.5}
      ]

      compliance = Metrics.schema_compliance(snos)
      assert compliance == 0.5
    end
  end

  describe "mean_entailment/1" do
    test "calculates mean across pairs" do
      pairs = [
        {SNO.new("Premise 1"), SNO.new("Conclusion 1")},
        {SNO.new("Premise 2"), SNO.new("Conclusion 2")}
      ]

      mean = Metrics.mean_entailment(pairs)
      assert mean >= 0.0
    end

    test "returns 0 for empty pairs" do
      assert Metrics.mean_entailment([]) == 0.0
    end
  end

  describe "convergence_delta/2" do
    test "calculates confidence change" do
      prev = SNO.new("Old", confidence: 0.6)
      curr = SNO.new("New", confidence: 0.8)

      assert Metrics.convergence_delta(prev, curr) == 0.2
    end

    test "can be negative" do
      prev = SNO.new("Old", confidence: 0.8)
      curr = SNO.new("New", confidence: 0.6)

      assert Metrics.convergence_delta(prev, curr) == -0.2
    end
  end

  describe "meets_targets?/1" do
    test "returns true when all targets met" do
      metrics = %{
        schema_compliance: 0.96,
        citation_accuracy: 0.97,
        mean_entailment: 0.55
      }

      assert Metrics.meets_targets?(metrics)
    end

    test "returns false when any target missed" do
      metrics = %{
        # Below 0.95
        schema_compliance: 0.90,
        citation_accuracy: 0.97,
        mean_entailment: 0.55
      }

      refute Metrics.meets_targets?(metrics)
    end
  end

  describe "report/2" do
    test "generates comprehensive report" do
      snos = [SNO.new("A", confidence: 0.8)]
      challenges = [Challenge.new("id", :contradiction, "Test")]

      report = Metrics.report(snos, challenges)

      assert Map.has_key?(report, :count)
      assert Map.has_key?(report, :schema_compliance)
      assert Map.has_key?(report, :citation_accuracy)
      assert Map.has_key?(report, :chirality)
    end
  end

  describe "property tests" do
    property "pass_rate is between 0 and 1" do
      check all(
              confidences <- list_of(float(min: 0.0, max: 1.0), min_length: 1, max_length: 10),
              threshold <- float(min: 0.0, max: 1.0)
            ) do
        snos = Enum.map(confidences, &SNO.new("Test", confidence: &1))
        rate = Metrics.pass_rate(snos, threshold)

        assert rate >= 0.0 and rate <= 1.0
      end
    end

    property "fisher_rao_distance is non-negative" do
      check all(
              n <- integer(2..5),
              dist1 <- list_of(float(min: 0.01, max: 1.0), length: n),
              dist2 <- list_of(float(min: 0.01, max: 1.0), length: n)
            ) do
        distance = Metrics.fisher_rao_distance(dist1, dist2)
        assert distance >= 0.0
      end
    end
  end
end

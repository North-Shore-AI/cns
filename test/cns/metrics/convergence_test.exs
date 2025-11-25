defmodule CNS.Metrics.ConvergenceTest do
  use ExUnit.Case, async: true

  alias CNS.Metrics
  alias CNS.{SNO, Evidence}

  describe "convergence_score/2" do
    test "returns 1.0 for identical SNOs" do
      sno = %SNO{
        claim: "Test claim",
        confidence: 0.8,
        evidence: [%Evidence{content: "E1", source: "test"}]
      }

      score = Metrics.convergence_score(sno, sno)
      assert score == 1.0
    end

    test "returns lower score for different confidence levels" do
      prev = %SNO{
        claim: "Previous claim",
        confidence: 0.6,
        evidence: [%Evidence{content: "E1", source: "test"}]
      }

      curr = %SNO{
        claim: "Current claim",
        confidence: 0.9,
        evidence: [%Evidence{content: "E1", source: "test"}]
      }

      score = Metrics.convergence_score(prev, curr)
      assert score < 1.0
      assert score >= 0.0
      # Score should be 1.0 - |0.9 - 0.6| = 0.7
      assert_in_delta(score, 0.7, 0.01)
    end

    test "handles missing confidence values" do
      prev = %SNO{claim: "Previous"}
      curr = %SNO{claim: "Current"}

      score = Metrics.convergence_score(prev, curr)
      assert score == 1.0
    end
  end

  describe "evidential_entanglement/2" do
    test "computes overlap correctly" do
      sno_a = %SNO{
        claim: "Claim A",
        evidence: [
          %Evidence{source: "test", content: "E1"},
          %Evidence{source: "test", content: "E2"},
          %Evidence{source: "test", content: "E3"}
        ]
      }

      sno_b = %SNO{
        claim: "Claim B",
        evidence: [
          %Evidence{source: "test", content: "E2"},
          %Evidence{source: "test", content: "E3"},
          %Evidence{source: "test", content: "E4"}
        ]
      }

      entanglement = Metrics.evidential_entanglement(sno_a, sno_b)

      # Intersection: {E2, E3} = 2
      # Union: {E1, E2, E3, E4} = 4
      # Expected: 2/4 = 0.5
      assert_in_delta(entanglement, 0.5, 0.01)
    end

    test "returns 0.0 for no overlap" do
      sno_a = %SNO{
        claim: "Claim A",
        evidence: [
          %Evidence{source: "test", content: "E1"},
          %Evidence{source: "test", content: "E2"}
        ]
      }

      sno_b = %SNO{
        claim: "Claim B",
        evidence: [
          %Evidence{source: "test", content: "E3"},
          %Evidence{source: "test", content: "E4"}
        ]
      }

      entanglement = Metrics.evidential_entanglement(sno_a, sno_b)
      assert entanglement == 0.0
    end

    test "returns 1.0 for identical evidence sets" do
      evidence = [
        %Evidence{source: "test", content: "E1"},
        %Evidence{source: "test", content: "E2"}
      ]

      sno_a = %SNO{claim: "Claim A", evidence: evidence}
      sno_b = %SNO{claim: "Claim B", evidence: evidence}

      entanglement = Metrics.evidential_entanglement(sno_a, sno_b)
      assert entanglement == 1.0
    end

    test "handles empty evidence sets" do
      sno_a = %SNO{claim: "Claim A", evidence: []}
      sno_b = %SNO{claim: "Claim B", evidence: []}

      entanglement = Metrics.evidential_entanglement(sno_a, sno_b)
      assert entanglement == 0.0
    end
  end

  describe "overall_quality/1" do
    test "computes overall quality score" do
      sno = %SNO{
        claim: "High quality claim",
        confidence: 0.9,
        evidence: [
          %Evidence{source: "test", content: "E1", validity: 0.9},
          %Evidence{source: "test", content: "E2", validity: 0.8}
        ]
      }

      result = Metrics.overall_quality(sno)
      assert Map.has_key?(result, :score)
      assert Map.has_key?(result, :meets_threshold)
      assert Map.has_key?(result, :breakdown)
      assert result.score > 0
    end
  end
end
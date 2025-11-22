defmodule CNS.Metrics.ChiralityTest do
  use ExUnit.Case, async: true
  alias CNS.Metrics.Chirality
  alias CNS.Metrics.Chirality.{FisherRaoStats, ChiralityResult}

  describe "build_fisher_rao_stats/2" do
    test "computes mean and inverse variance" do
      vectors =
        Nx.tensor([
          [1.0, 2.0, 3.0],
          [2.0, 3.0, 4.0],
          [1.5, 2.5, 3.5]
        ])

      stats = Chirality.build_fisher_rao_stats(vectors)

      assert %FisherRaoStats{} = stats
      assert Nx.shape(stats.mean) == {3}
      assert Nx.shape(stats.inv_var) == {3}
    end

    test "handles list input" do
      vectors = [[1.0, 2.0], [1.5, 2.5], [1.2, 2.1]]
      stats = Chirality.build_fisher_rao_stats(vectors)

      assert %FisherRaoStats{} = stats
    end

    test "epsilon prevents division by zero" do
      # Zero variance in first dimension
      vectors =
        Nx.tensor([
          [1.0, 2.0],
          [1.0, 3.0],
          [1.0, 2.5]
        ])

      stats = Chirality.build_fisher_rao_stats(vectors, 1.0e-6)

      # Should not raise, inv_var should be finite
      inv_var_values = Nx.to_flat_list(stats.inv_var)
      assert Enum.all?(inv_var_values, &is_number/1)
      assert Enum.all?(inv_var_values, fn v -> v < 1.0e12 end)
    end

    test "mean is computed correctly" do
      vectors =
        Nx.tensor([
          [1.0, 4.0],
          [2.0, 5.0],
          [3.0, 6.0]
        ])

      stats = Chirality.build_fisher_rao_stats(vectors)
      mean_values = Nx.to_flat_list(stats.mean)

      assert_in_delta Enum.at(mean_values, 0), 2.0, 1.0e-6
      assert_in_delta Enum.at(mean_values, 1), 5.0, 1.0e-6
    end
  end

  describe "fisher_rao_distance/3" do
    test "computes correct distance with unit weights" do
      stats = %FisherRaoStats{
        mean: Nx.tensor([0.0, 0.0]),
        inv_var: Nx.tensor([1.0, 1.0])
      }

      vec_a = Nx.tensor([0.0, 0.0])
      vec_b = Nx.tensor([3.0, 4.0])

      distance = Chirality.fisher_rao_distance(vec_a, vec_b, stats)

      # Should equal Euclidean distance when inv_var = 1
      assert_in_delta distance, 5.0, 1.0e-6
    end

    test "weights by inverse variance" do
      stats = %FisherRaoStats{
        mean: Nx.tensor([0.0, 0.0]),
        inv_var: Nx.tensor([4.0, 1.0])
      }

      vec_a = Nx.tensor([0.0, 0.0])
      vec_b = Nx.tensor([1.0, 2.0])

      distance = Chirality.fisher_rao_distance(vec_a, vec_b, stats)

      # sqrt(4*1^2 + 1*2^2) = sqrt(4 + 4) = sqrt(8)
      assert_in_delta distance, :math.sqrt(8), 1.0e-6
    end

    test "distance is zero for identical vectors" do
      stats = %FisherRaoStats{
        mean: Nx.tensor([1.0, 2.0]),
        inv_var: Nx.tensor([1.0, 1.0])
      }

      vec = Nx.tensor([1.0, 2.0])
      distance = Chirality.fisher_rao_distance(vec, vec, stats)

      assert_in_delta distance, 0.0, 1.0e-6
    end

    test "is symmetric" do
      stats = %FisherRaoStats{
        mean: Nx.tensor([0.0, 0.0]),
        inv_var: Nx.tensor([2.0, 3.0])
      }

      vec_a = Nx.tensor([1.0, 2.0])
      vec_b = Nx.tensor([4.0, 5.0])

      dist_ab = Chirality.fisher_rao_distance(vec_a, vec_b, stats)
      dist_ba = Chirality.fisher_rao_distance(vec_b, vec_a, stats)

      assert_in_delta dist_ab, dist_ba, 1.0e-6
    end
  end

  describe "compute_chirality_score/3" do
    test "high distance produces high chirality" do
      score = Chirality.compute_chirality_score(10.0, 0.0, false)

      # norm_distance = 10/11 ≈ 0.909
      # score = 0.909 * 0.6 + 1.0 * 0.2 = 0.545 + 0.2 = 0.745
      assert score > 0.7
      assert score < 0.8
    end

    test "low distance produces low chirality" do
      score = Chirality.compute_chirality_score(0.1, 0.0, false)

      # norm_distance = 0.1/1.1 ≈ 0.091
      # score = 0.091 * 0.6 + 1.0 * 0.2 = 0.055 + 0.2 = 0.255
      assert score < 0.3
    end

    test "polarity conflict adds penalty" do
      score_without = Chirality.compute_chirality_score(1.0, 0.0, false)
      score_with = Chirality.compute_chirality_score(1.0, 0.0, true)

      assert_in_delta score_with - score_without, 0.25, 1.0e-6
    end

    test "high evidence overlap reduces score" do
      score_low_overlap = Chirality.compute_chirality_score(1.0, 0.1, false)
      score_high_overlap = Chirality.compute_chirality_score(1.0, 0.9, false)

      assert score_low_overlap > score_high_overlap
    end

    test "score is clamped to max 1.0" do
      # High distance + low overlap + conflict could exceed 1.0
      score = Chirality.compute_chirality_score(100.0, 0.0, true)

      assert score == 1.0
    end

    test "overlap is clamped to valid range" do
      # Test with overlap > 1.0
      score = Chirality.compute_chirality_score(1.0, 1.5, false)
      # Should be treated as 1.0
      assert is_float(score)
    end
  end

  describe "ChiralityResult struct" do
    test "can be constructed with all fields" do
      result = %ChiralityResult{
        fisher_rao_distance: 2.5,
        evidence_overlap: 0.3,
        polarity_conflict: true,
        chirality_score: 0.65
      }

      assert result.fisher_rao_distance == 2.5
      assert result.evidence_overlap == 0.3
      assert result.polarity_conflict == true
      assert result.chirality_score == 0.65
    end
  end
end

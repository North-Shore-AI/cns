defmodule CNS.Logic.BettiTest do
  use ExUnit.Case, async: true
  alias CNS.Logic.Betti

  describe "compute_graph_stats/2" do
    test "simple acyclic graph" do
      claim_ids = ["c1", "c2", "c3"]

      relations = [
        {"c2", "supports", "c1"},
        {"c3", "supports", "c1"}
      ]

      stats = Betti.compute_graph_stats(claim_ids, relations)

      assert stats.nodes == 3
      assert stats.edges == 2
      assert stats.components == 1
      assert stats.beta1 == 0
      assert stats.cycles == []
      assert stats.polarity_conflict == false
    end

    test "graph with cycle" do
      claim_ids = ["c1", "c2", "c3"]

      relations = [
        {"c1", "supports", "c2"},
        {"c2", "supports", "c3"},
        {"c3", "supports", "c1"}
      ]

      stats = Betti.compute_graph_stats(claim_ids, relations)

      assert stats.beta1 >= 1
      assert length(stats.cycles) >= 1
    end

    test "polarity conflict detection" do
      claim_ids = ["c1", "c2", "c3"]

      relations = [
        {"c2", "supports", "c1"},
        {"c3", "refutes", "c1"}
      ]

      stats = Betti.compute_graph_stats(claim_ids, relations)

      assert stats.polarity_conflict == true
    end

    test "disconnected components" do
      claim_ids = ["c1", "c2", "c3", "c4"]

      relations = [
        {"c1", "supports", "c2"},
        {"c3", "supports", "c4"}
      ]

      stats = Betti.compute_graph_stats(claim_ids, relations)

      assert stats.components == 2
    end

    test "empty graph" do
      stats = Betti.compute_graph_stats([], [])

      assert stats.nodes == 0
      assert stats.edges == 0
      assert stats.beta1 == 0
    end

    test "case-insensitive id normalization" do
      claim_ids = ["C1", "c2"]
      relations = [{"C2", "supports", "c1"}]

      stats = Betti.compute_graph_stats(claim_ids, relations)

      assert stats.edges == 1
    end

    test "graph with multiple cycles" do
      claim_ids = ["c1", "c2", "c3", "c4"]

      relations = [
        {"c1", "supports", "c2"},
        {"c2", "supports", "c3"},
        {"c3", "supports", "c1"},
        {"c1", "supports", "c4"},
        {"c4", "supports", "c2"}
      ]

      stats = Betti.compute_graph_stats(claim_ids, relations)

      # 5 edges - 4 nodes + 1 component = 2
      assert stats.beta1 >= 1
    end

    test "single node no edges" do
      stats = Betti.compute_graph_stats(["c1"], [])

      assert stats.nodes == 1
      assert stats.edges == 0
      assert stats.components == 1
      assert stats.beta1 == 0
    end
  end

  describe "polarity_conflict?/2" do
    test "returns true when both supports and refutes exist" do
      relations = [
        {"c2", "supports", "c1"},
        {"c3", "refutes", "c1"}
      ]

      assert Betti.polarity_conflict?(relations, "c1") == true
    end

    test "returns false with only supports" do
      relations = [
        {"c2", "supports", "c1"},
        {"c3", "supports", "c1"}
      ]

      assert Betti.polarity_conflict?(relations, "c1") == false
    end

    test "returns false with only refutes" do
      relations = [
        {"c2", "refutes", "c1"},
        {"c3", "refutes", "c1"}
      ]

      assert Betti.polarity_conflict?(relations, "c1") == false
    end

    test "handles custom target" do
      relations = [
        {"c1", "supports", "c5"},
        {"c2", "refutes", "c5"}
      ]

      assert Betti.polarity_conflict?(relations, "c5") == true
    end

    test "case insensitive target matching" do
      relations = [
        {"c2", "supports", "C1"},
        {"c3", "refutes", "c1"}
      ]

      assert Betti.polarity_conflict?(relations, "c1") == true
    end

    test "case insensitive label matching" do
      relations = [
        {"c2", "SUPPORTS", "c1"},
        {"c3", "REFUTES", "c1"}
      ]

      assert Betti.polarity_conflict?(relations, "c1") == true
    end

    test "returns false for empty relations" do
      assert Betti.polarity_conflict?([], "c1") == false
    end

    test "returns false for unrelated target" do
      relations = [
        {"c2", "supports", "c1"},
        {"c3", "refutes", "c1"}
      ]

      assert Betti.polarity_conflict?(relations, "c5") == false
    end
  end
end

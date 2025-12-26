defmodule CNS.Critics.NoveltyTest do
  use ExUnit.Case, async: true

  alias CNS.Critics.Novelty
  alias CNS.{Evidence, SNO}

  describe "start_link/1" do
    test "starts the novelty critic" do
      assert {:ok, pid} = Novelty.start_link(name: :test_novelty)
      assert is_pid(pid)
      GenServer.stop(pid)
    end
  end

  describe "evaluate/1 without GenServer" do
    test "returns high score for original claim" do
      sno =
        SNO.new("This is a novel synthesis combining multiple perspectives on quantum computing")

      {:ok, result} = Novelty.evaluate(sno)

      assert result.score > 0.5
      assert is_list(result.issues)
      assert is_map(result.details)
    end

    test "penalizes claim that copies evidence" do
      evidence = [
        Evidence.new("Source", "The exact same text as the claim", validity: 0.8)
      ]

      sno = SNO.new("The exact same text as the claim", evidence: evidence)

      {:ok, result} = Novelty.evaluate(sno)

      assert result.details.originality_score < 0.5
    end

    test "rewards claim with new content beyond evidence" do
      evidence = [
        Evidence.new("Source", "Original study data", validity: 0.8)
      ]

      sno =
        SNO.new(
          "A novel interpretation suggests that the original study data implies broader conclusions about social behavior",
          evidence: evidence
        )

      {:ok, result} = Novelty.evaluate(sno)

      assert result.details.originality_score > 0.5
    end

    test "checks information density" do
      sno = SNO.new("The the a an is are was were")

      {:ok, result} = Novelty.evaluate(sno)

      assert result.details.density_score < 0.5
      assert Enum.any?(result.issues, &String.contains?(&1, "density"))
    end

    test "rewards high information density" do
      sno =
        SNO.new(
          "Quantum entanglement enables instantaneous information correlation across vast distances"
        )

      {:ok, result} = Novelty.evaluate(sno)

      assert result.details.density_score > 0.4
    end

    test "penalizes overly long claims" do
      long_claim = String.duplicate("word ", 160) |> String.trim()
      sno = SNO.new(long_claim)

      {:ok, result} = Novelty.evaluate(sno)

      assert result.details.parsimony_score < 1.0
      assert Enum.any?(result.issues, &String.contains?(&1, "verbose"))
    end

    test "detects trivial claims" do
      sno = SNO.new("Yes.")

      {:ok, result} = Novelty.evaluate(sno)

      assert result.details.nontrivial_score == 0.0
      assert Enum.any?(result.issues, &String.contains?(&1, "trivial"))
    end

    test "accepts non-trivial claims" do
      sno =
        SNO.new(
          "The relationship between economic growth and environmental sustainability is complex"
        )

      {:ok, result} = Novelty.evaluate(sno)

      assert result.details.nontrivial_score == 1.0
    end

    test "handles claims with no evidence" do
      sno = SNO.new("A claim without any evidence references")

      {:ok, result} = Novelty.evaluate(sno)

      # Should have good originality since nothing to copy from
      assert result.details.originality_score >= 0.8
    end
  end

  describe "call/3 with GenServer" do
    setup do
      {:ok, pid} = Novelty.start_link(name: :"novelty_#{:erlang.unique_integer()}")
      %{novelty: pid}
    end

    test "evaluates via GenServer", %{novelty: novelty} do
      sno = SNO.new("A novel claim for testing")

      {:ok, result} = Novelty.call(novelty, sno)

      assert result.score > 0
      assert is_map(result.details)
    end
  end

  describe "name/0" do
    test "returns :novelty" do
      assert Novelty.name() == :novelty
    end
  end

  describe "weight/0" do
    test "returns 0.15" do
      assert Novelty.weight() == 0.15
    end
  end

  describe "details" do
    test "includes all score components" do
      sno = SNO.new("Test claim for novelty analysis")

      {:ok, result} = Novelty.evaluate(sno)

      assert Map.has_key?(result.details, :originality_score)
      assert Map.has_key?(result.details, :density_score)
      assert Map.has_key?(result.details, :parsimony_score)
      assert Map.has_key?(result.details, :nontrivial_score)
    end
  end
end

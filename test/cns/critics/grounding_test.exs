defmodule CNS.Critics.GroundingTest do
  use ExUnit.Case, async: true

  alias CNS.Critics.Grounding
  alias CNS.{Evidence, SNO}

  describe "start_link/1" do
    test "starts the grounding critic" do
      assert {:ok, pid} = Grounding.start_link(name: :test_grounding)
      assert is_pid(pid)
      GenServer.stop(pid)
    end
  end

  describe "evaluate/1 without GenServer" do
    test "returns high score for well-grounded SNO" do
      evidence = [
        Evidence.new("Source A", "Content related to the claim", validity: 0.9),
        Evidence.new("Source B", "More supporting content", validity: 0.85)
      ]

      sno = SNO.new("This is a well-grounded claim", evidence: evidence)

      {:ok, result} = Grounding.evaluate(sno)

      assert result.score > 0.5
      assert is_list(result.issues)
      assert is_map(result.details)
    end

    test "returns lower score for SNO without evidence" do
      sno = SNO.new("Claim without any evidence")

      {:ok, result} = Grounding.evaluate(sno)

      assert result.score < 0.5
      assert Enum.any?(result.issues, &String.contains?(&1, "no_evidence"))
    end

    test "penalizes low validity evidence" do
      evidence = [
        Evidence.new("Weak source", "Weak content", validity: 0.3)
      ]

      sno = SNO.new("Claim with weak evidence", evidence: evidence)

      {:ok, result} = Grounding.evaluate(sno)

      assert result.details.validity_score < 0.5
    end

    test "rewards high validity evidence" do
      evidence = [
        Evidence.new("Strong source", "Strong content", validity: 0.95)
      ]

      sno = SNO.new("Claim with strong evidence", evidence: evidence)

      {:ok, result} = Grounding.evaluate(sno)

      assert result.details.validity_score > 0.9
    end

    test "checks evidence relevance" do
      evidence = [
        Evidence.new("Source", "completely unrelated random text", validity: 0.8)
      ]

      sno = SNO.new("Claim about climate change", evidence: evidence)

      {:ok, result} = Grounding.evaluate(sno)

      # Should have lower relevance score
      assert result.details.relevance_score < 0.5
    end

    test "rewards evidence matching claim keywords" do
      evidence = [
        Evidence.new("Source", "climate change affects global temperatures", validity: 0.8)
      ]

      sno = SNO.new("Climate change affects temperatures globally", evidence: evidence)

      {:ok, result} = Grounding.evaluate(sno)

      # Should have higher relevance score
      assert result.details.relevance_score > 0.3
    end

    test "checks source diversity" do
      evidence = [
        Evidence.new("Same source", "Content 1", validity: 0.8),
        Evidence.new("Same source", "Content 2", validity: 0.8),
        Evidence.new("Same source", "Content 3", validity: 0.8)
      ]

      sno = SNO.new("Claim with low diversity", evidence: evidence)

      {:ok, result} = Grounding.evaluate(sno)

      assert result.details.diversity_score < 0.5
      assert Enum.any?(result.issues, &String.contains?(&1, "diversity"))
    end

    test "rewards diverse sources" do
      evidence = [
        Evidence.new("Source A", "Content 1", validity: 0.8),
        Evidence.new("Source B", "Content 2", validity: 0.8),
        Evidence.new("Source C", "Content 3", validity: 0.8)
      ]

      sno = SNO.new("Claim with diverse sources", evidence: evidence)

      {:ok, result} = Grounding.evaluate(sno)

      assert result.details.diversity_score == 1.0
    end
  end

  describe "call/3 with GenServer" do
    setup do
      {:ok, pid} = Grounding.start_link(name: :"grounding_#{:erlang.unique_integer()}")
      %{grounding: pid}
    end

    test "evaluates via GenServer", %{grounding: grounding} do
      evidence = Evidence.new("Source", "Content", validity: 0.8)
      sno = SNO.new("Test claim", evidence: [evidence])

      {:ok, result} = Grounding.call(grounding, sno)

      assert result.score > 0
      assert is_map(result.details)
    end
  end

  describe "name/0" do
    test "returns :grounding" do
      assert Grounding.name() == :grounding
    end
  end

  describe "weight/0" do
    test "returns 0.4" do
      assert Grounding.weight() == 0.4
    end
  end

  describe "details" do
    test "includes all score components" do
      evidence = Evidence.new("Source", "Content", validity: 0.7)
      sno = SNO.new("Test claim with evidence", evidence: [evidence])

      {:ok, result} = Grounding.evaluate(sno)

      assert Map.has_key?(result.details, :coverage_score)
      assert Map.has_key?(result.details, :validity_score)
      assert Map.has_key?(result.details, :relevance_score)
      assert Map.has_key?(result.details, :diversity_score)
      assert Map.has_key?(result.details, :evidence_count)
      assert Map.has_key?(result.details, :avg_validity)
    end
  end
end

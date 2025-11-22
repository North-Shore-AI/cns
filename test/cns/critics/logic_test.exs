defmodule CNS.Critics.LogicTest do
  use ExUnit.Case, async: true

  alias CNS.Critics.Logic
  alias CNS.{SNO, Evidence}

  describe "start_link/1" do
    test "starts the logic critic" do
      assert {:ok, pid} = Logic.start_link(name: :test_logic)
      assert is_pid(pid)
      GenServer.stop(pid)
    end
  end

  describe "evaluate/3 with GenServer" do
    setup do
      {:ok, pid} = Logic.start_link(name: :"logic_#{:erlang.unique_integer()}")
      %{logic: pid}
    end

    test "returns high score for consistent SNO", %{logic: logic} do
      evidence = Evidence.new("Source", "Supporting content", validity: 0.8)

      sno =
        SNO.new("This is a well-structured claim with evidence",
          evidence: [evidence],
          confidence: 0.9
        )

      {:ok, result} = Logic.call(logic, sno)

      assert result.score > 0.5
      assert is_list(result.issues)
      assert is_map(result.details)
    end

    test "returns lower score for SNO without evidence", %{logic: logic} do
      sno = SNO.new("Claim without evidence")

      {:ok, result} = Logic.call(logic, sno)

      assert result.score < 1.0
      assert Enum.any?(result.issues, &String.contains?(&1, "no_evidence"))
    end

    test "penalizes very short claims", %{logic: logic} do
      sno = SNO.new("Short")

      {:ok, result} = Logic.call(logic, sno)

      assert Enum.any?(result.issues, &String.contains?(&1, "claim_too_short"))
    end

    test "detects contradictions in children", %{logic: logic} do
      child1 = SNO.new("The drug is effective")
      child2 = SNO.new("The drug is not effective")

      sno = SNO.new("Analysis of drug efficacy", children: [child1, child2])

      {:ok, result} = Logic.call(logic, sno)

      assert result.details.contradictions_found > 0
      assert Enum.any?(result.issues, &String.contains?(&1, "contradiction"))
    end
  end

  describe "evaluate/1 without GenServer" do
    test "works synchronously" do
      evidence = Evidence.new("Source", "Content", validity: 0.85)
      sno = SNO.new("A valid claim", evidence: [evidence])

      {:ok, result} = Logic.evaluate(sno)

      assert result.score > 0
      assert is_list(result.issues)
    end
  end

  describe "name/0" do
    test "returns :logic" do
      assert Logic.name() == :logic
    end
  end

  describe "weight/0" do
    test "returns 0.3" do
      assert Logic.weight() == 0.3
    end
  end

  describe "details" do
    setup do
      {:ok, pid} = Logic.start_link(name: :"logic_detail_#{:erlang.unique_integer()}")
      %{logic: pid}
    end

    test "includes all score components", %{logic: logic} do
      sno = SNO.new("Test claim with sufficient content")

      {:ok, result} = Logic.call(logic, sno)

      assert Map.has_key?(result.details, :cycle_score)
      assert Map.has_key?(result.details, :contradiction_score)
      assert Map.has_key?(result.details, :entailment_score)
      assert Map.has_key?(result.details, :structure_score)
    end
  end
end

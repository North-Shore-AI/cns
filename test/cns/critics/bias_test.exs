defmodule CNS.Critics.BiasTest do
  use ExUnit.Case, async: true

  alias CNS.Critics.Bias
  alias CNS.{Evidence, SNO}

  describe "start_link/1" do
    test "starts the bias critic" do
      assert {:ok, pid} = Bias.start_link(name: :test_bias)
      assert is_pid(pid)
      GenServer.stop(pid)
    end
  end

  describe "evaluate/1 without GenServer" do
    test "returns high score for balanced claim" do
      sno =
        SNO.new(
          "While remote work offers flexibility, it also presents challenges for team collaboration"
        )

      {:ok, result} = Bias.evaluate(sno)

      assert result.score > 0.7
      assert is_list(result.issues)
      assert is_map(result.details)
    end

    test "penalizes loaded negative language" do
      sno = SNO.new("This terrible policy is a complete disaster for everyone")

      {:ok, result} = Bias.evaluate(sno)

      assert result.details.loaded_score < 1.0
      assert Enum.any?(result.issues, &String.contains?(&1, "loaded_language"))
    end

    test "penalizes loaded positive language" do
      sno = SNO.new("This amazing revolutionary approach is absolutely incredible")

      {:ok, result} = Bias.evaluate(sno)

      assert result.details.loaded_score < 1.0
      assert Enum.any?(result.issues, &String.contains?(&1, "loaded_language"))
    end

    test "detects one-sided positive framing" do
      sno =
        SNO.new(
          "The policy brings many benefits and advantages, improving outcomes and making things better"
        )

      {:ok, result} = Bias.evaluate(sno)

      assert result.details.framing_score < 1.0
      assert Enum.any?(result.issues, &String.contains?(&1, "one_sided"))
    end

    test "detects one-sided negative framing" do
      sno =
        SNO.new(
          "The policy causes harm and disadvantage, leading to negative decline and bad outcomes"
        )

      {:ok, result} = Bias.evaluate(sno)

      assert result.details.framing_score < 1.0
      assert Enum.any?(result.issues, &String.contains?(&1, "one_sided"))
    end

    test "rewards balanced framing" do
      sno =
        SNO.new(
          "The policy has both benefits and disadvantages, with positive effects on some and negative impacts on others"
        )

      {:ok, result} = Bias.evaluate(sno)

      assert result.details.framing_score == 1.0
    end

    test "penalizes absolutist language" do
      sno = SNO.new("Everyone always agrees that nothing ever works")

      {:ok, result} = Bias.evaluate(sno)

      assert result.details.absolutist_score < 1.0
      assert Enum.any?(result.issues, &String.contains?(&1, "absolutist"))
    end

    test "accepts qualified language" do
      sno = SNO.new("Some researchers suggest that certain conditions may influence outcomes")

      {:ok, result} = Bias.evaluate(sno)

      assert result.details.absolutist_score == 1.0
    end

    test "penalizes low evidence diversity" do
      evidence = [
        Evidence.new("Same source", "Content 1", validity: 0.8),
        Evidence.new("Same source", "Content 2", validity: 0.8)
      ]

      sno = SNO.new("Claim with concentrated sources", evidence: evidence)

      {:ok, result} = Bias.evaluate(sno)

      assert result.details.diversity_score < 1.0
    end

    test "rewards evidence diversity" do
      evidence = [
        Evidence.new("Source A", "Content 1", validity: 0.8),
        Evidence.new("Source B", "Content 2", validity: 0.8)
      ]

      sno = SNO.new("Claim with diverse sources", evidence: evidence)

      {:ok, result} = Bias.evaluate(sno)

      assert result.details.diversity_score == 1.0
    end

    test "rewards balance indicators in long claims" do
      sno =
        SNO.new(
          "While the evidence suggests positive outcomes in some cases, however, there are also documented instances where the opposite was true"
        )

      {:ok, result} = Bias.evaluate(sno)

      assert result.details.balance_score == 1.0
    end

    test "penalizes lack of balance in long claims" do
      long_claim =
        "This policy is good and it works well and it helps people and it improves things and it makes everything better for everyone involved in the process"

      sno = SNO.new(long_claim)

      {:ok, result} = Bias.evaluate(sno)

      assert result.details.balance_score < 1.0
      assert Enum.any?(result.issues, &String.contains?(&1, "unbalanced"))
    end
  end

  describe "call/3 with GenServer" do
    setup do
      {:ok, pid} = Bias.start_link(name: :"bias_#{:erlang.unique_integer()}")
      %{bias: pid}
    end

    test "evaluates via GenServer", %{bias: bias} do
      sno = SNO.new("A balanced claim for testing")

      {:ok, result} = Bias.call(bias, sno)

      assert result.score > 0
      assert is_map(result.details)
    end
  end

  describe "name/0" do
    test "returns :bias" do
      assert Bias.name() == :bias
    end
  end

  describe "weight/0" do
    test "returns 0.05" do
      assert Bias.weight() == 0.05
    end
  end

  describe "details" do
    test "includes all score components" do
      sno = SNO.new("Test claim for bias analysis")

      {:ok, result} = Bias.evaluate(sno)

      assert Map.has_key?(result.details, :loaded_score)
      assert Map.has_key?(result.details, :framing_score)
      assert Map.has_key?(result.details, :absolutist_score)
      assert Map.has_key?(result.details, :diversity_score)
      assert Map.has_key?(result.details, :balance_score)
    end
  end
end

defmodule CNS.Critics.CausalTest do
  use ExUnit.Case, async: true

  alias CNS.Critics.Causal
  alias CNS.SNO

  describe "start_link/1" do
    test "starts the causal critic" do
      assert {:ok, pid} = Causal.start_link(name: :test_causal)
      assert is_pid(pid)
      GenServer.stop(pid)
    end
  end

  describe "evaluate/1 without GenServer" do
    test "returns high score for well-formed causal claim" do
      sno =
        SNO.new(
          "Exercise causes improved cardiovascular health through increased blood flow",
          confidence: 0.7
        )

      {:ok, result} = Causal.evaluate(sno)

      assert result.score > 0.5
      assert is_list(result.issues)
      assert is_map(result.details)
    end

    test "identifies causal claims" do
      sno = SNO.new("Smoking causes lung cancer", confidence: 0.7)

      {:ok, result} = Causal.evaluate(sno)

      assert result.details.causal_type == :causal
    end

    test "identifies correlational claims" do
      sno = SNO.new("Height is correlated with income", confidence: 0.7)

      {:ok, result} = Causal.evaluate(sno)

      assert result.details.causal_type == :correlational
    end

    test "identifies descriptive claims" do
      sno = SNO.new("The sky is blue", confidence: 0.9)

      {:ok, result} = Causal.evaluate(sno)

      assert result.details.causal_type == :descriptive
    end

    test "penalizes correlation-causation confusion" do
      sno =
        SNO.new(
          "Ice cream sales are correlated with drowning deaths, therefore ice cream causes drowning",
          confidence: 0.8
        )

      {:ok, result} = Causal.evaluate(sno)

      assert result.details.confusion_score < 1.0
      assert Enum.any?(result.issues, &String.contains?(&1, "correlation_causation"))
    end

    test "rewards hedged causal claims" do
      sno =
        SNO.new(
          "The data suggests that exercise may improve mood",
          confidence: 0.6
        )

      {:ok, result} = Causal.evaluate(sno)

      assert result.details.confusion_score == 1.0
    end

    test "rewards claims with mechanism explanation" do
      sno =
        SNO.new(
          "Caffeine causes alertness through blocking adenosine receptors in the brain",
          confidence: 0.7
        )

      {:ok, result} = Causal.evaluate(sno)

      assert result.details.mechanism_score == 1.0
    end

    test "penalizes causal claims without mechanism" do
      sno =
        SNO.new(
          "Sugar causes hyperactivity in children",
          confidence: 0.7
        )

      {:ok, result} = Causal.evaluate(sno)

      assert result.details.mechanism_score < 1.0
      assert Enum.any?(result.issues, &String.contains?(&1, "no_mechanism"))
    end

    test "rewards claims with temporal ordering" do
      sno =
        SNO.new(
          "The treatment causes improvement because first it binds to receptors, then symptoms subside subsequently",
          confidence: 0.7
        )

      {:ok, result} = Causal.evaluate(sno)

      assert result.details.temporal_score == 1.0
    end

    test "penalizes causal claims without temporal ordering" do
      sno =
        SNO.new(
          "The drug causes improvement",
          confidence: 0.7
        )

      {:ok, result} = Causal.evaluate(sno)

      assert result.details.temporal_score < 1.0
      assert Enum.any?(result.issues, &String.contains?(&1, "no_temporal"))
    end

    test "penalizes overconfident causal claims" do
      sno =
        SNO.new(
          "This definitely causes that outcome",
          confidence: 0.95
        )

      {:ok, result} = Causal.evaluate(sno)

      assert result.details.confidence_score < 1.0
      assert Enum.any?(result.issues, &String.contains?(&1, "overconfident_causation"))
    end

    test "penalizes low confidence causal claims" do
      sno =
        SNO.new(
          "This causes that effect",
          confidence: 0.3
        )

      {:ok, result} = Causal.evaluate(sno)

      assert result.details.confidence_score < 1.0
      assert Enum.any?(result.issues, &String.contains?(&1, "uncertain_causation"))
    end

    test "accepts moderate confidence causal claims" do
      sno =
        SNO.new(
          "Exercise leads to improved health",
          confidence: 0.7
        )

      {:ok, result} = Causal.evaluate(sno)

      assert result.details.confidence_score == 1.0
    end

    test "handles non-causal claims well" do
      sno = SNO.new("The meeting is scheduled for Tuesday", confidence: 0.9)

      {:ok, result} = Causal.evaluate(sno)

      # Non-causal claims should have high scores since no causal issues
      assert result.score > 0.7
    end
  end

  describe "call/3 with GenServer" do
    setup do
      {:ok, pid} = Causal.start_link(name: :"causal_#{:erlang.unique_integer()}")
      %{causal: pid}
    end

    test "evaluates via GenServer", %{causal: causal} do
      sno = SNO.new("A test claim about causation", confidence: 0.7)

      {:ok, result} = Causal.call(causal, sno)

      assert result.score > 0
      assert is_map(result.details)
    end
  end

  describe "name/0" do
    test "returns :causal" do
      assert Causal.name() == :causal
    end
  end

  describe "weight/0" do
    test "returns 0.1" do
      assert Causal.weight() == 0.1
    end
  end

  describe "details" do
    test "includes all score components" do
      sno = SNO.new("Test causal claim", confidence: 0.7)

      {:ok, result} = Causal.evaluate(sno)

      assert Map.has_key?(result.details, :causal_type)
      assert Map.has_key?(result.details, :confusion_score)
      assert Map.has_key?(result.details, :mechanism_score)
      assert Map.has_key?(result.details, :temporal_score)
      assert Map.has_key?(result.details, :confidence_score)
    end
  end
end

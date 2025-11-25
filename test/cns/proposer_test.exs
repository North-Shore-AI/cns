defmodule CNS.ProposerTest do
  use ExUnit.Case, async: true

  alias CNS.{SNO, Config}
  alias CNS.Agents.Proposer

  describe "extract_claims/2" do
    test "extracts claims from text" do
      text = "Studies show coffee improves focus. Research indicates tea has health benefits."
      assert {:ok, claims} = Proposer.extract_claims(text)

      assert length(claims) >= 1
      assert Enum.all?(claims, &match?(%SNO{}, &1))
    end

    test "respects min_confidence filter" do
      text = "Maybe this could be true. Definitely this is proven."
      assert {:ok, claims} = Proposer.extract_claims(text, min_confidence: 0.6)

      # Should filter out uncertain claims
      assert Enum.all?(claims, fn c -> c.confidence >= 0.6 end)
    end

    test "respects max_claims limit" do
      text = "First. Second. Third. Fourth. Fifth. Sixth. Seventh. Eighth. Ninth. Tenth."
      assert {:ok, claims} = Proposer.extract_claims(text, max_claims: 3)

      assert length(claims) <= 3
    end

    test "handles empty text" do
      assert {:ok, []} = Proposer.extract_claims("")
    end

    test "filters questions" do
      text = "Is this true? This is definitely true."
      assert {:ok, claims} = Proposer.extract_claims(text)

      refute Enum.any?(claims, fn c -> String.ends_with?(c.claim, "?") end)
    end
  end

  describe "generate_hypothesis/2" do
    test "generates hypothesis from question" do
      assert {:ok, hypothesis} = Proposer.generate_hypothesis("Does exercise improve sleep?")

      assert is_binary(hypothesis.claim)
      assert hypothesis.provenance.origin == :proposer
      assert hypothesis.metadata.source_question == "Does exercise improve sleep?"
    end

    test "sets initial confidence" do
      assert {:ok, hypothesis} = Proposer.generate_hypothesis("Question?", initial_confidence: 0.7)
      assert hypothesis.confidence == 0.7
    end
  end

  describe "score_confidence/1" do
    test "high confidence for proven claims" do
      score = Proposer.score_confidence("Studies conclusively demonstrate the effect")
      assert score > 0.6
    end

    test "medium confidence for research claims" do
      score = Proposer.score_confidence("Research shows some evidence")
      assert score >= 0.5
    end

    test "low confidence for uncertain claims" do
      score = Proposer.score_confidence("Maybe this could possibly be true")
      assert score < 0.5
    end

    test "score is between 0 and 1" do
      texts = [
        "Conclusively proven with evidence from multiple studies",
        "Maybe perhaps possibly",
        "Normal text without markers"
      ]

      for text <- texts do
        score = Proposer.score_confidence(text)
        assert score >= 0.0 and score <= 1.0
      end
    end
  end

  describe "extract_evidence/2" do
    test "extracts citations" do
      text = "According to Smith (2023), the effect was significant."
      assert {:ok, evidence} = Proposer.extract_evidence(text)

      assert length(evidence) >= 1
      assert Enum.any?(evidence, fn e -> String.contains?(e.source, "Smith") end)
    end

    test "extracts study references" do
      text = "A study shows the results are positive."
      assert {:ok, evidence} = Proposer.extract_evidence(text)

      assert length(evidence) >= 1
    end

    test "handles text without evidence" do
      assert {:ok, []} = Proposer.extract_evidence("Just a plain statement")
    end
  end

  describe "process/2" do
    test "processes input with config" do
      config = %Config{proposer: %{min_confidence: 0.3, max_claims: 5}}
      assert {:ok, result} = Proposer.process("Test claim statement.", config)

      assert Map.has_key?(result, :claims)
      assert Map.has_key?(result, :count)
      assert Map.has_key?(result, :avg_confidence)
    end
  end
end

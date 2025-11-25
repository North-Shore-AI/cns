defmodule CNS.AntagonistTest do
  use ExUnit.Case, async: true

  alias CNS.{SNO, Challenge, Evidence, Config}
  alias CNS.Agents.Antagonist

  describe "challenge/2" do
    test "generates challenges for a claim" do
      sno = SNO.new("Coffee always improves focus", confidence: 0.8)
      assert {:ok, challenges} = Antagonist.challenge(sno)

      assert is_list(challenges)
      assert Enum.all?(challenges, &match?(%Challenge{}, &1))
    end

    test "respects max_challenges" do
      sno = SNO.new("This claim should generate many challenges", confidence: 0.9)
      assert {:ok, challenges} = Antagonist.challenge(sno, max_challenges: 2)

      assert length(challenges) <= 2
    end
  end

  describe "find_contradictions/1" do
    test "finds absolute terms" do
      sno = SNO.new("This is always true in all cases", id: "test-id")
      contradictions = Antagonist.find_contradictions(sno)

      assert length(contradictions) > 0
      assert Enum.all?(contradictions, fn c -> c.challenge_type == :contradiction end)
    end

    test "finds internal contradictions" do
      sno = SNO.new("This causes both increase and decrease", id: "test-id")
      contradictions = Antagonist.find_contradictions(sno)

      assert Enum.any?(contradictions, fn c ->
               String.contains?(c.description, "contradictory")
             end)
    end

    test "returns empty for balanced claims" do
      sno = SNO.new("This sometimes has effects", id: "test-id")
      contradictions = Antagonist.find_contradictions(sno)

      # May still return some challenges for other reasons
      contradictory =
        Enum.filter(contradictions, fn c ->
          String.contains?(c.description, "absolute term")
        end)

      assert length(contradictory) == 0
    end
  end

  describe "find_evidence_gaps/1" do
    test "flags missing evidence" do
      sno = SNO.new("Claim without evidence", id: "test-id", evidence: [])
      gaps = Antagonist.find_evidence_gaps(sno)

      assert length(gaps) > 0
      assert Enum.any?(gaps, fn c -> c.challenge_type == :evidence_gap end)
    end

    test "flags low validity evidence" do
      evidence = [Evidence.new("Weak", "Data", validity: 0.3)]
      sno = SNO.new("Claim", id: "test-id", evidence: evidence)
      gaps = Antagonist.find_evidence_gaps(sno)

      assert Enum.any?(gaps, fn c ->
               String.contains?(c.description, "low validity")
             end)
    end

    test "accepts high quality evidence" do
      evidence = [Evidence.new("Strong", "Data", validity: 0.9)]
      sno = SNO.new("Claim", id: "test-id", evidence: evidence)
      gaps = Antagonist.find_evidence_gaps(sno)

      # Should not flag high quality evidence
      low_validity_gaps =
        Enum.filter(gaps, fn c ->
          String.contains?(c.description, "low validity")
        end)

      assert length(low_validity_gaps) == 0
    end
  end

  describe "find_scope_issues/1" do
    test "flags broad generalizations" do
      sno = SNO.new("In general, this typically happens", id: "test-id")
      issues = Antagonist.find_scope_issues(sno)

      assert Enum.any?(issues, fn c -> c.challenge_type == :scope end)
    end

    test "flags short high-confidence claims" do
      sno = SNO.new("This is true", id: "test-id", confidence: 0.9)
      issues = Antagonist.find_scope_issues(sno)

      assert Enum.any?(issues, fn c ->
               String.contains?(c.description, "context")
             end)
    end
  end

  describe "find_logical_issues/1" do
    test "flags causal claims without evidence" do
      sno = SNO.new("This causes that effect", id: "test-id", evidence: [])
      issues = Antagonist.find_logical_issues(sno)

      assert Enum.any?(issues, fn c -> c.challenge_type == :logical end)
    end

    test "flags potential circular reasoning" do
      sno = SNO.new("This is true because it is true", id: "test-id")
      issues = Antagonist.find_logical_issues(sno)

      assert Enum.any?(issues, fn c ->
               String.contains?(c.description, "circular")
             end)
    end
  end

  describe "generate_alternatives/1" do
    test "suggests alternatives for high confidence claims" do
      sno = SNO.new("Very confident claim", id: "test-id", confidence: 0.9)
      alternatives = Antagonist.generate_alternatives(sno)

      assert length(alternatives) > 0
      assert Enum.all?(alternatives, fn c -> c.challenge_type == :alternative end)
    end

    test "skips low confidence claims" do
      sno = SNO.new("Uncertain claim", id: "test-id", confidence: 0.5)
      alternatives = Antagonist.generate_alternatives(sno)

      assert length(alternatives) == 0
    end
  end

  describe "score_chirality/1" do
    test "calculates average chirality score" do
      challenges = [
        Challenge.new("id", :contradiction, "Test", severity: :high, confidence: 0.9),
        Challenge.new("id", :scope, "Test", severity: :low, confidence: 0.5)
      ]

      score = Antagonist.score_chirality(challenges)
      assert score > 0
    end

    test "returns 0 for empty challenges" do
      assert Antagonist.score_chirality([]) == 0.0
    end
  end

  describe "flag_issues/1" do
    test "groups by severity" do
      challenges = [
        Challenge.new("id", :contradiction, "High", severity: :high),
        Challenge.new("id", :scope, "Low", severity: :low),
        Challenge.new("id", :evidence_gap, "Medium", severity: :medium)
      ]

      flagged = Antagonist.flag_issues(challenges)

      assert length(flagged.high) == 1
      assert length(flagged.medium) == 1
      assert length(flagged.low) == 1
    end
  end

  describe "process/2" do
    test "processes claims through antagonist" do
      claims = [
        SNO.new("First claim", id: "c1"),
        SNO.new("Second claim", id: "c2")
      ]

      config = %Config{}

      assert {:ok, result} = Antagonist.process(claims, config)

      assert Map.has_key?(result, :challenges)
      assert Map.has_key?(result, :chirality_score)
      assert Map.has_key?(result, :by_severity)
    end
  end
end

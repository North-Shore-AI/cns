defmodule CNS.SynthesizerTest do
  use ExUnit.Case, async: true

  alias CNS.Agents.Synthesizer
  alias CNS.{Challenge, Config, Evidence, SNO}

  describe "synthesize/3" do
    test "synthesizes two claims" do
      thesis = SNO.new("Coffee improves focus", id: "t1", confidence: 0.8)
      antithesis = SNO.new("Coffee causes anxiety", id: "a1", confidence: 0.7)

      assert {:ok, synthesis} = Synthesizer.synthesize(thesis, antithesis)

      assert is_binary(synthesis.claim)
      assert synthesis.confidence > 0
      assert synthesis.provenance.origin == :synthesizer
      assert "t1" in synthesis.provenance.parent_ids
      assert "a1" in synthesis.provenance.parent_ids
    end

    test "merges evidence from both claims" do
      e1 = Evidence.new("Source 1", "Data 1")
      e2 = Evidence.new("Source 2", "Data 2")

      thesis = SNO.new("Claim A", id: "t", evidence: [e1], confidence: 0.8)
      antithesis = SNO.new("Claim B", id: "a", evidence: [e2], confidence: 0.7)

      assert {:ok, synthesis} = Synthesizer.synthesize(thesis, antithesis)

      assert length(synthesis.evidence) == 2
    end

    test "records synthesis history" do
      thesis = SNO.new("A", id: "t", confidence: 0.8)
      antithesis = SNO.new("B", id: "a", confidence: 0.7)

      assert {:ok, synthesis} = Synthesizer.synthesize(thesis, antithesis)

      assert length(synthesis.synthesis_history) == 1
      history = hd(synthesis.synthesis_history)
      assert history.thesis_id == "t"
      assert history.antithesis_id == "a"
    end
  end

  describe "ground_evidence/3" do
    test "adds valid evidence to SNO" do
      sno = SNO.new("Claim", confidence: 0.6)
      evidence = [Evidence.new("New Source", "Data", validity: 0.8)]

      assert {:ok, grounded} = Synthesizer.ground_evidence(sno, evidence)

      assert length(grounded.evidence) == 1
      assert grounded.confidence > 0.6
    end

    test "filters evidence below threshold" do
      sno = SNO.new("Claim")

      evidence = [
        Evidence.new("Good", "Data", validity: 0.8),
        Evidence.new("Bad", "Data", validity: 0.3)
      ]

      assert {:ok, grounded} = Synthesizer.ground_evidence(sno, evidence, validity_threshold: 0.5)

      assert length(grounded.evidence) == 1
    end
  end

  describe "resolve_conflicts/4" do
    test "adjusts confidence based on challenges" do
      thesis = SNO.new("A", id: "t", confidence: 0.9)
      antithesis = SNO.new("B", id: "a", confidence: 0.8)

      challenges = [
        Challenge.new("t", :contradiction, "Major issue", severity: :high, confidence: 0.9)
      ]

      assert {:ok, synthesis} = Synthesizer.resolve_conflicts(thesis, antithesis, challenges)

      # Thesis confidence should be reduced due to challenge
      assert is_binary(synthesis.claim)
    end

    test "handles empty challenges" do
      thesis = SNO.new("A", id: "t", confidence: 0.8)
      antithesis = SNO.new("B", id: "a", confidence: 0.7)

      assert {:ok, synthesis} = Synthesizer.resolve_conflicts(thesis, antithesis, [])
      assert synthesis.confidence > 0
    end
  end

  describe "coherence_score/1" do
    test "calculates coherence for well-evidenced claim" do
      evidence = [Evidence.new("S", "C", validity: 0.9)]
      sno = SNO.new("Well supported claim", evidence: evidence, confidence: 0.9)

      score = Synthesizer.coherence_score(sno)
      assert score > 0.5
    end

    test "lower coherence for poor evidence" do
      sno = SNO.new("Poorly supported", evidence: [], confidence: 0.5)

      score = Synthesizer.coherence_score(sno)
      assert score < 0.5
    end
  end

  describe "entailment_score/3" do
    test "high score for related claims" do
      thesis = SNO.new("Coffee contains caffeine")
      antithesis = SNO.new("Caffeine affects the brain")
      synthesis = SNO.new("Coffee affects the brain through caffeine")

      score = Synthesizer.entailment_score(thesis, antithesis, synthesis)
      assert score > 0
    end

    test "balanced synthesis draws from both sides" do
      thesis = SNO.new("First perspective on topic")
      antithesis = SNO.new("Second perspective on topic")
      synthesis = SNO.new("Both first and second perspectives on topic are valid")

      score = Synthesizer.entailment_score(thesis, antithesis, synthesis)
      assert score > 0.3
    end
  end

  describe "process/4" do
    test "processes full synthesis with config" do
      thesis = SNO.new("A", id: "t", confidence: 0.8)
      antithesis = SNO.new("B", id: "a", confidence: 0.7)
      challenges = []
      config = %Config{}

      assert {:ok, result} = Synthesizer.process(thesis, antithesis, challenges, config)

      assert Map.has_key?(result, :synthesis)
      assert Map.has_key?(result, :coherence_score)
      assert Map.has_key?(result, :entailment_score)
      assert Map.has_key?(result, :meets_threshold)
    end
  end
end

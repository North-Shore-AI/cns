defmodule CNS.SNOTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias CNS.{SNO, Evidence, Provenance}

  doctest CNS.SNO

  describe "new/2" do
    test "creates SNO with defaults" do
      sno = SNO.new("Test claim")

      assert sno.claim == "Test claim"
      assert sno.confidence == 0.5
      assert sno.evidence == []
      assert sno.metadata == %{}
      assert is_binary(sno.id)
    end

    test "creates SNO with custom options" do
      evidence = [Evidence.new("Source", "Content")]
      sno = SNO.new("Claim", confidence: 0.9, evidence: evidence)

      assert sno.confidence == 0.9
      assert length(sno.evidence) == 1
    end

    test "creates SNO with provenance" do
      prov = Provenance.new(:proposer)
      sno = SNO.new("Claim", provenance: prov)

      assert sno.provenance.origin == :proposer
    end
  end

  describe "validate/1" do
    test "validates correct SNO" do
      sno = SNO.new("Valid claim")
      assert {:ok, ^sno} = SNO.validate(sno)
    end

    test "rejects empty claim" do
      sno = %SNO{claim: "", confidence: 0.5}
      assert {:error, errors} = SNO.validate(sno)
      assert "claim must be a non-empty string" in errors
    end

    test "rejects invalid confidence" do
      sno = %SNO{claim: "Test", confidence: 1.5}
      assert {:error, errors} = SNO.validate(sno)
      assert "confidence must be a number between 0.0 and 1.0" in errors
    end

    test "rejects non-list evidence" do
      sno = %SNO{claim: "Test", confidence: 0.5, evidence: "not a list"}
      assert {:error, errors} = SNO.validate(sno)
      assert "evidence must be a list" in errors
    end
  end

  describe "to_map/1 and from_map/1" do
    test "round-trips SNO" do
      evidence = [Evidence.new("Source", "Content")]
      prov = Provenance.new(:proposer)

      sno =
        SNO.new("Test claim",
          confidence: 0.8,
          evidence: evidence,
          provenance: prov,
          metadata: %{key: "value"}
        )

      map = SNO.to_map(sno)
      assert {:ok, restored} = SNO.from_map(map)

      assert restored.claim == sno.claim
      assert restored.confidence == sno.confidence
      assert length(restored.evidence) == 1
    end

    test "handles nested children" do
      child = SNO.new("Child claim")
      parent = SNO.new("Parent claim", children: [child])

      map = SNO.to_map(parent)
      assert {:ok, restored} = SNO.from_map(map)

      assert length(restored.children) == 1
      assert hd(restored.children).claim == "Child claim"
    end
  end

  describe "to_json/1 and from_json/1" do
    test "serializes to JSON" do
      sno = SNO.new("Test", confidence: 0.8)
      assert {:ok, json} = SNO.to_json(sno)
      assert is_binary(json)
      assert String.contains?(json, "Test")
    end

    test "deserializes from JSON" do
      json = ~s({"claim": "From JSON", "confidence": 0.7})
      assert {:ok, sno} = SNO.from_json(json)
      assert sno.claim == "From JSON"
      assert sno.confidence == 0.7
    end
  end

  describe "add_evidence/2" do
    test "adds evidence to SNO" do
      sno = SNO.new("Claim")
      evidence = Evidence.new("Source", "Content")

      updated = SNO.add_evidence(sno, evidence)
      assert length(updated.evidence) == 1
    end
  end

  describe "update_confidence/2" do
    test "updates confidence" do
      sno = SNO.new("Claim", confidence: 0.5)
      updated = SNO.update_confidence(sno, 0.9)
      assert updated.confidence == 0.9
    end
  end

  describe "evidence_score/1" do
    test "returns 0 for empty evidence" do
      sno = SNO.new("Claim")
      assert SNO.evidence_score(sno) == 0.0
    end

    test "calculates average validity" do
      e1 = Evidence.new("S1", "C1", validity: 0.8)
      e2 = Evidence.new("S2", "C2", validity: 0.6)
      sno = SNO.new("Claim", evidence: [e1, e2])

      assert SNO.evidence_score(sno) == 0.7
    end
  end

  describe "quality_score/1" do
    test "combines confidence and evidence score" do
      e = Evidence.new("S", "C", validity: 0.8)
      sno = SNO.new("Claim", evidence: [e], confidence: 0.9)

      score = SNO.quality_score(sno)
      assert score == 0.72
    end

    test "handles empty evidence" do
      sno = SNO.new("Claim", confidence: 0.8)
      score = SNO.quality_score(sno)
      assert score == 0.4
    end
  end

  describe "meets_threshold?/2" do
    test "returns true when meets threshold" do
      e = Evidence.new("S", "C", validity: 0.9)
      sno = SNO.new("Claim", evidence: [e], confidence: 0.9)
      assert SNO.meets_threshold?(sno, 0.5)
    end

    test "returns false when below threshold" do
      sno = SNO.new("Claim", confidence: 0.3)
      refute SNO.meets_threshold?(sno, 0.5)
    end
  end

  describe "word_count/1" do
    test "counts words in claim" do
      sno = SNO.new("This is a test claim")
      assert SNO.word_count(sno) == 5
    end

    test "handles multiple spaces" do
      sno = SNO.new("Words   with   spaces")
      assert SNO.word_count(sno) == 3
    end
  end

  describe "property tests" do
    property "confidence is always clamped to 0-1" do
      check all(conf <- float(min: 0.0, max: 1.0)) do
        sno = SNO.new("Test", confidence: conf)
        {:ok, _} = SNO.validate(sno)
      end
    end

    property "evidence score is average of validities" do
      check all(validities <- list_of(float(min: 0.0, max: 1.0), min_length: 1, max_length: 5)) do
        evidence = Enum.map(validities, &Evidence.new("S", "C", validity: &1))
        sno = SNO.new("Test", evidence: evidence)

        expected = Enum.sum(validities) / length(validities)
        assert_in_delta SNO.evidence_score(sno), expected, 0.0001
      end
    end
  end
end

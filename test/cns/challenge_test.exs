defmodule CNS.ChallengeTest do
  use ExUnit.Case, async: true

  alias CNS.{Challenge, Evidence}

  doctest CNS.Challenge

  describe "new/4" do
    test "creates challenge with defaults" do
      challenge = Challenge.new("target-1", :contradiction, "Test description")

      assert challenge.target_id == "target-1"
      assert challenge.challenge_type == :contradiction
      assert challenge.description == "Test description"
      assert challenge.severity == :medium
      assert challenge.confidence == 0.5
      assert challenge.resolution == :pending
    end

    test "creates challenge with custom options" do
      evidence = [Evidence.new("Counter source", "Data")]

      challenge =
        Challenge.new("target-1", :evidence_gap, "Missing evidence",
          counter_evidence: evidence,
          severity: :high,
          confidence: 0.9
        )

      assert challenge.severity == :high
      assert challenge.confidence == 0.9
      assert length(challenge.counter_evidence) == 1
    end
  end

  describe "validate/1" do
    test "validates correct challenge" do
      challenge = Challenge.new("t1", :contradiction, "Test")
      assert {:ok, ^challenge} = Challenge.validate(challenge)
    end

    test "validates all challenge types" do
      for type <- [:contradiction, :evidence_gap, :scope, :logical, :alternative] do
        challenge = Challenge.new("t1", type, "Test")
        assert {:ok, _} = Challenge.validate(challenge)
      end
    end

    test "validates all severity levels" do
      for severity <- [:high, :medium, :low] do
        challenge = Challenge.new("t1", :contradiction, "Test", severity: severity)
        assert {:ok, _} = Challenge.validate(challenge)
      end
    end

    test "rejects empty target_id" do
      challenge = %Challenge{target_id: "", challenge_type: :contradiction, description: "Test"}
      assert {:error, errors} = Challenge.validate(challenge)
      assert "target_id must be a non-empty string" in errors
    end

    test "rejects invalid challenge_type" do
      challenge = %Challenge{target_id: "t1", challenge_type: :invalid, description: "Test"}
      assert {:error, errors} = Challenge.validate(challenge)
      assert Enum.any?(errors, &String.contains?(&1, "challenge_type"))
    end
  end

  describe "to_map/1 and from_map/1" do
    test "round-trips challenge" do
      challenge =
        Challenge.new("t1", :evidence_gap, "Test",
          severity: :high,
          confidence: 0.8
        )

      map = Challenge.to_map(challenge)
      assert {:ok, restored} = Challenge.from_map(map)

      assert restored.target_id == challenge.target_id
      assert restored.challenge_type == challenge.challenge_type
      assert restored.severity == challenge.severity
      assert restored.confidence == challenge.confidence
    end
  end

  describe "chirality_score/1" do
    test "high severity contradiction has high score" do
      challenge =
        Challenge.new("t1", :contradiction, "Test",
          severity: :high,
          confidence: 0.9
        )

      score = Challenge.chirality_score(challenge)
      assert score > 0.7
    end

    test "low severity alternative has low score" do
      challenge =
        Challenge.new("t1", :alternative, "Test",
          severity: :low,
          confidence: 0.5
        )

      score = Challenge.chirality_score(challenge)
      assert score < 0.2
    end

    test "score is weighted by type and severity" do
      high = Challenge.new("t1", :contradiction, "Test", severity: :high, confidence: 1.0)
      low = Challenge.new("t1", :scope, "Test", severity: :low, confidence: 1.0)

      assert Challenge.chirality_score(high) > Challenge.chirality_score(low)
    end
  end

  describe "critical?/1" do
    test "returns true for high severity high confidence" do
      challenge =
        Challenge.new("t1", :contradiction, "Test",
          severity: :high,
          confidence: 0.8
        )

      assert Challenge.critical?(challenge)
    end

    test "returns false for medium severity" do
      challenge =
        Challenge.new("t1", :contradiction, "Test",
          severity: :medium,
          confidence: 0.9
        )

      refute Challenge.critical?(challenge)
    end

    test "returns false for low confidence" do
      challenge =
        Challenge.new("t1", :contradiction, "Test",
          severity: :high,
          confidence: 0.5
        )

      refute Challenge.critical?(challenge)
    end
  end

  describe "resolve/2" do
    test "sets resolution status" do
      challenge = Challenge.new("t1", :contradiction, "Test")
      resolved = Challenge.resolve(challenge, :accepted)

      assert resolved.resolution == :accepted
    end

    test "accepts all resolution types" do
      challenge = Challenge.new("t1", :contradiction, "Test")

      for resolution <- [:accepted, :rejected, :partial, :pending] do
        resolved = Challenge.resolve(challenge, resolution)
        assert resolved.resolution == resolution
      end
    end
  end

  describe "pending?/1" do
    test "returns true for pending resolution" do
      challenge = Challenge.new("t1", :contradiction, "Test")
      assert Challenge.pending?(challenge)
    end

    test "returns false for resolved" do
      challenge = Challenge.new("t1", :contradiction, "Test")
      resolved = Challenge.resolve(challenge, :accepted)
      refute Challenge.pending?(resolved)
    end
  end
end

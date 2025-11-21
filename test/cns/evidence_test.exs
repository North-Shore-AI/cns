defmodule CNS.EvidenceTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias CNS.Evidence

  doctest CNS.Evidence

  describe "new/3" do
    test "creates evidence with defaults" do
      evidence = Evidence.new("Test Source", "Content")

      assert evidence.source == "Test Source"
      assert evidence.content == "Content"
      assert evidence.validity == 1.0
      assert evidence.relevance == 1.0
      assert evidence.retrieval_method == :manual
      assert is_binary(evidence.id)
    end

    test "creates evidence with custom options" do
      evidence =
        Evidence.new("Source", "Content",
          validity: 0.8,
          relevance: 0.7,
          retrieval_method: :search
        )

      assert evidence.validity == 0.8
      assert evidence.relevance == 0.7
      assert evidence.retrieval_method == :search
    end

    test "creates evidence with custom id" do
      evidence = Evidence.new("Source", "Content", id: "custom-id")
      assert evidence.id == "custom-id"
    end
  end

  describe "validate/1" do
    test "validates correct evidence" do
      evidence = Evidence.new("Valid Source", "Content")
      assert {:ok, ^evidence} = Evidence.validate(evidence)
    end

    test "rejects empty source" do
      evidence = %Evidence{source: "", validity: 0.5}
      assert {:error, errors} = Evidence.validate(evidence)
      assert "source must be a non-empty string" in errors
    end

    test "rejects invalid validity" do
      evidence = %Evidence{source: "Test", validity: 1.5}
      assert {:error, errors} = Evidence.validate(evidence)
      assert "validity must be a number between 0.0 and 1.0" in errors
    end

    test "rejects invalid relevance" do
      evidence = %Evidence{source: "Test", validity: 0.5, relevance: -0.1}
      assert {:error, errors} = Evidence.validate(evidence)
      assert "relevance must be a number between 0.0 and 1.0" in errors
    end

    test "rejects invalid retrieval_method" do
      evidence = %Evidence{
        source: "Test",
        validity: 0.5,
        relevance: 0.5,
        retrieval_method: :invalid
      }

      assert {:error, errors} = Evidence.validate(evidence)
      assert "retrieval_method must be one of :manual, :search, :citation, :inference" in errors
    end
  end

  describe "to_map/1 and from_map/1" do
    test "round-trips evidence" do
      evidence =
        Evidence.new("Source", "Content",
          validity: 0.8,
          relevance: 0.6,
          retrieval_method: :citation
        )

      map = Evidence.to_map(evidence)
      assert {:ok, restored} = Evidence.from_map(map)

      assert restored.source == evidence.source
      assert restored.content == evidence.content
      assert restored.validity == evidence.validity
      assert restored.relevance == evidence.relevance
    end

    test "handles string keys" do
      map = %{
        "source" => "Test",
        "content" => "Data",
        "validity" => 0.9
      }

      assert {:ok, evidence} = Evidence.from_map(map)
      assert evidence.source == "Test"
      assert evidence.validity == 0.9
    end
  end

  describe "score/1" do
    test "calculates combined score" do
      evidence = Evidence.new("S", "C", validity: 0.8, relevance: 0.5)
      assert Evidence.score(evidence) == 0.4
    end

    test "returns 1.0 for perfect evidence" do
      evidence = Evidence.new("S", "C", validity: 1.0, relevance: 1.0)
      assert Evidence.score(evidence) == 1.0
    end
  end

  describe "meets_threshold?/2" do
    test "returns true when score meets threshold" do
      evidence = Evidence.new("S", "C", validity: 0.9, relevance: 0.9)
      assert Evidence.meets_threshold?(evidence, 0.5)
    end

    test "returns false when score below threshold" do
      evidence = Evidence.new("S", "C", validity: 0.5, relevance: 0.5)
      refute Evidence.meets_threshold?(evidence, 0.5)
    end
  end

  describe "property tests" do
    property "validity is always between 0 and 1" do
      check all(
              validity <- float(min: 0.0, max: 1.0),
              relevance <- float(min: 0.0, max: 1.0)
            ) do
        evidence =
          Evidence.new("Source", "Content",
            validity: validity,
            relevance: relevance
          )

        {:ok, _} = Evidence.validate(evidence)
      end
    end

    property "score is product of validity and relevance" do
      check all(
              validity <- float(min: 0.0, max: 1.0),
              relevance <- float(min: 0.0, max: 1.0)
            ) do
        evidence =
          Evidence.new("S", "C",
            validity: validity,
            relevance: relevance
          )

        expected = Float.round(validity * relevance, 4)
        assert Evidence.score(evidence) == expected
      end
    end
  end
end

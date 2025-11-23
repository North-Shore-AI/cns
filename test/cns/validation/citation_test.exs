defmodule CNS.Validation.CitationTest do
  use ExUnit.Case, async: true

  alias CNS.Validation.Citation

  describe "extract_document_ids/1" do
    test "extracts document IDs from text with CLAIM format" do
      text = "CLAIM[c1] (Document 12345): Some claim text"
      ids = Citation.extract_document_ids(text)

      assert "12345" in ids
    end

    test "extracts multiple document IDs" do
      text = """
      CLAIM[c1] (Document 12345): First claim
      CLAIM[c2] (Document 67890): Second claim
      """

      ids = Citation.extract_document_ids(text)

      assert "12345" in ids
      assert "67890" in ids
    end

    test "returns empty list when no document IDs" do
      text = "CLAIM[c1]: Some claim without document ID"
      ids = Citation.extract_document_ids(text)

      assert ids == []
    end
  end

  describe "validate/2" do
    test "passes when all cited documents exist in corpus" do
      claim_text = "CLAIM[c1] (Document 12345): Some claim"
      corpus = MapSet.new(["12345", "67890"])

      result = Citation.validate(claim_text, corpus)

      assert result.valid == true
      assert result.missing == []
    end

    test "fails when cited document not in corpus" do
      claim_text = "CLAIM[c1] (Document 99999): Some claim"
      corpus = MapSet.new(["12345", "67890"])

      result = Citation.validate(claim_text, corpus)

      assert result.valid == false
      assert "99999" in result.missing
    end

    test "reports all missing documents" do
      claim_text = """
      CLAIM[c1] (Document 11111): Claim 1
      CLAIM[c2] (Document 22222): Claim 2
      """

      corpus = MapSet.new(["12345"])

      result = Citation.validate(claim_text, corpus)

      assert result.valid == false
      assert length(result.missing) == 2
    end
  end

  describe "check_hallucination/2" do
    test "detects potential hallucination" do
      claim_text = "CLAIM[c1] (Document 99999): Unverified claim"
      corpus = MapSet.new(["12345"])

      result = Citation.check_hallucination(claim_text, corpus)

      assert result.hallucination_detected == true
    end

    test "no hallucination when documents exist" do
      claim_text = "CLAIM[c1] (Document 12345): Verified claim"
      corpus = MapSet.new(["12345"])

      result = Citation.check_hallucination(claim_text, corpus)

      assert result.hallucination_detected == false
    end
  end
end

defmodule CNS.Validation.SemanticTest do
  use ExUnit.Case, async: true
  alias CNS.Validation.Semantic
  alias CNS.Validation.Semantic.{Config, ValidationResult}

  describe "ValidationResult struct" do
    test "can be constructed with all fields" do
      result = %ValidationResult{
        citation_valid: true,
        cited_ids: MapSet.new(["123", "456"]),
        missing_ids: MapSet.new(),
        entailment_score: 0.85,
        entailment_pass: true,
        semantic_similarity: 0.75,
        similarity_pass: true,
        paraphrase_accepted: true,
        overall_pass: true,
        schema_valid: true,
        schema_errors: []
      }

      assert result.citation_valid == true
      assert result.overall_pass == true
    end
  end

  describe "Config struct" do
    test "has default values" do
      config = %Config{}

      assert config.entailment_threshold == 0.75
      assert config.similarity_threshold == 0.7
    end

    test "can override defaults" do
      config = %Config{entailment_threshold: 0.8, similarity_threshold: 0.6}

      assert config.entailment_threshold == 0.8
      assert config.similarity_threshold == 0.6
    end
  end

  describe "extract_document_ids/1" do
    test "extracts numeric document IDs" do
      text = "Evidence from Document 12345 and Document 67890 supports the claim."
      ids = Semantic.extract_document_ids(text)

      assert MapSet.member?(ids, "12345")
      assert MapSet.member?(ids, "67890")
    end

    test "extracts bracketed references" do
      text = "The data [DocID: abc123] shows that [ref:xyz789] is correct."
      ids = Semantic.extract_document_ids(text)

      assert MapSet.member?(ids, "abc123")
      assert MapSet.member?(ids, "xyz789")
    end

    test "handles CLAIM format document references" do
      text = "CLAIM[c2] (Document 12345): Some evidence text"
      ids = Semantic.extract_document_ids(text)

      assert MapSet.member?(ids, "12345")
    end

    test "returns empty set for no references" do
      text = "No document references here."
      ids = Semantic.extract_document_ids(text)

      assert MapSet.size(ids) == 0
    end

    test "deduplicates repeated IDs" do
      text = "Document 123 and again Document 123"
      ids = Semantic.extract_document_ids(text)

      assert MapSet.size(ids) == 1
    end
  end

  describe "validate_citations/3" do
    test "passes when all cited docs exist in corpus" do
      text = "Evidence from Document 123 and Document 456"
      corpus = %{"123" => %{}, "456" => %{}, "789" => %{}}
      gold_ids = MapSet.new(["123", "456"])

      {valid, cited, missing} = Semantic.validate_citations(text, corpus, gold_ids)

      assert valid == true
      assert MapSet.size(cited) == 2
      assert MapSet.size(missing) == 0
    end

    test "fails when gold evidence is missing" do
      text = "Evidence from Document 123"
      corpus = %{"123" => %{}, "456" => %{}}
      gold_ids = MapSet.new(["123", "456"])

      {valid, _cited, missing} = Semantic.validate_citations(text, corpus, gold_ids)

      assert valid == false
      assert MapSet.member?(missing, "456")
    end

    test "filters out non-existent corpus docs" do
      text = "Document 123 and Document 999"
      corpus = %{"123" => %{}}
      gold_ids = MapSet.new(["123"])

      {valid, cited, _missing} = Semantic.validate_citations(text, corpus, gold_ids)

      assert valid == true
      assert MapSet.size(cited) == 1
      refute MapSet.member?(cited, "999")
    end
  end

  describe "compute_similarity/2" do
    test "identical texts have similarity 1.0" do
      text = "The quick brown fox"
      similarity = Semantic.compute_similarity(text, text)

      assert_in_delta similarity, 1.0, 0.01
    end

    test "different texts have lower similarity" do
      text_a = "The quick brown fox jumps over the lazy dog"
      text_b = "A slow red cat sleeps under the active cat"

      similarity = Semantic.compute_similarity(text_a, text_b)

      assert similarity < 1.0
      assert similarity >= 0.0
    end

    test "empty strings have similarity 0.0" do
      similarity = Semantic.compute_similarity("", "")

      assert similarity == 0.0
    end
  end

  describe "validate_claim/5" do
    setup do
      config = %Config{
        entailment_threshold: 0.75,
        similarity_threshold: 0.7
      }

      corpus = %{
        "123" => %{"text" => "Supporting evidence text"},
        "456" => %{"text" => "More evidence"}
      }

      {:ok, config: config, corpus: corpus}
    end

    test "passes for valid claim with matching evidence", %{config: config, corpus: corpus} do
      generated = "The evidence supports the claim"
      gold = "The evidence supports the claim"
      full_output = "CLAIM[c1]: Main claim\nCLAIM[c2] (Document 123): Evidence"
      gold_ids = MapSet.new(["123"])

      result = Semantic.validate_claim(config, generated, gold, full_output, corpus, gold_ids)

      assert result.citation_valid == true
      assert result.similarity_pass == true
    end

    test "fails for missing citations", %{config: config, corpus: corpus} do
      generated = "The evidence supports the claim"
      gold = "The evidence supports the claim"
      full_output = "CLAIM[c1]: Main claim"
      gold_ids = MapSet.new(["123"])

      result = Semantic.validate_claim(config, generated, gold, full_output, corpus, gold_ids)

      assert result.citation_valid == false
      assert result.overall_pass == false
    end

    test "fails for low similarity", %{config: config, corpus: corpus} do
      generated = "Completely different content about weather"
      gold = "The evidence supports the scientific claim"
      full_output = "CLAIM[c1]: Main\nCLAIM[c2] (Document 123): Evidence"
      gold_ids = MapSet.new(["123"])

      result = Semantic.validate_claim(config, generated, gold, full_output, corpus, gold_ids)

      assert result.similarity_pass == false
    end
  end

  describe "failed_result/3" do
    test "creates result with all stages failed" do
      cited = MapSet.new(["123"])
      missing = MapSet.new(["456"])

      result = Semantic.failed_result(false, cited, missing)

      assert result.citation_valid == false
      assert result.entailment_pass == false
      assert result.similarity_pass == false
      assert result.paraphrase_accepted == false
      assert result.overall_pass == false
    end
  end
end

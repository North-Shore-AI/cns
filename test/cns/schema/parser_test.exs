defmodule CNS.Schema.ParserTest do
  use ExUnit.Case, async: true
  alias CNS.Schema.Parser

  describe "parse_claims/1" do
    test "parses basic claim" do
      text = "CLAIM[c1]: The hypothesis text"
      claims = Parser.parse_claims(text)

      assert Map.has_key?(claims, "c1")
      assert claims["c1"].text == "The hypothesis text"
      assert claims["c1"].document_id == nil
    end

    test "parses claim with document reference" do
      text = "CLAIM[c2] (Document 12345): Supporting claim"
      claims = Parser.parse_claims(text)

      assert claims["c2"].text == "Supporting claim"
      assert claims["c2"].document_id == "12345"
    end

    test "parses multiple claims" do
      text = """
      CLAIM[c1]: First claim
      Some other text
      CLAIM[c2] (Document 123): Second claim
      CLAIM[c3]: Third claim
      """

      claims = Parser.parse_claims(text)

      assert map_size(claims) == 3
      assert Map.has_key?(claims, "c1")
      assert Map.has_key?(claims, "c2")
      assert Map.has_key?(claims, "c3")
    end

    test "case insensitive CLAIM keyword" do
      text = "claim[c1]: lowercase\nCLAIM[c2]: uppercase\nClaim[c3]: mixed"
      claims = Parser.parse_claims(text)

      assert map_size(claims) == 3
    end

    test "handles complex claim IDs" do
      text = "CLAIM[claim_1_final]: Test"
      claims = Parser.parse_claims(text)

      assert Map.has_key?(claims, "claim_1_final")
    end

    test "trims whitespace" do
      text = "CLAIM[c1]  :   Text with spaces   "
      claims = Parser.parse_claims(text)

      assert claims["c1"].text == "Text with spaces"
    end

    test "empty input returns empty map" do
      assert Parser.parse_claims("") == %{}
      assert Parser.parse_claims([]) == %{}
    end
  end

  describe "parse_relation/1" do
    test "parses relation with colon" do
      result = Parser.parse_relation("RELATION: c2 supports c1")

      assert result == {"c2", "supports", "c1"}
    end

    test "parses relation with dash" do
      result = Parser.parse_relation("RELATION - c3 refutes c1")

      assert result == {"c3", "refutes", "c1"}
    end

    test "parses contrasts relation" do
      result = Parser.parse_relation("RELATION: c4 contrasts c2")

      assert result == {"c4", "contrasts", "c2"}
    end

    test "case insensitive keywords" do
      result = Parser.parse_relation("relation: c1 SUPPORTS c2")

      assert result == {"c1", "supports", "c2"}
    end

    test "returns nil for non-relation lines" do
      assert Parser.parse_relation("CLAIM[c1]: text") == nil
      assert Parser.parse_relation("just some text") == nil
      assert Parser.parse_relation("") == nil
    end

    test "handles various whitespace" do
      result = Parser.parse_relation("RELATION:   c1   supports   c2")

      assert result == {"c1", "supports", "c2"}
    end
  end

  describe "parse_relations/1" do
    test "parses multiple relations" do
      text = """
      RELATION: c2 supports c1
      Some other text
      RELATION - c3 refutes c1
      """

      relations = Parser.parse_relations(text)

      assert length(relations) == 2
      assert {"c2", "supports", "c1"} in relations
      assert {"c3", "refutes", "c1"} in relations
    end
  end

  describe "parse/1" do
    test "parses complete output" do
      text = """
      CLAIM[c1]: Main hypothesis
      CLAIM[c2] (Document 123): Supporting evidence
      CLAIM[c3] (Document 456): Counter evidence
      RELATION: c2 supports c1
      RELATION: c3 refutes c1
      """

      {claims, relations} = Parser.parse(text)

      assert map_size(claims) == 3
      assert length(relations) == 2
    end
  end
end

defmodule CNS.Pipeline.ConvertersTest do
  use ExUnit.Case, async: true
  alias CNS.Pipeline.Converters
  alias CNS.Pipeline.Schema.TrainingExample

  describe "build_prompt/1" do
    test "creates extraction prompt with passage" do
      passage = "Evidence sentence from document."
      prompt = Converters.build_prompt(passage)

      assert prompt =~ "Extract"
      assert prompt =~ passage
      assert prompt =~ "CLAIM"
      assert prompt =~ "RELATION"
    end
  end

  describe "build_completion/2" do
    test "formats claim and evidence as completion" do
      claim_text = "Main hypothesis"

      evidence = [
        {"Supporting evidence", "supports", "[123:0]"},
        {"Counter evidence", "refutes", "[456:1]"}
      ]

      completion = Converters.build_completion(claim_text, evidence)

      assert completion =~ "CLAIM[c1]: Main hypothesis"
      assert completion =~ "CLAIM[c2]: Supporting evidence [123:0]"
      assert completion =~ "CLAIM[c3]: Counter evidence [456:1]"
      assert completion =~ "RELATION: c2 supports c1"
      assert completion =~ "RELATION: c3 refutes c1"
    end

    test "handles empty evidence" do
      completion = Converters.build_completion("Claim", [])

      assert completion =~ "CLAIM[c1]: Claim"
      refute completion =~ "c2"
    end
  end

  describe "normalize_label/1" do
    test "normalizes SUPPORTS variants" do
      assert Converters.normalize_label("SUPPORTS") == "supports"
      assert Converters.normalize_label("SUPPORT") == "supports"
      assert Converters.normalize_label("supports") == "supports"
    end

    test "normalizes REFUTES variants" do
      assert Converters.normalize_label("REFUTES") == "refutes"
      assert Converters.normalize_label("CONTRADICT") == "refutes"
      assert Converters.normalize_label("refutes") == "refutes"
    end

    test "lowercases unknown labels" do
      assert Converters.normalize_label("NEUTRAL") == "neutral"
    end
  end

  describe "parse_scifact_entry/2" do
    test "converts scifact entry to training example" do
      entry = %{
        "id" => 1,
        "claim" => "Test claim",
        "evidence" => %{
          "123" => [
            %{"label" => "SUPPORTS", "sentences" => [0, 1]}
          ]
        }
      }

      corpus = %{
        "123" => %{sentences: ["First sentence.", "Second sentence."]}
      }

      example = Converters.parse_scifact_entry(entry, corpus)

      assert %TrainingExample{} = example
      assert example.completion =~ "Test claim"
      assert example.metadata.source == "scifact"
    end

    test "handles entry with no evidence" do
      entry = %{
        "id" => 2,
        "claim" => "Claim without evidence",
        "evidence" => %{}
      }

      example = Converters.parse_scifact_entry(entry, %{})

      assert example.completion =~ "Claim without evidence"
    end
  end

  describe "gather_evidence/2" do
    test "collects evidence sentences with labels" do
      entry = %{
        "evidence" => %{
          "doc1" => [
            %{"label" => "SUPPORTS", "sentences" => [0]},
            %{"label" => "REFUTES", "sentences" => [1]}
          ]
        }
      }

      corpus = %{
        "doc1" => %{sentences: ["Support text", "Refute text"]}
      }

      evidence = Converters.gather_evidence(entry, corpus)

      assert length(evidence) == 2
      assert {"Support text", "supports", _} = Enum.at(evidence, 0)
      assert {"Refute text", "refutes", _} = Enum.at(evidence, 1)
    end

    test "returns empty list for missing corpus entry" do
      entry = %{
        "evidence" => %{
          "missing" => [%{"label" => "SUPPORTS", "sentences" => [0]}]
        }
      }

      evidence = Converters.gather_evidence(entry, %{})

      assert evidence == []
    end
  end

  describe "has_evidence?/1" do
    test "returns true for entry with evidence" do
      entry = %{"evidence" => %{"123" => [%{}]}}
      assert Converters.has_evidence?(entry) == true
    end

    test "returns false for empty evidence" do
      entry = %{"evidence" => %{}}
      assert Converters.has_evidence?(entry) == false
    end

    test "returns false for no evidence key" do
      entry = %{}
      assert Converters.has_evidence?(entry) == false
    end
  end
end

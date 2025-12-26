defmodule CNS.Pipeline.SchemaTest do
  use ExUnit.Case, async: true
  alias CNS.Pipeline.Schema.{ClaimEntry, Lineage, TrainingExample}

  describe "TrainingExample struct" do
    test "can be constructed with required fields" do
      example = %TrainingExample{
        prompt: "Extract claims from this passage",
        completion: "CLAIM[c1]: Main claim",
        metadata: %{source: "scifact", claim_id: "123"}
      }

      assert example.prompt =~ "Extract"
      assert example.completion =~ "CLAIM"
      assert example.metadata.source == "scifact"
    end

    test "to_json/1 produces valid JSON" do
      example = %TrainingExample{
        prompt: "Test prompt",
        completion: "Test completion",
        metadata: %{source: "test"}
      }

      json = TrainingExample.to_json(example)
      assert is_binary(json)

      decoded = Jason.decode!(json)
      assert decoded["prompt"] == "Test prompt"
      assert decoded["completion"] == "Test completion"
    end

    test "from_json/1 parses JSON string" do
      json = ~s({"prompt": "Test", "completion": "Output", "metadata": {"source": "test"}})
      {:ok, example} = TrainingExample.from_json(json)

      assert example.prompt == "Test"
      assert example.completion == "Output"
    end
  end

  describe "ClaimEntry struct" do
    test "can be constructed" do
      entry = %ClaimEntry{
        id: "c1",
        text: "Main claim text",
        evidence_ids: ["123", "456"],
        label: "SUPPORTS"
      }

      assert entry.id == "c1"
      assert length(entry.evidence_ids) == 2
    end
  end

  describe "Lineage struct" do
    test "tracks data transformations" do
      lineage = %Lineage{
        source_file: "claims.jsonl",
        timestamp: DateTime.utc_now(),
        transformations: ["load", "filter", "convert"],
        hash: "abc123"
      }

      assert lineage.source_file == "claims.jsonl"
      assert length(lineage.transformations) == 3
    end

    test "add_transformation/2 appends to history" do
      lineage = %Lineage{
        source_file: "test.jsonl",
        timestamp: DateTime.utc_now(),
        transformations: ["load"],
        hash: ""
      }

      updated = Lineage.add_transformation(lineage, "filter")

      assert length(updated.transformations) == 2
      assert "filter" in updated.transformations
    end
  end
end

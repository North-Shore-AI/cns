defmodule CNS.ProvenanceTest do
  use ExUnit.Case, async: true

  alias CNS.Provenance

  doctest CNS.Provenance

  describe "new/2" do
    test "creates provenance with defaults" do
      prov = Provenance.new(:proposer)

      assert prov.origin == :proposer
      assert prov.parent_ids == []
      assert prov.transformation == ""
      assert prov.iteration == 0
    end

    test "creates provenance with options" do
      prov =
        Provenance.new(:synthesizer,
          parent_ids: ["id1", "id2"],
          transformation: "synthesis",
          model_id: "gpt-4",
          iteration: 2
        )

      assert prov.origin == :synthesizer
      assert length(prov.parent_ids) == 2
      assert prov.transformation == "synthesis"
      assert prov.model_id == "gpt-4"
      assert prov.iteration == 2
    end
  end

  describe "validate/1" do
    test "validates correct provenance" do
      prov = Provenance.new(:proposer)
      assert {:ok, ^prov} = Provenance.validate(prov)
    end

    test "validates all origin types" do
      for origin <- [:proposer, :antagonist, :synthesizer, :external] do
        prov = Provenance.new(origin)
        assert {:ok, _} = Provenance.validate(prov)
      end
    end

    test "rejects invalid origin" do
      prov = %Provenance{origin: :invalid}
      assert {:error, errors} = Provenance.validate(prov)
      assert Enum.any?(errors, &String.contains?(&1, "origin"))
    end

    test "rejects non-string parent_ids" do
      prov = %Provenance{origin: :proposer, parent_ids: [1, 2]}
      assert {:error, errors} = Provenance.validate(prov)
      assert "all parent_ids must be strings" in errors
    end

    test "rejects negative iteration" do
      prov = %Provenance{origin: :proposer, parent_ids: [], iteration: -1}
      assert {:error, errors} = Provenance.validate(prov)
      assert "iteration must be a non-negative integer" in errors
    end
  end

  describe "to_map/1 and from_map/1" do
    test "round-trips provenance" do
      prov =
        Provenance.new(:synthesizer,
          parent_ids: ["p1", "p2"],
          transformation: "merge",
          model_id: "test-model"
        )

      map = Provenance.to_map(prov)
      assert {:ok, restored} = Provenance.from_map(map)

      assert restored.origin == prov.origin
      assert restored.parent_ids == prov.parent_ids
      assert restored.transformation == prov.transformation
      assert restored.model_id == prov.model_id
    end

    test "handles string keys" do
      map = %{
        "origin" => "proposer",
        "parent_ids" => ["a", "b"],
        "transformation" => "test"
      }

      assert {:ok, prov} = Provenance.from_map(map)
      assert prov.origin == :proposer
      assert prov.parent_ids == ["a", "b"]
    end
  end

  describe "is_synthesis?/1" do
    test "returns true for synthesizer origin" do
      prov = Provenance.new(:synthesizer)
      assert Provenance.is_synthesis?(prov)
    end

    test "returns false for other origins" do
      for origin <- [:proposer, :antagonist, :external] do
        prov = Provenance.new(origin)
        refute Provenance.is_synthesis?(prov)
      end
    end
  end

  describe "depth/1" do
    test "returns number of parents" do
      prov = Provenance.new(:synthesizer, parent_ids: ["a", "b", "c"])
      assert Provenance.depth(prov) == 3
    end

    test "returns 0 for no parents" do
      prov = Provenance.new(:proposer)
      assert Provenance.depth(prov) == 0
    end
  end
end

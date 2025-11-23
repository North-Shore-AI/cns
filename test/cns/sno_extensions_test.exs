defmodule CNS.SNO.ExtensionsTest do
  use ExUnit.Case, async: true

  alias CNS.SNO

  describe "topological fields" do
    test "SNO can store beta1 (first Betti number)" do
      sno = SNO.new("Test claim")
      sno = %{sno | metadata: Map.put(sno.metadata, :beta1, 2)}

      assert sno.metadata.beta1 == 2
    end

    test "SNO can store cycle count" do
      sno = SNO.new("Test claim")
      sno = %{sno | metadata: Map.put(sno.metadata, :cycle_count, 3)}

      assert sno.metadata.cycle_count == 3
    end
  end

  describe "geometric fields" do
    test "SNO can store chirality score" do
      sno = SNO.new("Test claim")
      sno = %{sno | metadata: Map.put(sno.metadata, :chirality, 0.65)}

      assert sno.metadata.chirality == 0.65
    end

    test "SNO can store evidence overlap" do
      sno = SNO.new("Test claim")
      sno = %{sno | metadata: Map.put(sno.metadata, :evidence_overlap, 0.3)}

      assert sno.metadata.evidence_overlap == 0.3
    end
  end

  describe "with_topology/2" do
    test "adds topological metrics to SNO" do
      sno = SNO.new("Test claim")

      topology = %{
        beta1: 1,
        cycle_count: 2,
        polarity_conflict: false
      }

      updated = SNO.with_topology(sno, topology)

      assert updated.metadata.topology.beta1 == 1
      assert updated.metadata.topology.cycle_count == 2
      assert updated.metadata.topology.polarity_conflict == false
    end
  end

  describe "with_chirality/2" do
    test "adds chirality metrics to SNO" do
      sno = SNO.new("Test claim")

      chirality = %{
        score: 0.55,
        evidence_overlap: 0.2,
        norm_distance: 0.8
      }

      updated = SNO.with_chirality(sno, chirality)

      assert updated.metadata.chirality.score == 0.55
    end
  end
end

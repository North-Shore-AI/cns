defmodule CNS.TopologyTest do
  use ExUnit.Case, async: true

  alias CNS.{Provenance, SNO, Topology}
  alias Graph

  describe "build_graph/1" do
    test "builds graph from SNOs" do
      s1 = SNO.new("A", id: "1")
      prov = Provenance.new(:synthesizer, parent_ids: ["1"])
      s2 = SNO.new("B", id: "2", provenance: prov)

      graph = Topology.build_graph([s1, s2])

      assert Graph.has_vertex?(graph, "1")
      assert Graph.has_vertex?(graph, "2")
      assert Graph.edge(graph, "1", "2") != nil
    end

    test "handles multiple parents" do
      s1 = SNO.new("A", id: "1")
      s2 = SNO.new("B", id: "2")
      prov = Provenance.new(:synthesizer, parent_ids: ["1", "2"])
      s3 = SNO.new("C", id: "3", provenance: prov)

      graph = Topology.build_graph([s1, s2, s3])

      assert Graph.edge(graph, "1", "3") != nil
      assert Graph.edge(graph, "2", "3") != nil
    end

    test "handles SNOs without provenance" do
      s1 = SNO.new("A", id: "1")
      s2 = SNO.new("B", id: "2")

      graph = Topology.build_graph([s1, s2])

      assert Graph.out_neighbors(graph, "1") == []
      assert Graph.out_neighbors(graph, "2") == []
    end
  end

  describe "betti_numbers/1" do
    test "calculates for linear graph" do
      graph = %{"a" => ["b"], "b" => ["c"], "c" => []}
      betti = Topology.betti_numbers(graph)

      # One connected component
      assert betti.b0 == 1
    end

    test "calculates for empty graph" do
      betti = Topology.betti_numbers(%{})

      assert betti.b0 == 0
      assert betti.b1 == 0
    end

    test "detects multiple components" do
      graph = %{"a" => ["b"], "b" => [], "c" => ["d"], "d" => []}
      betti = Topology.betti_numbers(graph)

      # Two components
      assert betti.b0 == 2
    end
  end

  describe "detect_cycles/1" do
    test "detects simple cycle" do
      graph = %{"a" => ["b"], "b" => ["a"]}
      cycles = Topology.detect_cycles(graph)

      assert length(cycles) > 0
    end

    test "returns empty for DAG" do
      graph = %{"a" => ["b"], "b" => ["c"], "c" => []}
      cycles = Topology.detect_cycles(graph)

      assert cycles == []
    end

    test "detects multiple cycles" do
      graph = %{
        "a" => ["b"],
        "b" => ["c", "a"],
        "c" => ["b"]
      }

      cycles = Topology.detect_cycles(graph)

      assert length(cycles) >= 1
    end
  end

  describe "dag?/1" do
    test "returns true for DAG" do
      graph = %{"a" => ["b"], "b" => ["c"], "c" => []}
      assert Topology.dag?(graph)
    end

    test "returns false for cyclic graph" do
      graph = %{"a" => ["b"], "b" => ["a"]}
      refute Topology.dag?(graph)
    end

    test "returns true for empty graph" do
      assert Topology.dag?(%{})
    end
  end

  describe "depth/1" do
    test "calculates depth for linear graph" do
      graph = %{"a" => ["b"], "b" => ["c"], "c" => []}
      assert Topology.depth(graph) == 2
    end

    test "returns 0 for single node" do
      graph = %{"a" => []}
      assert Topology.depth(graph) == 0
    end

    test "returns 0 for empty graph" do
      assert Topology.depth(%{}) == 0
    end

    test "handles branching" do
      graph = %{
        "a" => ["b", "c"],
        "b" => ["d"],
        "c" => [],
        "d" => []
      }

      assert Topology.depth(graph) == 2
    end
  end

  describe "find_roots/1" do
    test "finds root nodes" do
      graph = %{"a" => ["b", "c"], "b" => [], "c" => []}
      roots = Topology.find_roots(graph)

      assert "a" in roots
      assert length(roots) == 1
    end

    test "handles multiple roots" do
      graph = %{"a" => ["c"], "b" => ["c"], "c" => []}
      roots = Topology.find_roots(graph)

      assert "a" in roots
      assert "b" in roots
    end

    test "returns all nodes if no edges" do
      graph = %{"a" => [], "b" => [], "c" => []}
      roots = Topology.find_roots(graph)

      assert length(roots) == 3
    end
  end

  describe "find_leaves/1" do
    test "finds leaf nodes" do
      graph = %{"a" => ["b"], "b" => []}
      leaves = Topology.find_leaves(graph)

      assert "b" in leaves
      assert length(leaves) == 1
    end

    test "handles multiple leaves" do
      graph = %{"a" => ["b", "c"], "b" => [], "c" => []}
      leaves = Topology.find_leaves(graph)

      assert "b" in leaves
      assert "c" in leaves
    end
  end

  describe "connectivity/1" do
    test "calculates connectivity metrics" do
      graph = %{"a" => ["b"], "b" => ["c"], "c" => []}
      metrics = Topology.connectivity(graph)

      assert metrics.nodes == 3
      assert metrics.edges == 2
      assert metrics.density >= 0.0
      assert metrics.components == 1
    end

    test "handles empty graph" do
      metrics = Topology.connectivity(%{})

      assert metrics.nodes == 0
      assert metrics.edges == 0
    end
  end

  describe "all_paths/3" do
    test "finds all paths between nodes" do
      graph = %{
        "a" => ["b", "c"],
        "b" => ["d"],
        "c" => ["d"],
        "d" => []
      }

      paths = Topology.all_paths(graph, "a", "d")

      assert length(paths) == 2
      assert Enum.all?(paths, fn p -> hd(p) == "a" and List.last(p) == "d" end)
    end

    test "returns empty for no path" do
      graph = %{"a" => [], "b" => []}
      paths = Topology.all_paths(graph, "a", "b")

      assert paths == []
    end

    test "finds direct path" do
      graph = %{"a" => ["b"], "b" => []}
      paths = Topology.all_paths(graph, "a", "b")

      assert paths == [["a", "b"]]
    end
  end

  describe "topological_sort/1" do
    test "sorts DAG" do
      graph = %{"a" => ["b"], "b" => ["c"], "c" => []}
      assert {:ok, sorted} = Topology.topological_sort(graph)

      assert length(sorted) == 3
      # a should come before b, b before c
      assert Enum.find_index(sorted, &(&1 == "a")) < Enum.find_index(sorted, &(&1 == "b"))
      assert Enum.find_index(sorted, &(&1 == "b")) < Enum.find_index(sorted, &(&1 == "c"))
    end

    test "returns error for cyclic graph" do
      graph = %{"a" => ["b"], "b" => ["a"]}
      assert {:error, :has_cycle} = Topology.topological_sort(graph)
    end

    test "handles independent nodes" do
      graph = %{"a" => [], "b" => [], "c" => []}
      assert {:ok, sorted} = Topology.topological_sort(graph)

      assert length(sorted) == 3
    end
  end
end

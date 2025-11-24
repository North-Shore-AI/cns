defmodule CNS.Topology.SurrogatesTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias CNS.Topology.Surrogates

  describe "compute_beta1_surrogate/1" do
    test "returns 0 for empty graph" do
      assert Surrogates.compute_beta1_surrogate(%{}) == 0
    end

    test "returns 0 for DAG (no cycles)" do
      graph = %{
        "a" => ["b", "c"],
        "b" => ["d"],
        "c" => ["d"],
        "d" => []
      }

      assert Surrogates.compute_beta1_surrogate(graph) == 0
    end

    test "detects single cycle" do
      graph = %{
        "a" => ["b"],
        "b" => ["c"],
        "c" => ["a"]
      }

      assert Surrogates.compute_beta1_surrogate(graph) == 1
    end

    test "detects self-loop" do
      graph = %{
        "a" => ["a"]
      }

      assert Surrogates.compute_beta1_surrogate(graph) == 1
    end

    test "counts multiple independent cycles" do
      graph = %{
        # First cycle: a -> b -> a
        "a" => ["b"],
        "b" => ["a"],
        # Second cycle: c -> d -> c
        "c" => ["d"],
        "d" => ["c"],
        # No connection between cycles
        "e" => []
      }

      assert Surrogates.compute_beta1_surrogate(graph) == 2
    end

    test "handles nested cycles correctly" do
      graph = %{
        # Outer cycle: a -> b -> c -> a
        # Inner cycle: b -> c -> b
        "a" => ["b"],
        "b" => ["c"],
        "c" => ["a", "b"]
      }

      # Cyclomatic complexity counts two independent cycles
      assert Surrogates.compute_beta1_surrogate(graph) == 2
    end

    test "handles large graphs efficiently" do
      # Create a graph with 100+ nodes
      nodes = for i <- 1..100, do: "node_#{i}"

      # Create a linear chain with one cycle at the end
      graph =
        nodes
        |> Enum.chunk_every(2, 1, :discard)
        |> Enum.reduce(%{}, fn [from, to], acc ->
          Map.put(acc, from, [to])
        end)
        # Add cycle
        |> Map.put(List.last(nodes), [List.first(nodes)])

      result = Surrogates.compute_beta1_surrogate(graph)
      assert result == 1
    end

    property "beta1 is always non-negative" do
      check all(graph <- graph_generator()) do
        result = Surrogates.compute_beta1_surrogate(graph)
        assert result >= 0
      end
    end

    property "adding an edge can only increase or maintain beta1" do
      check all(
              graph <- non_empty_graph_generator(),
              node1 <- member_of(Map.keys(graph)),
              node2 <- member_of(Map.keys(graph))
            ) do
        original_beta1 = Surrogates.compute_beta1_surrogate(graph)

        # Add an edge
        updated_graph = Map.update(graph, node1, [node2], &[node2 | &1])
        new_beta1 = Surrogates.compute_beta1_surrogate(updated_graph)

        assert new_beta1 >= original_beta1
      end
    end
  end

  describe "compute_fragility_surrogate/2" do
    test "returns 0 for empty embeddings" do
      assert Surrogates.compute_fragility_surrogate([]) == 0.0
    end

    test "returns 0 for single point" do
      assert Surrogates.compute_fragility_surrogate([[0.5, 0.5]]) == 0.0
    end

    test "returns low fragility for uniform embeddings" do
      embeddings = [
        [0.5, 0.5],
        [0.51, 0.49],
        [0.49, 0.51],
        [0.5, 0.48],
        [0.52, 0.5]
      ]

      fragility = Surrogates.compute_fragility_surrogate(embeddings)
      assert fragility < 0.3
    end

    test "returns high fragility for high variance embeddings" do
      embeddings = [
        [0.1, 0.1],
        [0.9, 0.9],
        [0.1, 0.9],
        [0.9, 0.1],
        [0.5, 0.5]
      ]

      fragility = Surrogates.compute_fragility_surrogate(embeddings)
      assert fragility > 0.5
    end

    test "computes correct distance for two points" do
      embeddings = [
        [0.0, 0.0],
        [1.0, 0.0]
      ]

      fragility = Surrogates.compute_fragility_surrogate(embeddings)
      assert fragility > 0.0 and fragility < 1.0
    end

    test "k-nearest neighbors parameter works" do
      embeddings = for _ <- 1..10, do: [Enum.random(0..100) / 100, Enum.random(0..100) / 100]

      fragility_k3 = Surrogates.compute_fragility_surrogate(embeddings, k: 3)
      fragility_k7 = Surrogates.compute_fragility_surrogate(embeddings, k: 7)

      # Different k values should give different results
      assert fragility_k3 != fragility_k7
    end

    test "supports different distance metrics" do
      embeddings = [
        [0.6, 0.8],
        [0.8, 0.6],
        [1.0, 0.0],
        [0.0, 1.0]
      ]

      fragility_cosine = Surrogates.compute_fragility_surrogate(embeddings, metric: :cosine)
      fragility_euclidean = Surrogates.compute_fragility_surrogate(embeddings, metric: :euclidean)

      # Different metrics should give different results
      assert fragility_cosine != fragility_euclidean
    end

    test "handles Nx tensors as input" do
      embeddings_list = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
      embeddings_tensor = Nx.tensor(embeddings_list, type: :f32)

      fragility_list = Surrogates.compute_fragility_surrogate(embeddings_list)
      fragility_tensor = Surrogates.compute_fragility_surrogate(embeddings_tensor)

      assert_in_delta fragility_list, fragility_tensor, 0.001
    end

    property "fragility is always non-negative" do
      check all(embeddings <- embeddings_generator()) do
        fragility = Surrogates.compute_fragility_surrogate(embeddings)
        assert fragility >= 0.0
      end
    end

    property "fragility is normalized to [0, 1] range" do
      check all(embeddings <- non_empty_embeddings_generator()) do
        fragility = Surrogates.compute_fragility_surrogate(embeddings)
        assert fragility >= 0.0 and fragility <= 1.0
      end
    end
  end

  describe "compute_surrogates/2" do
    test "computes both surrogates from SNO-like structure" do
      sno = %{
        causal_links: [{"a", "b"}, {"b", "c"}, {"c", "a"}],
        embeddings: [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
      }

      result = Surrogates.compute_surrogates(sno)

      assert Map.has_key?(result, :beta1)
      assert Map.has_key?(result, :fragility)
      # Has a cycle
      assert result.beta1 == 1
      assert is_float(result.fragility)
    end

    test "handles missing causal_links" do
      sno = %{
        embeddings: [[0.1, 0.2], [0.3, 0.4]]
      }

      result = Surrogates.compute_surrogates(sno)

      assert result.beta1 == 0
      assert result.fragility > 0.0
    end

    test "handles missing embeddings" do
      sno = %{
        causal_links: [{"a", "b"}, {"b", "c"}]
      }

      result = Surrogates.compute_surrogates(sno)

      # No cycle
      assert result.beta1 == 0
      assert result.fragility == 0.0
    end

    test "handles empty SNO" do
      result = Surrogates.compute_surrogates(%{})

      assert result.beta1 == 0
      assert result.fragility == 0.0
    end
  end

  describe "validate_correlation/2" do
    test "computes correlations with ground truth" do
      examples = [
        %{surrogates: %{beta1: 0, fragility: 0.1}, label: 0},
        %{surrogates: %{beta1: 0, fragility: 0.2}, label: 0},
        %{surrogates: %{beta1: 1, fragility: 0.7}, label: 1},
        %{surrogates: %{beta1: 2, fragility: 0.8}, label: 1},
        %{surrogates: %{beta1: 0, fragility: 0.15}, label: 0},
        %{surrogates: %{beta1: 1, fragility: 0.6}, label: 1}
      ]

      result = Surrogates.validate_correlation(examples)

      assert Map.has_key?(result, :beta1_correlation)
      assert Map.has_key?(result, :beta1_p_value)
      assert Map.has_key?(result, :fragility_correlation)
      assert Map.has_key?(result, :fragility_p_value)
      assert Map.has_key?(result, :combined_correlation)
      assert Map.has_key?(result, :passes_gate1)
      assert result.n_samples == 6
    end

    test "correctly identifies passing Gate 1 threshold" do
      # Create examples with strong correlation
      examples =
        for i <- 0..19 do
          label = rem(i, 2)

          %{
            surrogates: %{
              beta1: label * 2 + Enum.random(-1..1) * 0.1,
              fragility: label * 0.5 + Enum.random(-1..1) * 0.05
            },
            label: label
          }
        end

      result = Surrogates.validate_correlation(examples)

      # With strong correlation, should pass Gate 1
      assert result.beta1_correlation > 0.35
      assert result.passes_gate1 == true
    end

    test "supports Spearman correlation" do
      examples = [
        %{surrogates: %{beta1: 0, fragility: 0.1}, label: 0},
        %{surrogates: %{beta1: 1, fragility: 0.9}, label: 1},
        %{surrogates: %{beta1: 2, fragility: 0.2}, label: 0}
      ]

      pearson_result = Surrogates.validate_correlation(examples, metric: :pearson)
      spearman_result = Surrogates.validate_correlation(examples, metric: :spearman)

      # Should compute different correlations
      assert pearson_result.beta1_correlation != spearman_result.beta1_correlation
    end
  end

  describe "integration tests" do
    test "full pipeline from causal links to validation" do
      # Create synthetic data with known properties
      examples =
        for i <- 0..29 do
          has_cycle = rem(i, 3) == 0

          causal_links =
            if has_cycle do
              [{"a", "b"}, {"b", "c"}, {"c", "a"}]
            else
              [{"a", "b"}, {"b", "c"}]
            end

          # Create embeddings with fragility correlating to cycles
          embeddings =
            if has_cycle do
              # High variance embeddings for cyclic structures
              for _ <- 1..5, do: [Enum.random(0..100) / 100, Enum.random(0..100) / 100]
            else
              # Low variance embeddings for DAGs
              base = [0.5, 0.5]

              for _ <- 1..5 do
                [
                  Enum.at(base, 0) + Enum.random(-5..5) / 100,
                  Enum.at(base, 1) + Enum.random(-5..5) / 100
                ]
              end
            end

          sno = %{causal_links: causal_links, embeddings: embeddings}
          surrogates = Surrogates.compute_surrogates(sno)

          %{
            surrogates: surrogates,
            label: if(has_cycle, do: 1, else: 0)
          }
        end

      validation = Surrogates.validate_correlation(examples)

      # Should show some correlation
      assert validation.beta1_correlation > 0.0
      assert validation.n_samples == 30
    end
  end

  # Property generators

  defp graph_generator do
    gen all(node_count <- integer(0..10)) do
      nodes =
        if node_count <= 0 do
          []
        else
          for i <- 1..node_count, do: "node_#{i}"
        end

      edges =
        for from <- nodes, into: %{} do
          # Random subset of nodes as children
          max_children = max(0, min(3, node_count - 1))
          children = Enum.take_random(nodes -- [from], Enum.random(0..max_children))
          {from, children}
        end

      edges
    end
  end

  defp non_empty_graph_generator do
    gen all(node_count <- integer(2..10)) do
      nodes = for i <- 1..node_count, do: "node_#{i}"

      edges =
        for from <- nodes, into: %{} do
          max_children = max(0, min(3, node_count - 1))
          children = Enum.take_random(nodes -- [from], Enum.random(0..max_children))
          {from, children}
        end

      edges
    end
  end

  defp embeddings_generator do
    gen all(
          count <- integer(0..10),
          dim <- integer(2..5)
        ) do
      if count <= 0 do
        []
      else
        for _ <- 1..count do
          for _ <- 1..dim do
            Enum.random(0..100) / 100
          end
        end
      end
    end
  end

  defp non_empty_embeddings_generator do
    gen all(
          count <- integer(2..10),
          dim <- integer(2..5)
        ) do
      for _ <- 1..count do
        for _ <- 1..dim do
          Enum.random(0..100) / 100
        end
      end
    end
  end
end

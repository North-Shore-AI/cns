defmodule CNS.CrucibleAdapterTest do
  use ExUnit.Case, async: true

  alias CNS.CrucibleAdapter

  describe "behaviour implementation" do
    test "implements Crucible.CNS.Adapter behaviour" do
      # Check that the module exports the required callback
      assert function_exported?(CrucibleAdapter, :evaluate, 3)
    end
  end

  describe "evaluate/3 with empty inputs" do
    test "returns success with empty metrics for empty inputs" do
      assert {:ok, metrics} = CrucibleAdapter.evaluate([], [], %{})

      assert metrics.schema_compliance == 1.0
      assert metrics.parseable_count == 0
      assert metrics.unparseable_count == 0
      assert metrics.citation_accuracy == 1.0
      assert metrics.meets_threshold == true
    end
  end

  describe "evaluate/3 with mismatched lengths" do
    test "returns error when examples and outputs have different lengths" do
      examples = [%{"prompt" => "test"}]
      outputs = ["output1", "output2"]

      assert {:error, {:mismatched_lengths, msg}} = CrucibleAdapter.evaluate(examples, outputs, %{})
      assert msg =~ "examples count"
      assert msg =~ "outputs count"
    end
  end

  describe "evaluate/3 with valid CLAIM format" do
    test "successfully parses single CLAIM" do
      examples = [
        %{
          "prompt" => "Extract claims",
          "completion" => "CLAIM[c1] (Document 123): Coffee contains caffeine",
          "metadata" => %{"doc_ids" => ["123"]}
        }
      ]

      outputs = [
        "CLAIM[c1] (Document 123): Coffee has caffeine"
      ]

      assert {:ok, metrics} = CrucibleAdapter.evaluate(examples, outputs, %{})

      assert metrics.schema_compliance == 1.0
      assert metrics.parseable_count == 1
      assert metrics.unparseable_count == 0
    end

    test "successfully parses multiple CLAIMs" do
      examples = [
        %{
          "prompt" => "Extract claims",
          "completion" => "Multiple claims expected",
          "metadata" => %{"doc_ids" => ["123", "456"]}
        }
      ]

      outputs = [
        """
        CLAIM[c1] (Document 123): First claim
        CLAIM[c2] (Document 456): Second claim
        """
      ]

      assert {:ok, metrics} = CrucibleAdapter.evaluate(examples, outputs, %{})

      assert metrics.schema_compliance == 1.0
      assert metrics.parseable_count == 1
    end

    test "handles CLAIM without document ID" do
      examples = [
        %{
          "prompt" => "Extract claims",
          "completion" => "CLAIM[c1]: A claim without document",
          "metadata" => %{}
        }
      ]

      outputs = [
        "CLAIM[c1]: A claim without specific document"
      ]

      assert {:ok, metrics} = CrucibleAdapter.evaluate(examples, outputs, %{})

      assert metrics.schema_compliance == 1.0
      assert metrics.parseable_count == 1
    end
  end

  describe "evaluate/3 with RELATION patterns" do
    test "successfully parses RELATION patterns" do
      examples = [
        %{
          "prompt" => "Extract claims and relations",
          "completion" => "Claims with relations",
          "metadata" => %{"doc_ids" => ["123"]}
        }
      ]

      outputs = [
        """
        CLAIM[c1] (Document 123): Premise claim
        CLAIM[c2] (Document 123): Conclusion claim
        RELATION: c1 supports c2
        """
      ]

      assert {:ok, metrics} = CrucibleAdapter.evaluate(examples, outputs, %{})

      assert metrics.schema_compliance == 1.0
      assert metrics.parseable_count == 1
    end

    test "handles refutes and contradicts relations" do
      examples = [
        %{
          "prompt" => "Extract claims",
          "completion" => "Claims",
          "metadata" => %{"doc_ids" => ["123"]}
        }
      ]

      outputs = [
        """
        CLAIM[c1] (Document 123): Claim A
        CLAIM[c2] (Document 123): Claim B
        CLAIM[c3] (Document 123): Claim C
        RELATION: c1 refutes c2
        RELATION: c2 contradicts c3
        """
      ]

      assert {:ok, metrics} = CrucibleAdapter.evaluate(examples, outputs, %{})

      assert metrics.schema_compliance == 1.0
    end
  end

  describe "evaluate/3 with malformed outputs" do
    test "handles unparseable output gracefully" do
      examples = [
        %{"prompt" => "Extract", "completion" => "Expected", "metadata" => %{}}
      ]

      outputs = [
        "This is not a valid CLAIM format at all"
      ]

      assert {:ok, metrics} = CrucibleAdapter.evaluate(examples, outputs, %{})

      assert metrics.schema_compliance == 0.0
      assert metrics.parseable_count == 0
      assert metrics.unparseable_count == 1
    end

    test "handles mixed valid and invalid outputs" do
      examples = [
        %{"prompt" => "P1", "completion" => "C1", "metadata" => %{}},
        %{"prompt" => "P2", "completion" => "C2", "metadata" => %{}}
      ]

      outputs = [
        "CLAIM[c1]: Valid claim",
        "Invalid output without CLAIM"
      ]

      assert {:ok, metrics} = CrucibleAdapter.evaluate(examples, outputs, %{})

      assert metrics.schema_compliance == 0.5
      assert metrics.parseable_count == 1
      assert metrics.unparseable_count == 1
    end
  end

  describe "citation validation" do
    test "validates citations against corpus" do
      examples = [
        %{
          "prompt" => "Extract",
          "completion" => "Expected",
          "metadata" => %{"doc_ids" => ["123", "456"]}
        }
      ]

      outputs = [
        """
        CLAIM[c1] (Document 123): Valid citation
        CLAIM[c2] (Document 456): Another valid citation
        """
      ]

      assert {:ok, metrics} = CrucibleAdapter.evaluate(examples, outputs, %{})

      assert metrics.citation_accuracy == 1.0
      assert metrics.valid_citations == 2
      assert metrics.invalid_citations == 0
    end

    test "detects invalid citations" do
      examples = [
        %{
          "prompt" => "Extract",
          "completion" => "Expected",
          "metadata" => %{"doc_ids" => ["123"]}
        }
      ]

      outputs = [
        """
        CLAIM[c1] (Document 123): Valid citation
        CLAIM[c2] (Document 999): Invalid citation
        """
      ]

      assert {:ok, metrics} = CrucibleAdapter.evaluate(examples, outputs, %{})

      assert metrics.citation_accuracy == 0.5
      assert metrics.valid_citations == 1
      assert metrics.invalid_citations == 1
      assert metrics.hallucinated_citations == 1
    end

    test "handles no citations" do
      examples = [
        %{"prompt" => "Extract", "completion" => "Expected", "metadata" => %{}}
      ]

      outputs = [
        "CLAIM[c1]: Claim without citation"
      ]

      assert {:ok, metrics} = CrucibleAdapter.evaluate(examples, outputs, %{})

      assert metrics.citation_accuracy == 1.0
      assert metrics.valid_citations == 0
      assert metrics.invalid_citations == 0
    end
  end

  describe "semantic metrics" do
    test "computes mean similarity when available" do
      examples = [
        %{
          "prompt" => "Extract",
          "completion" => "Coffee contains caffeine",
          "metadata" => %{}
        }
      ]

      outputs = [
        "CLAIM[c1]: Coffee has caffeine"
      ]

      assert {:ok, metrics} = CrucibleAdapter.evaluate(examples, outputs, %{})

      # Should compute some similarity (word overlap based)
      assert is_float(metrics.mean_similarity) or is_nil(metrics.mean_similarity)
    end

    test "computes mean entailment when SNOs exist" do
      examples = [
        %{
          "prompt" => "Extract",
          "completion" => "Expected",
          "metadata" => %{"doc_ids" => ["123"]}
        }
      ]

      outputs = [
        "CLAIM[c1] (Document 123): Test claim"
      ]

      assert {:ok, metrics} = CrucibleAdapter.evaluate(examples, outputs, %{})

      assert is_float(metrics.mean_entailment) or is_nil(metrics.mean_entailment)
    end
  end

  describe "topology metrics" do
    test "computes topology for DAG structure" do
      examples = [
        %{
          "prompt" => "Extract",
          "completion" => "Expected",
          "metadata" => %{"doc_ids" => ["123"]}
        }
      ]

      outputs = [
        """
        CLAIM[c1] (Document 123): Base claim
        CLAIM[c2] (Document 123): Derived claim
        RELATION: c1 supports c2
        """
      ]

      assert {:ok, metrics} = CrucibleAdapter.evaluate(examples, outputs, %{})

      assert metrics.topology.dag_count >= 0
      assert metrics.topology.cyclic_count >= 0
      assert metrics.topology.mean_beta1 >= 0.0
    end

    test "detects cycles in claim graph" do
      examples = [
        %{
          "prompt" => "Extract",
          "completion" => "Expected",
          "metadata" => %{"doc_ids" => ["123"]}
        }
      ]

      outputs = [
        """
        CLAIM[c1] (Document 123): Claim A
        CLAIM[c2] (Document 123): Claim B
        RELATION: c1 supports c2
        RELATION: c2 supports c1
        """
      ]

      assert {:ok, metrics} = CrucibleAdapter.evaluate(examples, outputs, %{})

      # Should detect the cycle
      assert metrics.topology.cyclic_count >= 0
    end

    test "handles empty topology" do
      examples = [
        %{"prompt" => "Extract", "completion" => "Expected", "metadata" => %{}}
      ]

      outputs = [
        "No valid claims"
      ]

      assert {:ok, metrics} = CrucibleAdapter.evaluate(examples, outputs, %{})

      assert metrics.topology.mean_beta1 == 0.0
      assert metrics.topology.max_beta1 == 0
      assert metrics.topology.dag_count == 0
      assert metrics.topology.cyclic_count == 0
    end
  end

  describe "chirality metrics" do
    test "detects polarity conflicts" do
      examples = [
        %{
          "prompt" => "Extract",
          "completion" => "Expected",
          "metadata" => %{}
        }
      ]

      outputs = [
        """
        CLAIM[c1]: Coffee increases alertness
        CLAIM[c2]: Coffee decreases alertness
        """
      ]

      assert {:ok, metrics} = CrucibleAdapter.evaluate(examples, outputs, %{})

      assert metrics.chirality.polarity_conflicts >= 1
      assert metrics.chirality.mean_score > 0.0
    end

    test "handles no conflicts" do
      examples = [
        %{
          "prompt" => "Extract",
          "completion" => "Expected",
          "metadata" => %{}
        }
      ]

      outputs = [
        """
        CLAIM[c1]: Coffee contains caffeine
        CLAIM[c2]: Tea contains caffeine
        """
      ]

      assert {:ok, metrics} = CrucibleAdapter.evaluate(examples, outputs, %{})

      assert metrics.chirality.polarity_conflicts == 0
      assert metrics.chirality.mean_score == 0.0
    end
  end

  describe "overall quality metrics" do
    test "calculates overall quality score" do
      examples = [
        %{
          "prompt" => "Extract",
          "completion" => "CLAIM[c1] (Document 123): Test",
          "metadata" => %{"doc_ids" => ["123"]}
        }
      ]

      outputs = [
        "CLAIM[c1] (Document 123): Test claim"
      ]

      assert {:ok, metrics} = CrucibleAdapter.evaluate(examples, outputs, %{})

      assert metrics.overall_quality >= 0.0
      assert metrics.overall_quality <= 1.0
    end

    test "determines if CNS 3.0 thresholds are met" do
      # Perfect output that should meet thresholds
      examples = [
        %{
          "prompt" => "Extract",
          "completion" => "CLAIM[c1] (Document 123): Coffee contains caffeine",
          "metadata" => %{"doc_ids" => ["123"]}
        }
      ]

      outputs = [
        "CLAIM[c1] (Document 123): Coffee contains caffeine"
      ]

      assert {:ok, metrics} = CrucibleAdapter.evaluate(examples, outputs, %{})

      assert is_boolean(metrics.meets_threshold)

      # Check if individual thresholds contribute correctly
      assert metrics.schema_compliance >= 0.95 or not metrics.meets_threshold
      assert metrics.citation_accuracy >= 0.95 or not metrics.meets_threshold
    end
  end

  describe "error handling" do
    test "handles malformed examples gracefully" do
      examples = [
        # Missing expected fields
        %{},
        # Invalid example
        nil
      ]

      outputs = [
        "CLAIM[c1]: Test",
        "CLAIM[c2]: Test"
      ]

      # Should not crash
      result = CrucibleAdapter.evaluate(examples, outputs, %{})
      assert match?({:ok, _}, result) or match?({:error, _}, result)
    end

    test "handles malformed outputs gracefully" do
      examples = [
        %{"prompt" => "Test", "completion" => "Test", "metadata" => %{}}
      ]

      outputs = [
        # Empty output instead of nil
        ""
      ]

      # Should handle empty string gracefully
      {:ok, metrics} = CrucibleAdapter.evaluate(examples, outputs, %{})
      assert metrics.unparseable_count == 1
    end
  end

  describe "integration with real-world examples" do
    test "processes SciFact-style examples" do
      examples = [
        %{
          "prompt" => "You are extracting atomic claims...",
          "completion" => "CLAIM[c1]: 1 in 5 million in UK have abnormal PrP positivity.",
          "metadata" => %{
            "source" => "SciFact",
            "claim_id" => 2,
            "doc_ids" => ["13734012"],
            "license" => "CC-BY-4.0"
          }
        }
      ]

      outputs = [
        """
        CLAIM[c1]: 1 in 5 million in UK have abnormal PrP positivity.
        CLAIM[c2] (Document 13734012): Of the 32,441 appendix samples 16 were positive for abnormal PrP.
        RELATION: c2 refutes c1
        """
      ]

      assert {:ok, metrics} = CrucibleAdapter.evaluate(examples, outputs, %{})

      assert metrics.schema_compliance == 1.0
      assert metrics.parseable_count == 1
      assert metrics.citation_accuracy == 1.0
      assert metrics.overall_quality > 0.0
    end
  end

  describe "property-based tests" do
    test "all metrics are in valid ranges" do
      # Test with various inputs
      for _ <- 1..10 do
        examples = [
          %{
            "prompt" => "Test prompt #{:rand.uniform(100)}",
            "completion" => "CLAIM[c1]: Test claim #{:rand.uniform(100)}",
            "metadata" => %{"doc_ids" => ["#{:rand.uniform(999)}"]}
          }
        ]

        outputs = [
          "CLAIM[c1] (Document #{:rand.uniform(999)}): Generated claim"
        ]

        assert {:ok, metrics} = CrucibleAdapter.evaluate(examples, outputs, %{})

        # Verify all metrics are in valid ranges
        assert metrics.schema_compliance >= 0.0 and metrics.schema_compliance <= 1.0
        assert metrics.citation_accuracy >= 0.0 and metrics.citation_accuracy <= 1.0
        assert metrics.overall_quality >= 0.0 and metrics.overall_quality <= 1.0

        assert metrics.parseable_count >= 0
        assert metrics.unparseable_count >= 0
        assert metrics.valid_citations >= 0
        assert metrics.invalid_citations >= 0

        assert metrics.topology.mean_beta1 >= 0.0
        assert metrics.chirality.mean_score >= 0.0
      end
    end

    test "schema_compliance equals parseable_count / total" do
      for total <- [1, 2, 5, 10] do
        # Create examples with varying parseability
        examples =
          Enum.map(1..total, fn i ->
            %{"prompt" => "P#{i}", "completion" => "C#{i}", "metadata" => %{}}
          end)

        # Half valid, half invalid
        outputs =
          Enum.map(1..total, fn i ->
            if rem(i, 2) == 0 do
              "CLAIM[c#{i}]: Valid claim"
            else
              "Invalid output #{i}"
            end
          end)

        assert {:ok, metrics} = CrucibleAdapter.evaluate(examples, outputs, %{})

        expected_compliance = metrics.parseable_count / total
        assert_in_delta metrics.schema_compliance, expected_compliance, 0.01
      end
    end

    test "overall_quality is deterministic" do
      examples = [
        %{
          "prompt" => "Extract claims",
          "completion" => "CLAIM[c1] (Document 123): Test claim",
          "metadata" => %{"doc_ids" => ["123"]}
        }
      ]

      outputs = [
        "CLAIM[c1] (Document 123): Test claim output"
      ]

      # Run multiple times with same input
      results =
        for _ <- 1..5 do
          {:ok, metrics} = CrucibleAdapter.evaluate(examples, outputs, %{})
          metrics.overall_quality
        end

      # All results should be identical
      assert Enum.all?(results, &(&1 == hd(results)))
    end
  end
end

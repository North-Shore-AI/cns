defmodule CNS.TrainingTest do
  use ExUnit.Case, async: true

  alias CNS.{Training, SNO, Evidence}

  describe "prepare_dataset/2" do
    test "prepares dataset with default split" do
      snos = for i <- 1..10, do: SNO.new("Claim #{i}")

      assert {:ok, dataset} = Training.prepare_dataset(snos)

      assert Map.has_key?(dataset, :train)
      assert Map.has_key?(dataset, :validation)
      assert Map.has_key?(dataset, :test)
      # 80%
      assert length(dataset.train) == 8
      # 10%
      assert length(dataset.validation) == 1
    end

    test "respects custom split" do
      snos = for i <- 1..10, do: SNO.new("Claim #{i}")

      assert {:ok, dataset} = Training.prepare_dataset(snos, split: [0.6, 0.2, 0.2])

      assert length(dataset.train) == 6
      assert length(dataset.validation) == 2
    end

    test "handles different formats" do
      snos = [SNO.new("Test claim")]

      for format <- [:dialectical, :qa, :nli] do
        assert {:ok, dataset} = Training.prepare_dataset(snos, format: format)
        assert dataset.metadata.format == format
      end
    end

    test "includes evidence when enabled" do
      evidence = [Evidence.new("Source", "Content")]
      snos = [SNO.new("Test", evidence: evidence)]

      assert {:ok, dataset} = Training.prepare_dataset(snos, include_evidence: true)
      assert length(dataset.train) > 0
    end
  end

  describe "lora_config/1" do
    test "creates config with defaults" do
      config = Training.lora_config()

      assert config.rank == 16
      assert config.alpha == 32
      assert config.base_model == "mistral-7b"
      assert config.epochs == 3
    end

    test "respects custom options" do
      config = Training.lora_config(rank: 8, epochs: 5, target: :proposer)

      assert config.rank == 8
      assert config.epochs == 5
      assert config.target == :proposer
    end

    test "sets target modules based on target" do
      proposer_config = Training.lora_config(target: :proposer)
      synthesizer_config = Training.lora_config(target: :synthesizer)

      assert "k_proj" in proposer_config.target_modules
      assert "o_proj" in synthesizer_config.target_modules
    end
  end

  describe "train/2" do
    test "returns error when Tinkex not available" do
      snos = [SNO.new("Test")]
      {:ok, dataset} = Training.prepare_dataset(snos)
      config = Training.lora_config()

      result = Training.train(dataset, config)

      # Either succeeds with mock or returns not available
      assert match?({:ok, _}, result) or match?({:error, :tinkex_not_available}, result)
    end
  end

  describe "save_checkpoint/2 and load_checkpoint/1" do
    test "round-trips checkpoint" do
      state = %{
        epoch: 2,
        loss: 0.15,
        model_state: %{weights: [1, 2, 3]}
      }

      path = Path.join(System.tmp_dir!(), "cns_test_checkpoint_#{System.unique_integer()}")

      assert {:ok, ^path} = Training.save_checkpoint(state, path)
      assert {:ok, loaded} = Training.load_checkpoint(path)

      assert loaded.epoch == 2
      assert loaded.loss == 0.15

      # Cleanup
      File.rm(path)
    end

    test "creates directory if needed" do
      state = %{epoch: 1}
      path = Path.join([System.tmp_dir!(), "cns_nested_#{System.unique_integer()}", "checkpoint"])

      assert {:ok, _} = Training.save_checkpoint(state, path)

      # Cleanup
      File.rm_rf!(Path.dirname(path))
    end
  end

  describe "triplet_to_example/3" do
    test "creates training example from triplet" do
      thesis = SNO.new("Thesis claim", id: "t1", confidence: 0.8)
      antithesis = SNO.new("Antithesis claim", id: "a1", confidence: 0.7)
      synthesis = SNO.new("Synthesis result", id: "s1", confidence: 0.85)

      example = Training.triplet_to_example(thesis, antithesis, synthesis)

      assert Map.has_key?(example, :input)
      assert Map.has_key?(example, :output)
      assert example.output == "Synthesis result"
      assert example.metadata.thesis_id == "t1"
      assert example.metadata.antithesis_id == "a1"
    end
  end

  describe "evaluate/2" do
    test "calculates evaluation metrics" do
      test_data = [
        %{input: "A", output: "X"},
        %{input: "B", output: "Y"},
        %{input: "C", output: "Z"}
      ]

      predictions = [
        %{output: "X", confidence: 0.9},
        %{output: "Y", confidence: 0.8},
        # Wrong
        %{output: "W", confidence: 0.6}
      ]

      metrics = Training.evaluate(test_data, predictions)

      assert metrics.accuracy == 0.6667
      assert metrics.exact_matches == 2
      assert metrics.total == 3
    end

    test "handles perfect predictions" do
      test_data = [%{output: "A"}, %{output: "B"}]
      predictions = [%{output: "A"}, %{output: "B"}]

      metrics = Training.evaluate(test_data, predictions)
      assert metrics.accuracy == 1.0
    end

    test "raises for mismatched lengths" do
      assert_raise ArgumentError, fn ->
        Training.evaluate([%{output: "A"}], [%{output: "A"}, %{output: "B"}])
      end
    end
  end

  describe "training_report/1" do
    test "generates report string" do
      results = %{
        dataset: %{
          train: [1, 2, 3],
          validation: [4],
          test: [5]
        },
        epochs: 3,
        final_loss: 0.12,
        best_epoch: 2,
        eval: %{
          accuracy: 0.85,
          avg_confidence: 0.78
        },
        config: %{
          base_model: "mistral-7b",
          rank: 16
        }
      }

      report = Training.training_report(results)

      assert is_binary(report)
      assert String.contains?(report, "CNS Training Report")
      assert String.contains?(report, "Train: 3 examples")
      assert String.contains?(report, "Accuracy: 0.85")
    end
  end
end

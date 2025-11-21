defmodule CNSTest do
  use ExUnit.Case, async: true
  doctest CNS

  describe "synthesize/3" do
    test "synthesizes two claims" do
      thesis = %CNS.SNO{
        id: "t1",
        claim: "Coffee improves focus",
        confidence: 0.8,
        evidence: []
      }

      antithesis = %CNS.SNO{
        id: "a1",
        claim: "Coffee causes anxiety",
        confidence: 0.7,
        evidence: []
      }

      assert {:ok, synthesis} = CNS.synthesize(thesis, antithesis)
      assert is_binary(synthesis.claim)
      assert synthesis.confidence > 0
    end
  end

  describe "run/2" do
    test "runs pipeline with default config" do
      config = %CNS.Config{max_iterations: 2}
      assert {:ok, result} = CNS.run("Test question about coffee", config)
      assert Map.has_key?(result, :final_synthesis)
      assert Map.has_key?(result, :iterations)
    end
  end

  describe "version/0" do
    test "returns version string" do
      assert CNS.version() == "0.1.0"
    end
  end
end

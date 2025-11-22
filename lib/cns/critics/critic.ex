defmodule CNS.Critics.Critic do
  @moduledoc """
  Behaviour for CNS critics.

  Critics evaluate SNOs for specific quality dimensions:
  - Logic: Logical consistency, cycle detection, contradiction detection
  - Grounding: Factual accuracy, citation verification
  - Novelty: Originality, parsimony
  - Causal: Causal validity, correlation vs causation
  - Bias: Fairness, power shadow detection

  ## Implementing a Critic

      defmodule MyCritic do
        @behaviour CNS.Critics.Critic

        @impl true
        def name, do: :my_critic

        @impl true
        def weight, do: 0.2

        @impl true
        def evaluate(sno) do
          # Your evaluation logic
          {:ok, %{
            score: 0.85,
            issues: [],
            details: %{specific: "data"}
          }}
        end
      end
  """

  alias CNS.SNO

  @type evaluation_result :: %{
          score: float(),
          issues: [String.t()],
          details: map()
        }

  @doc """
  Evaluate an SNO and return quality metrics.
  """
  @callback evaluate(sno :: SNO.t()) ::
              {:ok, evaluation_result()} | {:error, term()}

  @doc """
  Return the critic's name as an atom.
  """
  @callback name() :: atom()

  @doc """
  Return the weight for this critic in aggregate scoring.
  """
  @callback weight() :: float()

  @doc """
  Optional initialization callback for GenServer-based critics.
  """
  @callback init_state(opts :: keyword()) :: map()

  @optional_callbacks [init_state: 1]
end

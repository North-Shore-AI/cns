defmodule CrucibleFramework.Ensemble.ML do
  @moduledoc """
  Behaviour contract for CrucibleFramework ensemble API.
  """

  @type infer_opts :: [
          strategy: :majority | :weighted_majority | :best_confidence | :unanimous,
          execution: :parallel | :sequential | :hedged | :cascade,
          timeout: pos_integer(),
          min_agreement: float()
        ]

  @type infer_result :: %{
          response: String.t(),
          confidence: float(),
          agreement: float(),
          model_responses: [%{model: String.t(), response: String.t(), confidence: float()}]
        }

  @callback infer(pool :: pid() | atom(), prompt :: String.t(), opts :: infer_opts()) ::
              {:ok, infer_result()} | {:error, term()}

  @callback infer_batch(pool :: pid() | atom(), prompts :: [String.t()], opts :: infer_opts()) ::
              {:ok, [infer_result()]} | {:error, term()}

  @callback create_pool(models :: [String.t()], opts :: keyword()) ::
              {:ok, pid()} | {:error, term()}

  @callback pool_status(pool :: pid() | atom()) ::
              {:ok, %{models: [String.t()], healthy: boolean()}} | {:error, term()}
end

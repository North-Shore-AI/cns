defmodule CrucibleFramework.Ensemble.ML do
  @moduledoc """
  Behaviour contract for CrucibleFramework ensemble API.

  DEPRECATED: This module represents the legacy Crucible contract interface.
  The Crucible Framework has moved to a new IR-based architecture.

  Migration path:
  - Use `CNS.CrucibleAdapter` for CNS integration with Crucible pipelines
  - Configure via: `config :crucible_framework, :cns_adapter, CNS.CrucibleAdapter`
  - See `Crucible.CNS.Adapter` behaviour for the new interface

  This module will be removed in a future version.
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

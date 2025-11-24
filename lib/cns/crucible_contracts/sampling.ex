defmodule CrucibleFramework.Sampling do
  @moduledoc """
  Behaviour contract for CrucibleFramework sampling API.
  CNS codes against this; mocked in tests.

  DEPRECATED: This module represents the legacy Crucible contract interface.
  The Crucible Framework has moved to a new IR-based architecture.

  Migration path:
  - Use `CNS.CrucibleAdapter` for CNS integration with Crucible pipelines
  - Configure via: `config :crucible_framework, :cns_adapter, CNS.CrucibleAdapter`
  - See `Crucible.CNS.Adapter` behaviour for the new interface

  This module will be removed in a future version.
  """
  @type sampling_params :: %{
          temperature: float(),
          max_tokens: pos_integer(),
          top_p: float(),
          stop_sequences: [String.t()]
        }

  @type response :: %{
          text: String.t(),
          tokens_used: non_neg_integer(),
          finish_reason: atom()
        }

  @callback generate(client :: pid() | atom(), prompt :: String.t(), params :: sampling_params()) ::
              {:ok, response()} | {:error, term()}

  @callback generate_batch(
              client :: pid() | atom(),
              prompts :: [String.t()],
              params :: sampling_params()
            ) ::
              {:ok, [response()]} | {:error, term()}

  @callback stream(client :: pid() | atom(), prompt :: String.t(), params :: sampling_params()) ::
              {:ok, Enumerable.t()} | {:error, term()}
end

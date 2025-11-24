defmodule CrucibleFramework.Lora do
  @moduledoc """
  Behaviour contract for CrucibleFramework LoRA training API.

  DEPRECATED: This module represents the legacy Crucible contract interface.
  The Crucible Framework has moved to a new IR-based architecture.

  Migration path:
  - Use `CNS.CrucibleAdapter` for CNS integration with Crucible pipelines
  - Configure via: `config :crucible_framework, :cns_adapter, CNS.CrucibleAdapter`
  - See `Crucible.CNS.Adapter` behaviour for the new interface

  This module will be removed in a future version.
  """
  @type config :: %{
          base_model: String.t(),
          lora_rank: pos_integer(),
          learning_rate: float(),
          batch_size: pos_integer(),
          num_epochs: pos_integer()
        }

  @type experiment :: %{
          id: String.t(),
          name: String.t(),
          config: config(),
          status: atom(),
          created_at: DateTime.t()
        }

  @type session :: pid()

  @callback create_experiment(opts :: keyword()) ::
              {:ok, experiment()} | {:error, term()}

  @callback start_session(experiment :: experiment()) ::
              {:ok, session()} | {:error, term()}

  @callback stop_session(session :: session()) ::
              :ok | {:error, term()}

  @callback train_step(session :: session(), batch :: [map()]) ::
              {:ok, %{loss: float(), step: non_neg_integer()}} | {:error, term()}

  @callback save_checkpoint(session :: session(), path :: String.t()) ::
              {:ok, String.t()} | {:error, term()}

  @callback load_checkpoint(session :: session(), path :: String.t()) ::
              :ok | {:error, term()}
end

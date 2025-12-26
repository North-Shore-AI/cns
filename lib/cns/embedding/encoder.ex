defmodule CNS.Embedding.Encoder do
  @moduledoc """
  Thin indirection layer for claim embeddings.

  Delegates to the module configured under `:cns, :embedding_provider` (must
  implement `encode/1`). This keeps CNS decoupled from any specific embedding
  backend (Gemini, OpenAI, Bumblebee, TEI, etc).
  """

  require Logger

  @type provider :: module()

  @spec encode(String.t()) :: {:ok, list(number())} | {:error, term()}
  def encode(text) when is_binary(text) do
    with module when not is_nil(module) <- provider(),
         {:module, _} <- Code.ensure_loaded(module),
         true <- function_exported?(module, :encode, 1) do
      Logger.info("[CNS.Embedding.Encoder] provider=#{inspect(module)} len=#{String.length(text)}")
      module.encode(text)
    else
      nil -> {:error, :no_embedding_provider}
      {:error, _} -> {:error, {:invalid_provider, provider()}}
      false -> {:error, {:invalid_provider, provider()}}
    end
  end

  def encode(_), do: {:error, :invalid_input}

  defp provider, do: Application.get_env(:cns, :embedding_provider)
end

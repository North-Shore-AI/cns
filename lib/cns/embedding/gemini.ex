defmodule CNS.Embedding.Gemini do
  @moduledoc """
  Gemini-based embedding provider using `gemini_ex`.

  Configure with:

      config :cns, :embedding_provider, CNS.Embedding.Gemini
      config :cns, CNS.Embedding.Gemini,
        model: "text-embedding-004",
        output_dimensionality: 768

  Requires `GEMINI_API_KEY` in the environment and the `gemini_ex` dependency.
  """

  @compile {:no_warn_undefined, Gemini}
  require Logger

  @default_model "text-embedding-004"

  @spec encode(String.t()) :: {:ok, list(number())} | {:error, term()}
  def encode(text) when is_binary(text) do
    unless Code.ensure_loaded?(Gemini) do
      {:error, :gemini_ex_not_loaded}
    else
      api_key = System.get_env("GEMINI_API_KEY")

      if is_nil(api_key) or api_key == "" do
        {:error, :missing_gemini_api_key}
      else
        opts = Application.get_env(:cns, __MODULE__, [])
        model = Keyword.get(opts, :model, @default_model)
        output_dim = Keyword.get(opts, :output_dimensionality)

        request_opts =
          []
          |> maybe_put(:model, model)
          |> maybe_put(:output_dimensionality, output_dim)

        Logger.info("[CNS.Embedding.Gemini] model=#{model} dim=#{output_dim || "default"}")

        case Gemini.embed_content(text, request_opts) do
          {:ok, %{embedding: %{values: values}}} when is_list(values) ->
            {:ok, values}

          {:ok, %{embedding: values}} when is_list(values) ->
            {:ok, values}

          {:ok, resp} ->
            Logger.error("[CNS.Embedding.Gemini] unexpected response: #{inspect(resp)}")
            {:error, {:unexpected_response, resp}}

          {:error, reason} ->
            {:error, reason}
        end
      end
    end
  end

  def encode(_), do: {:error, :invalid_input}

  defp maybe_put(keyword, _key, nil), do: keyword
  defp maybe_put(keyword, key, value), do: Keyword.put(keyword, key, value)
end

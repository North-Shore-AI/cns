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

  @compile {:no_warn_undefined, [Gemini, Gemini.Types.Response.EmbedContentResponse, Gemini.Error]}
  alias Gemini.Types.Response.EmbedContentResponse
  require Logger

  @default_model "text-embedding-004"
  @typep embed_result :: {:ok, map()} | {:error, Gemini.Error.t()}

  @spec encode(String.t()) :: {:ok, list(number())} | {:error, term()}
  def encode(text) when is_binary(text) do
    with true <- Code.ensure_loaded?(Gemini),
         api_key when not is_nil(api_key) and api_key != "" <- System.get_env("GEMINI_API_KEY") do
      perform_encoding(text, api_key)
    else
      false -> {:error, :gemini_ex_not_loaded}
      nil -> {:error, :missing_gemini_api_key}
      "" -> {:error, :missing_gemini_api_key}
    end
  end

  def encode(_), do: {:error, :invalid_input}

  defp perform_encoding(text, _api_key) do
    opts = Application.get_env(:cns, __MODULE__, [])
    model = Keyword.get(opts, :model, @default_model)
    output_dim = Keyword.get(opts, :output_dimensionality)

    request_opts =
      []
      |> maybe_put(:model, model)
      |> maybe_put(:output_dimensionality, output_dim)

    Logger.info("[CNS.Embedding.Gemini] model=#{model} dim=#{output_dim || "default"}")
    handle_embed_response(embed_content(text, request_opts))
  end

  defp handle_embed_response({:ok, resp}) when is_struct(resp) do
    {:ok, EmbedContentResponse.get_values(resp)}
  end

  defp handle_embed_response({:error, reason}), do: {:error, reason}

  defp handle_embed_response(other) do
    Logger.error("[CNS.Embedding.Gemini] unexpected response: #{inspect(other)}")
    {:error, {:unexpected_response, other}}
  end

  @spec embed_content(String.t(), keyword()) :: embed_result
  defp embed_content(text, opts) do
    Gemini.embed_content(text, opts)
  end

  defp maybe_put(keyword, _key, nil), do: keyword
  defp maybe_put(keyword, key, value), do: Keyword.put(keyword, key, value)
end

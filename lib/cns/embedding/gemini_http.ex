defmodule CNS.Embedding.GeminiHTTP do
  @moduledoc """
  Minimal Gemini embedding provider via REST (bypasses gemini_ex auth heuristics).

  Requires `GEMINI_API_KEY` and the optional `req` dependency.
  Configurable model/dimension via:

      config :cns, :embedding_provider, CNS.Embedding.GeminiHTTP
      config :cns, CNS.Embedding.GeminiHTTP,
        model: "text-embedding-004",
        output_dimensionality: 768
  """

  require Logger

  @endpoint "https://generativelanguage.googleapis.com/v1beta/models"
  @default_model "text-embedding-004"

  @doc """
  Check if Req HTTP client is available (optional dependency).
  """
  def req_available? do
    Code.ensure_loaded?(Req)
  end

  @spec encode(String.t()) :: {:ok, list(number())} | {:error, term()}
  def encode(text) when is_binary(text) do
    if req_available?() do
      do_encode(text)
    else
      {:error, :req_not_available}
    end
  end

  def encode(_), do: {:error, :invalid_input}

  defp do_encode(text) do
    api_key =
      System.get_env("GEMINI_API_KEY") ||
        Application.get_env(:gemini_ex, :api_key) ||
        Application.get_env(:cns, :gemini_api_key)

    if is_nil(api_key) or api_key == "" do
      {:error, :missing_gemini_api_key}
    else
      opts = Application.get_env(:cns, __MODULE__, [])
      model = Keyword.get(opts, :model, @default_model)
      output_dim = Keyword.get(opts, :output_dimensionality)

      url = "#{@endpoint}/#{model}:embedContent?key=#{api_key}"

      body =
        %{
          model: model,
          content: %{parts: [%{text: text}]}
        }
        |> maybe_put(:outputDimensionality, output_dim)

      Logger.info("[CNS.Embedding.GeminiHTTP] model=#{model} dim=#{output_dim || "default"}")

      case Req.post(url: url, json: body) do
        {:ok, %{status: 200, body: %{"embedding" => %{"values" => values}}}} when is_list(values) ->
          {:ok, values}

        {:ok, %{status: status, body: resp_body}} ->
          Logger.error(
            "[CNS.Embedding.GeminiHTTP] unexpected response status=#{status} body=#{inspect(resp_body)}"
          )

          {:error, {:http_error, status, resp_body}}

        {:error, reason} ->
          {:error, reason}
      end
    end
  end

  defp maybe_put(map, _k, nil), do: map
  defp maybe_put(map, k, v), do: Map.put(map, k, v)
end

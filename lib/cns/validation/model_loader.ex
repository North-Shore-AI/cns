defmodule CNS.Validation.ModelLoader do
  @moduledoc """
  GenServer for lazily loading and caching NLI and embedding models.

  Models are loaded on first use and cached for subsequent calls.
  This avoids startup delays when validation is not used.

  ## Models Used

  - **NLI Model**: `cross-encoder/nli-deberta-v3-base` for entailment scoring
    - Classifies premise/hypothesis as entailment/neutral/contradiction
    - Smaller than large variant but still accurate

  - **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` for semantic similarity
    - Fast, lightweight (80MB) sentence embeddings
    - Good balance of speed and quality

  ## Usage

      # The GenServer is started automatically by CNS.Application
      # Get models on demand:
      {:ok, nli} = CNS.Validation.ModelLoader.get_nli_model()
      {:ok, embedding} = CNS.Validation.ModelLoader.get_embedding_model()
  """

  use GenServer
  require Logger

  # BART-large-MNLI is well-supported by Bumblebee for NLI
  # Alternative: facebook/bart-large-mnli
  @nli_model_repo "facebook/bart-large-mnli"

  # MiniLM for embeddings - lightweight and fast
  @embedding_model_repo "sentence-transformers/all-MiniLM-L6-v2"

  # Client API

  @doc """
  Starts the model loader GenServer.
  """
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Check if models are enabled (can be disabled for testing).
  """
  def models_enabled? do
    Application.get_env(:cns, :enable_ml_models, true) and bumblebee_available?()
  end

  @doc """
  Check if Bumblebee is available (optional dependency).
  """
  def bumblebee_available? do
    Code.ensure_loaded?(Bumblebee)
  end

  @doc """
  Get the NLI model for entailment scoring.

  Returns {:ok, serving} on success.
  Loads the model lazily on first call.
  """
  @spec get_nli_model() :: {:ok, Nx.Serving.t()} | {:error, term()}
  def get_nli_model do
    if models_enabled?() do
      GenServer.call(__MODULE__, :get_nli_model, :infinity)
    else
      {:error, :models_disabled}
    end
  end

  @doc """
  Get the embedding model for semantic similarity.

  Returns {:ok, serving} where serving is a Nx.Serving for text embeddings.
  Loads the model lazily on first call.
  """
  @spec get_embedding_model() :: {:ok, Nx.Serving.t()} | {:error, term()}
  def get_embedding_model do
    if models_enabled?() do
      GenServer.call(__MODULE__, :get_embedding_model, :infinity)
    else
      {:error, :models_disabled}
    end
  end

  @doc """
  Check if models are loaded.
  """
  @spec models_loaded?() :: %{nli: boolean(), embedding: boolean()}
  def models_loaded? do
    GenServer.call(__MODULE__, :models_loaded?)
  end

  @doc """
  Preload all models. Useful for warming up before inference.
  """
  @spec preload_all() :: :ok | {:error, term()}
  def preload_all do
    with {:ok, _} <- get_nli_model(),
         {:ok, _} <- get_embedding_model() do
      :ok
    end
  end

  # Server Callbacks

  @impl true
  def init(_opts) do
    state = %{
      nli_model: nil,
      embedding_model: nil
    }

    {:ok, state}
  end

  @impl true
  def handle_call(:get_nli_model, _from, %{nli_model: nil} = state) do
    Logger.info("[ModelLoader] Loading NLI model: #{@nli_model_repo}")

    case load_nli_model() do
      {:ok, model_data} ->
        {:reply, {:ok, model_data}, %{state | nli_model: model_data}}

      {:error, reason} = error ->
        Logger.error("[ModelLoader] Failed to load NLI model: #{inspect(reason)}")
        {:reply, error, state}
    end
  end

  def handle_call(:get_nli_model, _from, %{nli_model: model} = state) when not is_nil(model) do
    {:reply, {:ok, model}, state}
  end

  @impl true
  def handle_call(:get_embedding_model, _from, %{embedding_model: nil} = state) do
    Logger.info("[ModelLoader] Loading embedding model: #{@embedding_model_repo}")

    case load_embedding_model() do
      {:ok, serving} ->
        {:reply, {:ok, serving}, %{state | embedding_model: serving}}

      {:error, reason} = error ->
        Logger.error("[ModelLoader] Failed to load embedding model: #{inspect(reason)}")
        {:reply, error, state}
    end
  end

  def handle_call(:get_embedding_model, _from, %{embedding_model: model} = state)
      when not is_nil(model) do
    {:reply, {:ok, model}, state}
  end

  @impl true
  def handle_call(:models_loaded?, _from, state) do
    status = %{
      nli: state.nli_model != nil,
      embedding: state.embedding_model != nil
    }

    {:reply, status, state}
  end

  # Private Functions

  defp load_nli_model do
    if bumblebee_available?() do
      try do
        {:ok, model_info} =
          apply(Bumblebee, :load_model, [
            {:hf, @nli_model_repo},
            [architecture: :for_sequence_classification]
          ])

        {:ok, tokenizer} = apply(Bumblebee, :load_tokenizer, [{:hf, @nli_model_repo}])

        # Use standard NLI text classification; model outputs entailment/neutral/contradiction
        serving =
          apply(Bumblebee.Text, :text_classification, [
            model_info,
            tokenizer,
            [
              compile: [batch_size: 1, sequence_length: 512],
              defn_options: [compiler: EXLA]
            ]
          ])

        {:ok, serving}
      rescue
        e ->
          {:error, Exception.message(e)}
      end
    else
      {:error, :bumblebee_not_available}
    end
  end

  defp load_embedding_model do
    if bumblebee_available?() do
      try do
        {:ok, model_info} = apply(Bumblebee, :load_model, [{:hf, @embedding_model_repo}])
        {:ok, tokenizer} = apply(Bumblebee, :load_tokenizer, [{:hf, @embedding_model_repo}])

        # Create serving for text embedding
        serving =
          apply(Bumblebee.Text, :text_embedding, [
            model_info,
            tokenizer,
            [
              compile: [batch_size: 2, sequence_length: 256],
              defn_options: [compiler: EXLA],
              output_pool: :mean_pooling,
              output_attribute: :hidden_state
            ]
          ])

        {:ok, serving}
      rescue
        e ->
          {:error, Exception.message(e)}
      end
    else
      {:error, :bumblebee_not_available}
    end
  end
end

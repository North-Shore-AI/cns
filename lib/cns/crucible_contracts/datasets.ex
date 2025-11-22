defmodule CrucibleFramework.Datasets do
  @moduledoc """
  Behaviour contract for CrucibleFramework dataset loading.
  """

  @type dataset_opts :: [
          split: :train | :dev | :test,
          limit: pos_integer() | nil,
          shuffle: boolean(),
          seed: integer()
        ]

  @type dataset_info :: %{
          name: atom(),
          size: non_neg_integer(),
          splits: [:train | :dev | :test],
          features: [String.t()]
        }

  @callback load(name :: atom(), opts :: dataset_opts()) ::
              {:ok, [map()]} | {:error, term()}

  @callback stream(name :: atom(), opts :: dataset_opts()) ::
              {:ok, Enumerable.t()} | {:error, term()}

  @callback info(name :: atom()) ::
              {:ok, dataset_info()} | {:error, term()}

  @callback available() ::
              [atom()]
end

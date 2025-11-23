defmodule CNS.Application do
  @moduledoc false

  use Application

  @impl true
  def start(_type, _args) do
    children = [
      # Model loader for semantic validation (lazy loading)
      CNS.Validation.ModelLoader
    ]

    opts = [strategy: :one_for_one, name: CNS.Supervisor]
    Supervisor.start_link(children, opts)
  end
end

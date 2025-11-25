defmodule CNS.Pipeline do
  @moduledoc false
  @deprecated "Use CNS.Agents.Pipeline or the high-level CNS.run_pipeline/2"

  defdelegate run(input, config), to: CNS.Agents.Pipeline
  defdelegate iterate(state, config), to: CNS.Agents.Pipeline
  defdelegate check_convergence(prev, current, config), to: CNS.Agents.Pipeline
end
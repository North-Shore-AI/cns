defmodule CNS.Antagonist do
  @moduledoc false
  @deprecated "Use CNS.Agents.Antagonist or the high-level CNS API"

  defdelegate challenge(sno, opts \\ []), to: CNS.Agents.Antagonist
  defdelegate generate_counter_evidence(claim, opts \\ []), to: CNS.Agents.Antagonist
  defdelegate evaluate_weakness(sno, opts \\ []), to: CNS.Agents.Antagonist
end
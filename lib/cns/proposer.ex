defmodule CNS.Proposer do
  @moduledoc false
  @deprecated "Use CNS.Agents.Proposer or the high-level CNS.extract_claims/2"

  defdelegate extract_claims(text, opts \\ []), to: CNS.Agents.Proposer
  defdelegate generate_hypothesis(question, opts \\ []), to: CNS.Agents.Proposer
  defdelegate extract_evidence(text, opts \\ []), to: CNS.Agents.Proposer
end
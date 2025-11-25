defmodule CNS.Synthesizer do
  @moduledoc false
  @deprecated "Use CNS.Agents.Synthesizer or the high-level CNS.synthesize/3"

  defdelegate synthesize(thesis, antithesis, opts \\ []), to: CNS.Agents.Synthesizer
  defdelegate merge_evidence(snos, opts \\ []), to: CNS.Agents.Synthesizer
  defdelegate resolve_conflicts(sno_a, sno_b, opts \\ []), to: CNS.Agents.Synthesizer
end
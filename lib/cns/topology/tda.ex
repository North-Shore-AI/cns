defmodule CNS.Topology.TDA do
  @moduledoc """
  Deprecated thin wrapper around `CNS.Topology.Persistence`.

  The original stub implementation lived here; the production-grade persistent
  homology now lives in `CNS.Topology.Persistence` (powered by `ex_topology`).
  Use that module directly for new code. These functions delegate for
  compatibility.
  """

  alias CNS.SNO
  alias CNS.Topology.Persistence

  @deprecated "Use CNS.Topology.Persistence.compute/2"
  @spec compute_for_snos([SNO.t()], keyword()) :: Persistence.persistence_result()
  def compute_for_snos(snos, opts \\ []), do: Persistence.compute(List.wrap(snos), opts)

  @deprecated "Use CNS.Topology.Persistence.compute/2"
  @spec compute_for_sno(SNO.t(), keyword()) :: Persistence.persistence_result()
  def compute_for_sno(sno, opts \\ []), do: Persistence.compute([sno], opts)

  @deprecated "Use CNS.Topology.Persistence.summary/1"
  @spec summarize(Persistence.persistence_result(), keyword()) :: Persistence.summary()
  def summarize(result, _opts \\ []), do: Persistence.summary(result)
end

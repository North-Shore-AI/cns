# Claim Parsing Guide

## Overview

The `CNS.Schema.Parser` module extracts structured claims and relations from LLM output text. It recognizes multiple formats and normalizes them for downstream processing.

## Supported Formats

### Claims

```
CLAIM[id]: text content
CLAIM[id] (Document 12345): text with evidence reference
claim[ID]: case-insensitive
```

### Relations

```
RELATION: source_id label target_id
RELATION - source_id label target_id
relation: source SUPPORTS target
```

Labels: `supports`, `refutes`, `contrasts`

## Basic Usage

### Parsing Claims

```elixir
alias CNS.Schema.Parser

text = """
CLAIM[c1]: Main hypothesis about the topic
CLAIM[c2] (Document 123): Supporting evidence from source
CLAIM[c3] (Document 456): Counter-evidence
"""

claims = Parser.parse_claims(text)

# Result: %{
#   "c1" => %Parser.Claim{identifier: "c1", text: "Main hypothesis...", document_id: nil},
#   "c2" => %Parser.Claim{identifier: "c2", text: "Supporting evidence...", document_id: "123"},
#   "c3" => %Parser.Claim{identifier: "c3", text: "Counter-evidence", document_id: "456"}
# }
```

### Parsing Relations

```elixir
text = """
RELATION: c2 supports c1
RELATION - c3 refutes c1
"""

relations = Parser.parse_relations(text)

# Result: [
#   {"c2", "supports", "c1"},
#   {"c3", "refutes", "c1"}
# ]
```

### Parsing Complete Output

```elixir
{claims, relations} = Parser.parse(complete_output)
```

## The Claim Struct

```elixir
%CNS.Schema.Parser.Claim{
  identifier: String.t(),      # Claim ID (e.g., "c1")
  text: String.t(),            # Claim content
  document_id: String.t() | nil # Optional evidence document reference
}
```

## Relation Format

Relations are 3-tuples: `{source_id, label, target_id}`

- **source_id**: The claim providing support/refutation
- **label**: Semantic relationship (`"supports"`, `"refutes"`, `"contrasts"`)
- **target_id**: The claim being supported/refuted

## Edge Cases

### Case Insensitivity

Keywords are case-insensitive:

```elixir
Parser.parse_claims("claim[c1]: lowercase")
Parser.parse_claims("CLAIM[c1]: uppercase")
Parser.parse_claims("Claim[c1]: mixed")
# All produce the same result
```

### Whitespace Handling

Leading/trailing whitespace is trimmed:

```elixir
Parser.parse_claims("CLAIM[c1]  :   text with spaces   ")
# text field: "text with spaces"
```

### Complex Claim IDs

IDs can contain letters, numbers, and underscores:

```elixir
Parser.parse_claims("CLAIM[claim_1_final]: Complex ID")
# identifier: "claim_1_final"
```

## Working with Parsed Data

### Building a Claim Graph

```elixir
alias CNS.Logic.Betti

{claims, relations} = Parser.parse(output)
claim_ids = Map.keys(claims)

stats = Betti.compute_graph_stats(claim_ids, relations)
```

### Extracting Evidence Documents

```elixir
evidence_docs =
  claims
  |> Map.values()
  |> Enum.filter(& &1.document_id)
  |> Enum.map(& &1.document_id)
  |> MapSet.new()
```

### Validating Relation Targets

```elixir
claim_ids = MapSet.new(Map.keys(claims))

invalid_relations =
  relations
  |> Enum.filter(fn {src, _label, dst} ->
    not MapSet.member?(claim_ids, src) or not MapSet.member?(claim_ids, dst)
  end)
```

## Integration Example

Complete workflow parsing and analyzing LLM output:

```elixir
defmodule MyApp.ClaimAnalyzer do
  alias CNS.Schema.Parser
  alias CNS.Logic.Betti

  def analyze(llm_output) do
    {claims, relations} = Parser.parse(llm_output)

    # Build topology stats
    claim_ids = Map.keys(claims)
    stats = Betti.compute_graph_stats(claim_ids, relations)

    %{
      num_claims: map_size(claims),
      num_relations: length(relations),
      has_cycles: stats.beta1 > 0,
      has_conflict: stats.polarity_conflict,
      evidence_docs: extract_evidence_docs(claims)
    }
  end

  defp extract_evidence_docs(claims) do
    claims
    |> Map.values()
    |> Enum.map(& &1.document_id)
    |> Enum.reject(&is_nil/1)
  end
end
```

## Error Handling

The parser is lenient and skips unrecognized lines:

```elixir
text = """
Some random text
CLAIM[c1]: Valid claim
More random text
RELATION: c2 supports c1
"""

{claims, relations} = Parser.parse(text)
# Only valid lines are extracted
```

Non-matching lines return nil for single-line parsing:

```elixir
Parser.parse_relation("not a relation")
# nil

Parser.parse_claim_line("random text")
# nil
```

defmodule CNS.Schema.Parser do
  @moduledoc """
  Parse CLAIM[...] and RELATION formatted completions from LLM output.

  Supports formats:
  - CLAIM[id]: text
  - CLAIM[id] (Document 12345): text
  - RELATION: src label dst
  - RELATION - src label dst
  """

  defmodule Claim do
    @moduledoc "Parsed claim from LLM output"

    @type t :: %__MODULE__{
            identifier: String.t(),
            text: String.t(),
            document_id: String.t() | nil
          }

    @enforce_keys [:identifier, :text]
    defstruct [:identifier, :text, :document_id]
  end

  @type relation :: {String.t(), String.t(), String.t()}

  # Regex patterns
  @claim_pattern ~r/^CLAIM\[(?<id>[^\]]+)\]\s*(?:\(Document\s+(?<doc>\d+)\))?\s*:\s*(?<body>.*)$/i
  @relation_pattern ~r/^RELATION\s*[:\-]?\s*(?<src>\S+)\s+(?<label>supports|refutes|contrasts)\s+(?<dst>\S+)/i

  @doc """
  Parse all CLAIM lines from text.

  ## Parameters
    - input: Input text (string or list of lines)

  ## Returns
    Map of claim_id => Claim struct

  ## Examples

      iex> text = "CLAIM[c1]: The sky is blue"
      iex> CNS.Schema.Parser.parse_claims(text)
      %{"c1" => %CNS.Schema.Parser.Claim{identifier: "c1", text: "The sky is blue", document_id: nil}}
  """
  @spec parse_claims(String.t() | [String.t()]) :: %{String.t() => Claim.t()}
  def parse_claims(input) when is_binary(input) do
    input
    |> String.split("\n")
    |> parse_claims()
  end

  def parse_claims(lines) when is_list(lines) do
    lines
    |> Enum.reduce(%{}, fn line, acc ->
      case parse_claim_line(line) do
        nil -> acc
        %Claim{} = claim -> Map.put(acc, claim.identifier, claim)
      end
    end)
  end

  defp parse_claim_line(line) do
    case Regex.named_captures(@claim_pattern, String.trim(line)) do
      nil ->
        nil

      %{"id" => id, "body" => body} = captures ->
        doc_id = Map.get(captures, "doc")

        %Claim{
          identifier: String.trim(id),
          text: String.trim(body),
          document_id: if(doc_id && doc_id != "", do: doc_id, else: nil)
        }
    end
  end

  @doc """
  Parse a single RELATION line.

  ## Returns
    - `{source, label, target}` tuple if valid
    - `nil` if not a relation line

  ## Examples

      iex> CNS.Schema.Parser.parse_relation("RELATION: c2 supports c1")
      {"c2", "supports", "c1"}

      iex> CNS.Schema.Parser.parse_relation("not a relation")
      nil
  """
  @spec parse_relation(String.t()) :: relation() | nil
  def parse_relation(line) do
    case Regex.named_captures(@relation_pattern, String.trim(line)) do
      nil ->
        nil

      %{"src" => src, "label" => label, "dst" => dst} ->
        {
          String.trim(src),
          String.downcase(String.trim(label)),
          String.trim(dst)
        }
    end
  end

  @doc """
  Parse all relations from text.

  ## Examples

      iex> text = "RELATION: c2 supports c1\\nRELATION: c3 refutes c1"
      iex> CNS.Schema.Parser.parse_relations(text)
      [{"c2", "supports", "c1"}, {"c3", "refutes", "c1"}]
  """
  @spec parse_relations(String.t() | [String.t()]) :: [relation()]
  def parse_relations(input) when is_binary(input) do
    input
    |> String.split("\n")
    |> parse_relations()
  end

  def parse_relations(lines) when is_list(lines) do
    lines
    |> Enum.map(&parse_relation/1)
    |> Enum.reject(&is_nil/1)
  end

  @doc """
  Parse complete output into claims and relations.

  ## Examples

      iex> text = "CLAIM[c1]: Test\\nRELATION: c2 supports c1"
      iex> {claims, relations} = CNS.Schema.Parser.parse(text)
      iex> map_size(claims)
      1
      iex> length(relations)
      1
  """
  @spec parse(String.t()) :: {%{String.t() => Claim.t()}, [relation()]}
  def parse(text) do
    lines = String.split(text, "\n")
    claims = parse_claims(lines)
    relations = parse_relations(lines)
    {claims, relations}
  end
end

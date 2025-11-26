defmodule CNS.MixProject do
  use Mix.Project

  @version "0.1.0"
  @source_url "https://github.com/North-Shore-AI/cns"

  def project do
    [
      app: :cns,
      version: @version,
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      package: package(),
      docs: docs(),
      name: "CNS",
      description:
        "Chiral Narrative Synthesis - Dialectical reasoning framework for automated knowledge discovery",
      source_url: @source_url,
      homepage_url: @source_url,
      elixirc_paths: elixirc_paths(Mix.env()),
      test_coverage: [tool: ExCoveralls],
      preferred_cli_env: [
        coveralls: :test,
        "coveralls.detail": :test,
        "coveralls.post": :test,
        "coveralls.html": :test
      ],
      dialyzer: [
        plt_file: {:no_warn, "priv/plts/dialyzer.plt"},
        plt_add_apps: [:mix, :ex_unit],
        ignore_warnings: ".dialyzer_ignore.exs"
      ]
    ]
  end

  def application do
    [
      extra_applications: [:logger],
      mod: {CNS.Application, []}
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  defp deps do
    [
      # Core dependencies
      {:nx, "~> 0.7"},
      {:jason, "~> 1.4"},
      {:nimble_parsec, "~> 1.4"},
      {:uuid, "~> 1.1"},
      {:libgraph, "~> 0.16"},
      {:telemetry, "~> 1.2"},
      {:ex_topology, "~> 0.1.1"},
      {:gemini_ex, "~> 0.4", optional: true},
      {:req, "~> 0.5", optional: true},

      # ML/NLP dependencies for semantic validation (optional)
      {:bumblebee, "~> 0.5", optional: true},
      {:exla, "~> 0.7", optional: true},

      # Optional: Tinkex integration (umbrella sibling)
      # {:tinkex, in_umbrella: true, optional: true},

      # Development and testing
      {:ex_doc, "~> 0.31", only: :dev, runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:excoveralls, "~> 0.18", only: :test},
      {:stream_data, "~> 1.0", only: [:dev, :test]},
      {:mox, "~> 1.1", only: :test}
    ]
  end

  defp package do
    [
      name: "cns",
      description:
        "Chiral Narrative Synthesis - Dialectical reasoning framework for automated knowledge discovery",
      licenses: ["Apache-2.0"],
      links: %{
        "GitHub" => @source_url,
        "Docs" => "https://hexdocs.pm/cns"
      },
      maintainers: ["North-Shore-AI"],
      files: ~w(lib docs .formatter.exs mix.exs README.md LICENSE CHANGELOG.md)
    ]
  end

  defp docs do
    [
      main: "readme",
      name: "CNS",
      source_ref: "v#{@version}",
      source_url: @source_url,
      extras: [
        "README.md",
        "docs/guides/getting_started.md",
        "docs/guides/claim_parsing.md",
        "docs/guides/topology_analysis.md",
        "docs/guides/validation_pipeline.md",
        "docs/guides/data_pipeline.md",
        "docs/guides/api_reference.md"
      ],
      groups_for_extras: [
        Guides: [
          "docs/guides/getting_started.md",
          "docs/guides/claim_parsing.md",
          "docs/guides/topology_analysis.md",
          "docs/guides/validation_pipeline.md",
          "docs/guides/data_pipeline.md",
          "docs/guides/api_reference.md"
        ]
      ],
      groups_for_modules: [
        "Core Types": [
          CNS.SNO,
          CNS.Evidence,
          CNS.Challenge,
          CNS.Provenance
        ],
        "Schema & Parsing": [
          CNS.Schema.Parser
        ],
        "Logic & Topology": [
          CNS.Logic.Betti,
          CNS.Topology,
          CNS.Topology.TDA
        ],
        Metrics: [
          CNS.Metrics.Chirality,
          CNS.Metrics
        ],
        Validation: [
          CNS.Validation.Semantic
        ],
        Pipeline: [
          CNS.Pipeline.Schema,
          CNS.Pipeline.Converters,
          CNS.Proposer,
          CNS.Antagonist,
          CNS.Synthesizer,
          CNS.Pipeline
        ],
        Training: [
          CNS.Training.Evaluation,
          CNS.Training
        ],
        Configuration: [
          CNS.Config
        ]
      ]
    ]
  end
end

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
      files: ~w(lib .formatter.exs mix.exs README.md LICENSE CHANGELOG.md)
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
        "docs/20251121/architecture.md",
        "docs/20251121/api_reference.md",
        "docs/20251121/training_guide.md",
        "docs/20251121/getting_started.md"
      ],
      groups_for_modules: [
        "Core Types": [
          CNS.SNO,
          CNS.Evidence,
          CNS.Challenge,
          CNS.Provenance
        ],
        Pipeline: [
          CNS.Proposer,
          CNS.Antagonist,
          CNS.Synthesizer,
          CNS.Pipeline
        ],
        Metrics: [
          CNS.Metrics,
          CNS.Topology
        ],
        Training: [
          CNS.Training
        ],
        Configuration: [
          CNS.Config
        ]
      ]
    ]
  end
end

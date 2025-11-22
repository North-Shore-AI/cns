import Config

# CNS - Chiral Narrative Synthesis Configuration

# Default module configuration (production uses real implementations)
config :cns,
  sampling_module: CrucibleFramework.Sampling,
  lora_module: CrucibleFramework.Lora,
  ensemble_module: CrucibleFramework.Ensemble.ML,
  datasets_module: CrucibleFramework.Datasets

# Logger configuration
config :logger, :console,
  format: "$time $metadata[$level] $message\n",
  metadata: [:request_id]

# Import environment specific config
import_config "#{config_env()}.exs"

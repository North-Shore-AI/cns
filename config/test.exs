import Config

# Test configuration - use mocks for CrucibleFramework
config :cns,
  sampling_module: CrucibleFramework.SamplingMock,
  lora_module: CrucibleFramework.LoraMock,
  ensemble_module: CrucibleFramework.Ensemble.MLMock,
  datasets_module: CrucibleFramework.DatasetsMock,
  # Disable ML models in tests by default (use fallback heuristics)
  # Set to true to test with real model inference
  enable_ml_models: false

# Reduce log noise in tests
config :logger, level: :warning

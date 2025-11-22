import Config

# Test configuration - use mocks for CrucibleFramework
config :cns,
  sampling_module: CrucibleFramework.SamplingMock,
  lora_module: CrucibleFramework.LoraMock,
  ensemble_module: CrucibleFramework.Ensemble.MLMock,
  datasets_module: CrucibleFramework.DatasetsMock

# Reduce log noise in tests
config :logger, level: :warning

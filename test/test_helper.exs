# Define mocks before ExUnit.start()
Mox.defmock(CrucibleFramework.SamplingMock, for: CrucibleFramework.Sampling)
Mox.defmock(CrucibleFramework.LoraMock, for: CrucibleFramework.Lora)
Mox.defmock(CrucibleFramework.Ensemble.MLMock, for: CrucibleFramework.Ensemble.ML)
Mox.defmock(CrucibleFramework.DatasetsMock, for: CrucibleFramework.Datasets)

ExUnit.start()

# Configure ExUnit
ExUnit.configure(
  exclude: [:skip, :integration],
  formatters: [ExUnit.CLIFormatter]
)

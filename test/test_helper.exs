ExUnit.start()

# Configure ExUnit
ExUnit.configure(
  exclude: [:skip, :integration],
  formatters: [ExUnit.CLIFormatter]
)

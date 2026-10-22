# Contributing

Evo-Split is currently maintained as a research prototype. Bug reports and
small, focused pull requests are welcome.

Before submitting a change:

1. Build with `EVOSPLIT_WITH_MNN=OFF`.
2. Run all host-independent tests with CTest.
3. Run the minimal demo.
4. Do not commit model weights, datasets, build products, device identifiers,
   public IP addresses, credentials, or machine-specific absolute paths.
5. Keep public comments, diagnostics, documentation, and commit messages in
   English.

Changes to wire formats, semantic-boundary contracts, or update transaction
semantics should include a regression test and migration note.


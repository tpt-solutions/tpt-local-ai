# Contributing to tpt-local-ai

Thanks for your interest in improving `tpt-local-ai`! This is a Cargo workspace
of five small, dependency-conscious crates. Contributions of all sizes are
welcome.

## Local development setup

You only need a stable Rust toolchain (the MSRV is **1.80.0**):

```sh
rustup toolchain install stable
rustup component add rustfmt clippy
git clone https://github.com/tpt-ai/tpt-local-ai
cd tpt-local-ai
```

## The checks CI runs

Before opening a pull request, please run the same checks CI does. All of these
must pass:

```sh
# Formatting
cargo fmt --all --check

# Lints (warnings are denied)
cargo clippy --workspace --all-targets --all-features -- -D warnings

# Tests (no network access required; Hub tests use a mock server)
cargo test --workspace --all-features

# Docs build cleanly (docs.rs-ready)
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features
```

On Windows PowerShell, set `RUSTDOCFLAGS` with `$env:RUSTDOCFLAGS="-D warnings"`
before the `cargo doc` line.

### Supply-chain checks (optional but appreciated)

```sh
cargo install cargo-deny --locked
cargo deny check
```

## Running the examples

Every crate ships at least one runnable example, and the end-to-end pipeline
lives in the `cookbook` crate:

```sh
cargo run -p tpt-cookbook
cargo run -p tpt-jinja-chat --example render_llama3_template
cargo run -p tpt-lora-merge --example merge_lora_example
```

## Guidelines

- **Keep dependencies lean.** `tpt-jinja-chat` and `tpt-tokenizer-core` are
  dependency-free by design; `tpt-tokenizer-core` is also `no_std + alloc`. Do
  not add dependencies to these without discussion.
- **No network in tests.** The Hub client is tested against a mocked HTTP
  server; never add tests that hit the real network.
- **Document public items.** All crates use `#![warn(missing_docs)]`.
- **Security-sensitive parsing.** `tpt-safetensors-io` and `tpt-jinja-chat`
  parse untrusted input and have `cargo-fuzz` targets under `fuzz/`. If you
  touch the parsers, consider running the fuzzers.
- **Add tests** for new behavior and regressions.

## Publishing

Publishing is coordinated by maintainers in dependency order:
`tpt-hf-hub` → `tpt-jinja-chat` → `tpt-tokenizer-core` → `tpt-safetensors-io` →
`tpt-lora-merge`.

## License

By contributing, you agree that your contributions will be dual licensed under
MIT and Apache-2.0, matching the rest of the project.

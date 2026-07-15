# Contributing to tpt-local-ai

Thanks for your interest in `tpt-local-ai`!

## How to contribute

The code in this repository is developed and maintained by the project
maintainer (largely with the help of AI tooling), so **pull requests with code
changes are not expected or required**. You do not need to set up a dev
environment or run the test/lint suite to help out.

The most useful way to contribute is by **opening issues**:

- **Bug reports** — steps to reproduce, expected vs. actual behavior, and your
  environment (OS, Rust version, crate versions).
- **Feature requests / proposals** — what you want to do and the use case.
- **Questions** — anything unclear in the docs or crate APIs.

Please open issues at
<https://github.com/tpt-solutions/tpt-local-ai/issues>.

If you do want to share a concrete fix or idea, that's still welcome — open an
issue first to discuss it so we can avoid duplicate or conflicting work.

## Local development (for the curious)

If you'd like to build or experiment with the workspace yourself, you only need
a stable Rust toolchain (the MSRV is **1.80.0**):

```sh
rustup toolchain install stable
rustup component add rustfmt clippy
git clone https://github.com/tpt-solutions/tpt-local-ai
cd tpt-local-ai
```

The checks CI runs, in case you want to run them locally:

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features
```

On Windows PowerShell, set `RUSTDOCFLAGS` with
`$env:RUSTDOCFLAGS="-D warnings"` before the `cargo doc` line.

## License

By contributing (including by opening issues with code snippets or proposals),
you agree that your contributions will be dual licensed under MIT and
Apache-2.0, matching the rest of the project.

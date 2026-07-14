# tpt-local-ai — Release Checklist

Workspace of 5 Rust crates providing local-AI "plumbing" (HF Hub downloads, Jinja chat templates,
safetensors I/O, tokenization, LoRA merging). Optimized for a clean crates.io release.

Publish order: `tpt-hf-hub` → `tpt-jinja-chat` → `tpt-tokenizer-core` → `tpt-safetensors-io` → `tpt-lora-merge`
(last, since it depends on `tpt-safetensors-io`).

## 0. Workspace Bootstrap

- [ ] `git init`, add `.gitignore` (target/, Cargo.lock)
- [ ] Root `Cargo.toml` with `[workspace]` members + shared `workspace.package` fields (edition, license, repository, rust-version)
- [ ] `crates/` directory with one subfolder per crate
- [ ] Root `README.md` summarizing the 5-crate suite, linking to each
- [ ] `LICENSE-MIT` and `LICENSE-APACHE` at workspace root (dual license: MIT OR Apache-2.0)
- [ ] `.github/workflows/ci.yml` — matrix on stable + MSRV, steps: `cargo fmt --check`, `cargo clippy --all-targets --all-features -- -D warnings`, `cargo test --all`, `cargo doc --no-deps`
- [ ] Confirm `tpt-*` crate names are available on crates.io before first publish

## 1. tpt-hf-hub — async HF model downloader & cache manager

- [x] `Cargo.toml`: reqwest (rustls-tls, no native-tls), tokio, sha2, dirs
- [x] XDG cache path resolution: `~/.cache/tpt/hub`
- [x] `HubClient` with `download_file` and `snapshot_download`
- [x] Resumable downloads via HTTP Range requests
- [x] SHA256 verification of downloaded files
- [x] Atomic writes: download to `*.tmp`, rename on success
- [x] `ProgressReporter` trait for user-pluggable progress UI (no bundled TUI dep)
- [x] Unit tests + integration tests using a mocked HTTP server (e.g. `wiremock`) — no real network calls in tests
- [x] `examples/download_model.rs`
- [x] Doc comments on all public items + `#![warn(missing_docs)]`
- [x] Crate `README.md`
- [x] `Cargo.toml` metadata: description, keywords, categories, license, repository, readme

## 2. tpt-jinja-chat — pure-Rust Jinja subset parser for chat templates

- [x] Zero external dependencies (hand-rolled lexer + recursive-descent parser, manual error type — no `thiserror`)
- [x] Support: `{{ variable }}` substitution, `{% for %}/{% endfor %}`, `{% if %}/{% elif %}/{% else %}/{% endif %}`
- [x] `ChatTemplate::parse(&str) -> Result<ChatTemplate, TemplateError>`
- [x] `ChatTemplate::render(&self, context: &Context) -> Result<String, TemplateError>`
- [x] Unit tests covering real-world templates (Llama 3, Mistral `tokenizer_config.json` chat_template strings)
- [x] `examples/render_llama3_template.rs`
- [x] Doc comments + `#![warn(missing_docs)]`
- [x] Crate `README.md`
- [x] `Cargo.toml` metadata

## 3. tpt-safetensors-io — memory-mapped safetensors reader/writer

- [x] `Cargo.toml`: memmap2, serde, serde_json
- [x] `SafetensorsFile::open(path)` — mmap-backed, zero-copy
- [x] `tensor_names()` / `get_tensor(name) -> TensorView { dtype, shape, bytes }`
- [x] `SafetensorsBuilder` (builder pattern) for writing new files with correct 8-byte header alignment
- [x] Unit tests with small generated fixture `.safetensors` files (round-trip write→read)
- [x] `examples/inspect_safetensors.rs`
- [x] Doc comments + `#![warn(missing_docs)]`
- [x] Crate `README.md`
- [x] `Cargo.toml` metadata

## 4. tpt-tokenizer-core — pure-Rust BPE + WordPiece tokenizer

- [x] 100% pure Rust, no C++ bindings
- [x] `no_std` + `alloc` compatible; std-only conveniences (file loading) behind default-on `std` feature
- [x] HashMap-based vocab lookup
- [x] Shared `Tokenizer` trait with `encode`/`decode`
- [x] `BpeTokenizer::from_vocab_merges(...)`
- [x] `WordPieceTokenizer::from_vocab(...)`
- [x] Unit tests against known vocab/merge fixtures with expected token IDs
- [x] `examples/tokenize_text.rs`
- [x] Doc comments + `#![warn(missing_docs)]`
- [x] Crate `README.md`
- [x] `Cargo.toml` metadata

## 5. tpt-lora-merge — CPU-based LoRA weight merging

- [x] `Cargo.toml`: path dependency on `tpt-safetensors-io`, `ndarray`, `clap` (for CLI)
- [x] `merge_lora(base, lora, scale) -> MergedWeights` library function (B @ A delta scaled by alpha/r, added to base)
- [x] `[[bin]]` CLI: `--base`, `--lora`, `--output`, `--scale` args via clap
- [x] Unit tests validating merge math against hand-computed small matrices
- [x] Integration test: full CLI run producing a merged safetensors file
- [x] `examples/merge_lora_example.rs`
- [x] Doc comments + `#![warn(missing_docs)]`
- [x] Crate `README.md`
- [x] `Cargo.toml` metadata

## 6. Release Readiness (cross-cutting, all crates)

- [x] Every crate: `license`, `description`, `repository`, `keywords` (≤5), `categories`, `edition = "2021"`, `rust-version`, `readme` set in `Cargo.toml`
- [x] `cargo doc --workspace --no-deps` builds cleanly (docs.rs-ready)
- [x] `cargo clippy --workspace --all-targets --all-features -- -D warnings` clean
- [x] `cargo fmt --check` clean
- [x] `cargo test --workspace` passes
- [x] Every crate has at least one runnable example, manually exercised (not just compiled)

## 7. Publish

- [ ] `cargo publish --dry-run -p tpt-hf-hub`
- [ ] `cargo publish --dry-run -p tpt-jinja-chat`
- [ ] `cargo publish --dry-run -p tpt-tokenizer-core`
- [ ] `cargo publish --dry-run -p tpt-safetensors-io`
- [ ] `cargo publish --dry-run -p tpt-lora-merge`
- [ ] `cargo publish -p tpt-hf-hub`
- [ ] `cargo publish -p tpt-jinja-chat`
- [ ] `cargo publish -p tpt-tokenizer-core`
- [ ] `cargo publish -p tpt-safetensors-io`
- [ ] `cargo publish -p tpt-lora-merge`

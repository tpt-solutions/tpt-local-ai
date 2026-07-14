# tpt-local-ai — Release Checklist

Workspace of 5 Rust crates providing local-AI "plumbing" (HF Hub downloads, Jinja chat templates,
safetensors I/O, tokenization, LoRA merging). Optimized for a clean crates.io release.

Publish order: `tpt-hf-hub` → `tpt-jinja-chat` → `tpt-tokenizer-core` → `tpt-safetensors-io` → `tpt-lora-merge`
(last, since it depends on `tpt-safetensors-io`).

## 0. Workspace Bootstrap

- [x] `git init`, add `.gitignore` (target/, Cargo.lock)
- [x] Root `Cargo.toml` with `[workspace]` members + shared `workspace.package` fields (edition, license, repository, rust-version)
- [x] `crates/` directory with one subfolder per crate
- [x] Root `README.md` summarizing the 5-crate suite, linking to each
- [x] `LICENSE-MIT` and `LICENSE-APACHE` at workspace root (dual license: MIT OR Apache-2.0)
- [x] `.github/workflows/ci.yml` — matrix on stable + MSRV, steps: `cargo fmt --check`, `cargo clippy --all-targets --all-features -- -D warnings`, `cargo test --all`, `cargo doc --no-deps`
- [x] Confirm `tpt-*` crate names are available on crates.io before first publish (all 5 return 404 / available as of check)

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

### 1b. tpt-hf-hub — bugs & gaps found in review

- [x] Fix path traversal: sanitize `filename`/server-provided `rfilename` in `snapshot_download` (`client.rs:98`) to reject `..` segments and absolute paths before joining into `dest_dir`
- [x] Verify resumed downloads actually get `206 Partial Content` back before appending (`client.rs:122-159`); treat a `200` response to a `Range` request as "restart from scratch", not "append"
- [x] Add file locking (or an in-process mutex keyed by the tmp path) around the deterministic `.tmp` path (`client.rs:254-258`) to prevent concurrent-download corruption
- [x] Fix `rename`-over-existing-file TOCTOU (`client.rs:191`) so behavior is consistent on Windows vs POSIX
- [x] Harden `validate_repo_id` (`client.rs:247-252`) to reject `..` segments and Windows absolute paths (`C:\...`)
- [x] Fix `cache.rs:44-49` test flakiness: don't mutate the process-global `TPT_HUB_CACHE` env var without synchronization across parallel tests
- [x] Add auth-token support (`Authorization: Bearer <token>`, `HF_TOKEN` env var, `.with_token(...)`) for gated/private models
- [x] Add offline mode (`HF_HUB_OFFLINE`-style: cache-only, never hit network)
- [x] Support `HF_ENDPOINT` env var convention for mirrors
- [x] Add retry/backoff on transient network failures
- [x] Parallelize sibling downloads in `snapshot_download` (currently strictly sequential) with a concurrency limit knob
- [x] Add regression tests: Range-ignored-by-server corruption, path traversal via `rfilename`, concurrent download of the same file
- [x] Add a Windows (and ideally macOS) job to `.github/workflows/ci.yml` — currently `ubuntu-latest` only, despite Windows-sensitive rename/path logic

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

### 2b. tpt-jinja-chat — bugs & gaps found in review

- [x] Fix UTF-8 mangling: template text and quoted-string literals are built with `bytes[i] as char` (`parser.rs:90`, `parser.rs:437`), which mojibake-corrupts any multi-byte UTF-8 (accents, CJK, emoji); rewrite to be UTF-8-aware like `value.rs`'s JSON string parser
- [x] Fix potential panic in `find_close` (`parser.rs:102-113`): `str` slicing (`src[i..i+close.len()]`) while byte-scanning for `}}`/`%}`/`#}` can panic with "byte index is not a char boundary" on non-ASCII template source
- [x] Add filter (`|`) support: `tojson`, `trim`, `default`, `join`, `upper`/`lower`, `length`, `first`/`last`, `selectattr`, `map`, `reject`, `string` — `tojson` alone is used pervasively in real Llama 3.1/3.2 tool-calling templates and blocks most real-world adoption without it
- [x] Add `is` test expressions (`is defined`, `is none`, `is string`, `is iterable`, etc.)
- [x] Add `~` string-concatenation operator (only `+` supported today)
- [x] Add list literal support (`[]`, `['user', 'assistant']`) — currently unparseable
- [x] Add function-call syntax (`Expr::Call`) to unblock `raise_exception(...)`, `.items()`, `namespace()` idioms used by real Hub templates (Mistral/Zephyr role-alternation checks, Qwen-style `namespace()` state)
- [x] Add tuple/multi-variable `for` target (`{% for k, v in x.items() %}`)
- [x] Add non-ASCII template regression test (would have caught the mojibake bug)
- [x] Add a `cargo-fuzz` target for the scanner/parser (handles untrusted template input by design, zero fuzz coverage today)

## 3. tpt-safetensors-io — memory-mapped safetensors reader/writer

- [x] `Cargo.toml`: memmap2, serde, serde_json
- [x] `SafetensorsFile::open(path)` — mmap-backed, zero-copy
- [x] `tensor_names()` / `get_tensor(name) -> TensorView { dtype, shape, bytes }`
- [x] `SafetensorsBuilder` (builder pattern) for writing new files with correct 8-byte header alignment
- [x] Atomic writes: `write_to_file` writes to a temp file and renames over the target (never leaves a truncated file on interrupt)
- [x] Header validation: tensor `data_offsets` (start ≤ end, end within file bounds) checked against the mmap length before any `get_tensor` slicing, to reject a malformed/attacker-controlled header instead of panicking or OOB-slicing
- [x] Unit tests with small generated fixture `.safetensors` files (round-trip write→read)
- [x] `examples/inspect_safetensors.rs`
- [x] Doc comments + `#![warn(missing_docs)]`
- [x] Crate `README.md`
- [x] `Cargo.toml` metadata

### 3b. tpt-safetensors-io — bugs & gaps found in review

- [x] Fix integer overflow in header-length arithmetic (`reader.rs:194-199`, `reader.rs:274`): `8 + header_len` / `8 + header_len + info.end` can overflow `usize` on a corrupted/adversarial header, silently wrapping in release builds and defeating the bounds check
- [x] Add a cross-check that each tensor's `data_offsets` span equals `dtype.size_bytes() * numel()` before returning a `TensorView`, so callers reading `.data` directly are protected against a truncated/oversized declared region
- [x] Add a `# Safety` doc note on the `Mmap::map` call (`reader.rs:134`) — security-relevant entry point exposed transitively to `tpt-lora-merge`
- [x] Add `F8_E4M3`/`F8_E5M2` (fp8) dtype support (increasingly common in quantized checkpoints)
- [x] Add `U16`/`U32`/`U64` dtype support (part of the safetensors spec)
- [x] Add a streaming/chunked write path to `SafetensorsBuilder` — currently buffers the entire output file in a `Vec<u8>`, undermining the crate's "zero-copy" pitch on the write side
- [x] Consider an API to patch/replace a subset of tensors in an existing file without rebuilding from scratch
- [x] Add regression tests for corrupted/malicious headers (overflow, `start > end`, missing fields, non-object header)
- [x] Add round-trip tests for F16, BF16, I8/I16/I32/I64, U8, BOOL (currently only F32 is tested)
- [x] Add a `cargo-fuzz` target for header parsing (explicitly handles adversarial input per its own doc comments, zero fuzz coverage today)
- [x] Set `overflow-checks = true` in the release profile, or otherwise guard against silent wraparound in offset arithmetic workspace-wide

## 4. tpt-tokenizer-core — pure-Rust BPE + WordPiece tokenizer

- [x] 100% pure Rust, no C++ bindings
- [x] `no_std` + `alloc` compatible; std-only conveniences (file loading) behind default-on `std` feature
- [x] HashMap-based vocab lookup
- [x] Shared `Tokenizer` trait with `encode`/`decode`
- [x] `BpeTokenizer::from_vocab_merges(...)`
- [x] `WordPieceTokenizer::from_vocab(...)`
- [x] `decode` reconstructs inter-word spacing (WordPiece: space before each non-continuation token; BPE: documented as lossy since it has no `Ġ`-style marker)
- [x] Shared `split_words`/`parse_vocab_lines` helpers so BPE and WordPiece can't silently drift in whitespace/vocab-file handling
- [x] Unit tests against known vocab/merge fixtures with expected token IDs
- [x] `examples/tokenize_text.rs`
- [x] Doc comments + `#![warn(missing_docs)]`
- [x] Crate `README.md`
- [x] `Cargo.toml` metadata

### 4b. tpt-tokenizer-core — bugs & gaps found in review

- [ ] Address O(n²) per-word merge loop in `BpeTokenizer::tokenize_word` (`bpe.rs:63-89`) — perf cliff for pathologically long "words" (e.g. whitespace-free blobs, since `split_words` only splits on whitespace). NOTE: byte-level mode now pre-splits via `gpt2_split`, which breaks up long blobs and largely defuses the cliff; the greedy merge loop itself is still O(n²) and could later use a rank heap.
- [x] Add byte-level pre-tokenization / byte-fallback so `encode` never fails on arbitrary Unicode (currently returns `UnknownToken` for any char/sub-word missing from vocab with no `<unk>`) — `BpeTokenizer::with_byte_level()` + reversible byte↔unicode table in `pretokenize.rs`
- [x] Add GPT-2-style pre-tokenizer regex splitting (contractions, punctuation, digits) instead of bare whitespace splitting — regex-free `gpt2_split` in `pretokenize.rs`
- [x] Add BERT "basic tokenize" pass for WordPiece (lowercasing, accent stripping, CJK character spacing, punctuation splitting) — `bert_basic` in `pretokenize.rs` + `WordPieceTokenizer::with_lowercase()`. NOTE: accent stripping (NFD) intentionally omitted to keep the crate Unicode-table-free / dependency-free.
- [ ] Add a loader for the modern unified `tokenizer.json` format (most current Hub repos ship this, not legacy `vocab.txt`+`merges.txt`) — likely the single highest-leverage adoption fix for this crate. DEFERRED: needs a JSON parser, which conflicts with the crate's zero-dependency goal (revisit with a tiny internal parser or optional feature).
- [ ] Add special-token handling (BOS/EOS/CLS/SEP auto-insertion, `added_tokens`, padding/truncation, batch encoding API). PARTIAL: `BpeTokenizer::with_special_tokens()` matches registered tokens atomically (longest-match) and decodes them verbatim; BOS/EOS auto-insertion, padding/truncation and batch API still TODO.
- [ ] Add Unicode normalization (NFC/NFKC) before tokenization
- [x] Add tests for `from_files`/`from_file` disk-loading constructors, malformed vocab/merges files, empty input, `max_input_chars_per_word` overflow behavior, non-ASCII/multi-byte input

## 5. tpt-lora-merge — CPU-based LoRA weight merging

- [x] `Cargo.toml`: path dependency on `tpt-safetensors-io`, `ndarray`, `clap` (for CLI)
- [x] `merge_lora(base, lora, scale) -> MergedWeights` library function (B @ A delta scaled by alpha/r, added to base)
- [x] Tensors without a matching adapter (incl. non-2D weights) are copied through with their original dtype/bytes instead of being force-decoded to `f32`
- [x] `[[bin]]` CLI: `--base`, `--lora`, `--output`, `--scale` args via clap
- [x] CLI refuses to run if `--output` aliases `--base`/`--lora` (would truncate a source file)
- [x] Unit tests validating merge math against hand-computed small matrices
- [x] Integration test: full CLI run producing a merged safetensors file
- [x] `examples/merge_lora_example.rs`
- [x] Doc comments + `#![warn(missing_docs)]`
- [x] Crate `README.md`
- [x] `Cargo.toml` metadata

### 5b. tpt-lora-merge — bugs & gaps found in review

- [x] Preserve original `__metadata__` from base/LoRA safetensors files on merge (`merge.rs:59-67` currently only writes a `"merged_by"` key, discarding provenance/license/quantization metadata)
- [x] Error (not silently copy-through) when only one of `lora_A`/`lora_B` is present for a module (`merge.rs:107-113`) — likely indicates a corrupt/mis-named adapter file
- [x] Verify every tensor in the LoRA file was actually consumed; error/warn on unused adapter tensors instead of silently no-op'ing the whole merge (exit code 0) on a naming-convention mismatch
- [x] Support multiple LoRA adapters in one merge (weighted sum), not just a single `--scale`
- [x] Read `adapter_config.json` to auto-derive `alpha`/`r` instead of requiring manual `--scale`
- [x] Support alternate LoRA naming conventions (Kohya `lora_down`/`lora_up` + per-layer `.alpha` tensors, PEFT multi-adapter `.lora_A.<adapter_name>.weight`)
- [x] Preserve base dtype instead of always upcasting merged tensors to F32 (currently silently doubles output size for BF16/F16 bases); at minimum warn the CLI user about the size growth
- [x] Add a `--dry-run`/`--check` mode to validate adapter/base tensor alignment before committing to a merge+write
- [x] Add tests: shape-mismatch error path, partial-adapter-pair case, metadata preservation, multiple adapted tensors, non-2D tensor alongside a same-named adapter

## 6. Release Readiness (cross-cutting, all crates)

- [x] Every crate: `license`, `description`, `repository`, `keywords` (≤5), `categories`, `edition = "2021"`, `rust-version`, `readme` set in `Cargo.toml`
- [x] `cargo doc --workspace --no-deps` builds cleanly (docs.rs-ready)
- [x] `cargo clippy --workspace --all-targets --all-features -- -D warnings` clean
- [x] `cargo fmt --check` clean
- [x] `cargo test --workspace` passes
- [x] Every crate has at least one runnable example, manually exercised (not just compiled)

### 6b. Cross-cutting bugs & hygiene found in review

- [ ] Reconcile currently-uncommitted working-tree changes before any release push
- [x] Verify `Cargo.toml` `categories` values (e.g. `"template-engine"`, `"no-std"`) against crates.io's fixed category taxonomy before publish — all category slugs across the 5 crates return `200` from the crates.io categories API
- [x] Normalize `#[non_exhaustive]` usage across all 5 error enums — `HubError` already had it; added it to `TemplateError`, so all 5 (`HubError`/`TemplateError`/`SafetensorsError`/`MergeError`/`TokenizerError`) are now `#[non_exhaustive]`

## 7. Innovative / high-value additions

- [x] Cross-crate "cookbook" example chaining all 5 crates end-to-end: download a model + LoRA from the Hub → merge → load tokenizer → render a chat template → tokenize the result — implemented as the `crates/cookbook` binary (`cargo run -p tpt-cookbook`), runnable offline by default with an opt-in real Hub download
- [x] `cargo-fuzz` targets for `tpt-safetensors-io` header parsing and `tpt-jinja-chat` scanning (tracked per-crate above too)
- [ ] WASM demo for `tpt-jinja-chat` + `tpt-tokenizer-core` (both pure-Rust; tokenizer-core already `no_std`-compatible) — browser playground doubles as a "zero dependency" proof point
- [ ] GGUF metadata reading (in `tpt-safetensors-io` or a sibling crate) — GGUF is the dominant local-inference (llama.cpp) format; safetensors-only limits the "local-AI plumbing" pitch to the HF/PyTorch half
- [ ] `tokenizer.json` loader for `tpt-tokenizer-core` (tracked above too) — likely 10x's real-world usability

## 8. Usability / automation improvements

- [x] Add `cargo-deny` and/or `cargo-audit` to CI — supply-chain/vuln scanning, expected trust signal for crates parsing untrusted input (added `deny.toml` + a `cargo deny` CI job)
- [ ] Adopt `release-plz` or `cargo-release` + `cargo-workspaces` for coordinated multi-crate version bumps/changelog generation across the publish order above
- [ ] Add `cargo-semver-checks` to CI once crates move past 0.1
- [x] Add a Dependabot/Renovate config for external deps (reqwest, tokio, memmap2, ndarray, clap) — `.github/dependabot.yml` covers the `cargo` + `github-actions` ecosystems

## 9. Adoption / onboarding improvements

- [x] Make the cross-crate cookbook example (see §7) the root README's primary quickstart, replacing the current "defers entirely to per-crate READMEs" structure
- [x] Add crates.io/docs.rs/CI/license badges to root and per-crate READMEs
- [x] Add a README comparison section vs. closest alternatives (`hf-hub`, `minijinja`, `tokenizers`) explaining the actual differentiator (pure-Rust, zero/minimal-dep, `no_std`-friendly)
- [x] Add `CONTRIBUTING.md` with local dev setup + test/lint commands (currently only implicit in this checklist)

## 10. Publish

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

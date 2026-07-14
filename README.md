# tpt-local-ai

[![CI](https://github.com/tpt-ai/tpt-local-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/tpt-ai/tpt-local-ai/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

The missing "plumbing" for Rust developers building **local, privacy-first AI**
applications. Python has `huggingface_hub`; Rust needs the TPT equivalent.

`tpt-local-ai` is a Cargo workspace of five small, dependency-conscious crates
that cover the unglamorous but essential steps of running models locally:

| Crate | Purpose | Highlights |
|-------|---------|-----------|
| [`tpt-hf-hub`](crates/tpt-hf-hub) [![crates.io](https://img.shields.io/crates/v/tpt-hf-hub.svg)](https://crates.io/crates/tpt-hf-hub) [![docs.rs](https://img.shields.io/docsrs/tpt-hf-hub)](https://docs.rs/tpt-hf-hub) | Async Hugging Face model downloader & cache manager | Resumable Range downloads, SHA256 verification, atomic writes, pluggable progress UI |
| [`tpt-jinja-chat`](crates/tpt-jinja-chat) [![crates.io](https://img.shields.io/crates/v/tpt-jinja-chat.svg)](https://crates.io/crates/tpt-jinja-chat) [![docs.rs](https://img.shields.io/docsrs/tpt-jinja-chat)](https://docs.rs/tpt-jinja-chat) | Pure-Rust Jinja2 subset for chat templates | Zero dependencies, hand-rolled parser, `for`/`if`/`set`/filters/expressions |
| [`tpt-safetensors-io`](crates/tpt-safetensors-io) [![crates.io](https://img.shields.io/crates/v/tpt-safetensors-io.svg)](https://crates.io/crates/tpt-safetensors-io) [![docs.rs](https://img.shields.io/docsrs/tpt-safetensors-io)](https://docs.rs/tpt-safetensors-io) | Memory-mapped safetensors reader/writer | Zero-copy `memmap2`, streaming builder with correct header alignment |
| [`tpt-tokenizer-core`](crates/tpt-tokenizer-core) [![crates.io](https://img.shields.io/crates/v/tpt-tokenizer-core.svg)](https://crates.io/crates/tpt-tokenizer-core) [![docs.rs](https://img.shields.io/docsrs/tpt-tokenizer-core)](https://docs.rs/tpt-tokenizer-core) | Pure-Rust BPE + WordPiece tokenizer | `no_std` + `alloc` friendly, byte-level BPE, no C++ bindings |
| [`tpt-lora-merge`](crates/tpt-lora-merge) [![crates.io](https://img.shields.io/crates/v/tpt-lora-merge.svg)](https://crates.io/crates/tpt-lora-merge) [![docs.rs](https://img.shields.io/docsrs/tpt-lora-merge)](https://docs.rs/tpt-lora-merge) | CPU-based LoRA weight merging | `ndarray` math, multi-adapter weighted sum, ships a CLI |

## Quickstart: the whole pipeline

The [`cookbook`](crates/cookbook) example chains all five crates end-to-end —
resolve a Hub cache (and optionally download real files), write a base
checkpoint plus a LoRA adapter, merge them, render a chat template, and tokenize
the result:

```sh
# Runs fully offline by default:
cargo run -p tpt-cookbook

# Exercise a real Hub download too:
TPT_COOKBOOK_HUB_REPO=gpt2 TPT_COOKBOOK_HUB_FILE=config.json \
    cargo run -p tpt-cookbook
```

It prints each stage of the pipeline:

```text
[1/5] tpt-hf-hub            -> cache dir + optional download
[2/5] tpt-safetensors-io    -> write base + adapter checkpoints
[3/5] tpt-lora-merge        -> base + scale * (B @ A)
[4/5] tpt-jinja-chat        -> render chat template into a prompt
[5/5] tpt-tokenizer-core    -> tokenize (and round-trip) the prompt
```

See [`crates/cookbook/src/main.rs`](crates/cookbook/src/main.rs) for the fully
commented source. Each crate also ships its own focused `examples/`.

## How it compares

These crates deliberately trade breadth for a small, auditable, pure-/minimal-Rust
footprint. If you need the full feature set of the mainstream libraries, use them;
if you want lean plumbing you can read end-to-end, use these.

| Need | Mainstream option | `tpt-local-ai` differentiator |
|------|-------------------|-------------------------------|
| Hub downloads | [`hf-hub`](https://crates.io/crates/hf-hub) | rustls-only (no OpenSSL), pluggable progress trait, explicit offline mode, path-traversal hardening, retry/backoff, concurrency knob |
| Chat templates | [`minijinja`](https://crates.io/crates/minijinja) | **zero dependencies**, hand-rolled parser scoped to the Jinja subset real chat templates use (`tojson`, `raise_exception`, `namespace()`, ...), with a fuzz target |
| Tokenizers | [`tokenizers`](https://crates.io/crates/tokenizers) (HF) | 100% pure Rust, **`no_std` + `alloc`**, no C++/ONNX build step, byte-level BPE + WordPiece |
| safetensors I/O | [`safetensors`](https://crates.io/crates/safetensors) | mmap-backed zero-copy reads, streaming builder, adversarial-header hardening + fuzz target, `overflow-checks` on offset math |
| LoRA merging | (mostly Python / PEFT) | CPU-only Rust library **and** CLI, multi-adapter weighted sum, `adapter_config.json` auto-scale, dtype preservation, dry-run |

## MSRV

Rust **1.80.0**. Enforced in CI.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for local dev setup and the test/lint
commands. Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as below, without any additional terms or
conditions.

## License

Licensed under either of

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
* MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

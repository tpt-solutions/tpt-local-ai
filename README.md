# tpt-local-ai

The missing "plumbing" for Rust developers building **local, privacy-first AI**
applications. Python has `huggingface_hub`; Rust needs the TPT equivalent.

`tpt-local-ai` is a Cargo workspace of five small, dependency-conscious crates
that cover the unglamorous but essential steps of running models locally:

| Crate | Purpose | Highlights |
|-------|---------|-----------|
| [`tpt-hf-hub`](crates/tpt-hf-hub) | Async Hugging Face model downloader & cache manager | Resumable Range downloads, SHA256 verification, atomic writes, pluggable progress UI |
| [`tpt-jinja-chat`](crates/tpt-jinja-chat) | Pure-Rust Jinja2 subset for chat templates | Zero dependencies, hand-rolled parser, `for`/`if`/`set`/expressions |
| [`tpt-safetensors-io`](crates/tpt-safetensors-io) | Memory-mapped safetensors reader/writer | Zero-copy `memmap2`, builder with correct header alignment |
| [`tpt-tokenizer-core`](crates/tpt-tokenizer-core) | Pure-Rust BPE + WordPiece tokenizer | `no_std` + `alloc` friendly, no C++ bindings |
| [`tpt-lora-merge`](crates/tpt-lora-merge) | CPU-based LoRA weight merging | `ndarray` math, ships a CLI |

## License

Licensed under either of

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
* MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

## Contributing

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

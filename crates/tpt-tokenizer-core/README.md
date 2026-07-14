# tpt-tokenizer-core

[![crates.io](https://img.shields.io/crates/v/tpt-tokenizer-core.svg)](https://crates.io/crates/tpt-tokenizer-core)
[![docs.rs](https://img.shields.io/docsrs/tpt-tokenizer-core)](https://docs.rs/tpt-tokenizer-core)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

A small, **dependency-free** implementation of the two tokenization schemes used
by most open-weight LLMs, written in pure Rust:

- [`BpeTokenizer`] — Byte-Pair Encoding (GPT-2 / Llama style).
- [`WordPieceTokenizer`] — WordPiece (BERT style).

Both implement the shared `Tokenizer` trait (`encode` / `decode`).

## Features

- **`tokenizer.json` loading** — parse the modern unified Hugging Face format
  (`from_tokenizer_json_str` / `from_tokenizer_json_file`) with a built-in JSON
  parser, no `serde`. Auto-detects byte-level BPE, lowercasing, and special
  `added_tokens`.
- **Batch encoding** — BOS/EOS insertion, padding, and truncation via
  `EncodeConfig` + `TokenizerExt::encode_with` / `encode_batch`, returning an
  `Encoding { ids, attention_mask }`.
- **Byte-level BPE** — `BpeTokenizer::with_byte_level()` guarantees encoding
  never fails on arbitrary UTF-8 and `decode(encode(s)) == s`.
- **Special tokens** — register atomic markers (`<|endoftext|>`, `<s>`) that are
  never split.
- **Unicode normalization** (opt-in) — NFC/NFD/NFKC/NFKD via the `normalization`
  feature (see below).

## `no_std` + `alloc`

The tokenization logic is `#![no_std]` compatible: it only depends on `alloc`
and never touches the standard library. The `std` feature (enabled by default)
adds convenience constructors (`BpeTokenizer::from_files`,
`WordPieceTokenizer::from_file`, `from_tokenizer_json_file`) that load from disk.

```toml
# Fully `no_std` (you supply the vocab/merges at runtime):
tpt-tokenizer-core = { version = "0.1.0", default-features = false }
```

### Optional Unicode normalization

Correct NFC/NFKC needs the Unicode Character Database, so it is gated behind the
opt-in `normalization` feature (which pulls in `unicode-normalization`) to keep
the default build dependency-free:

```toml
tpt-tokenizer-core = { version = "0.1.0", features = ["normalization"] }
```

```rust
use tpt_tokenizer_core::{BpeTokenizer, NormalizationForm};

let tok = BpeTokenizer::from_vocab_merges(vocab, merges)
    .with_normalization(NormalizationForm::Nfc);
```

## Usage

```rust
use std::collections::BTreeMap;
use tpt_tokenizer_core::{BpeTokenizer, Tokenizer};

let mut vocab = BTreeMap::new();
vocab.insert("l".to_string(), 0u32);
vocab.insert("o".to_string(), 1u32);
vocab.insert("w".to_string(), 2u32);
vocab.insert("lo".to_string(), 3u32);
vocab.insert("low".to_string(), 4u32);
let merges = vec![
    ("l".to_string(), "o".to_string()),
    ("lo".to_string(), "w".to_string()),
];
let tok = BpeTokenizer::from_vocab_merges(vocab, merges);
assert_eq!(tok.encode("low").unwrap(), vec![4]);
```

## License

Licensed under either of MIT or Apache-2.0 at your option.

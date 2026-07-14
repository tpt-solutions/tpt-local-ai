# tpt-tokenizer-core

A small, **dependency-free** implementation of the two tokenization schemes used
by most open-weight LLMs, written in pure Rust:

- [`BpeTokenizer`] — Byte-Pair Encoding (GPT-2 / Llama style).
- [`WordPieceTokenizer`] — WordPiece (BERT style).

Both implement the shared `Tokenizer` trait (`encode` / `decode`).

## `no_std` + `alloc`

The tokenization logic is `#![no_std]` compatible: it only depends on `alloc`
and never touches the standard library. The `std` feature (enabled by default)
adds convenience constructors (`BpeTokenizer::from_files`,
`WordPieceTokenizer::from_file`) that load vocabularies from disk.

```toml
# Fully `no_std` (you supply the vocab/merges at runtime):
tpt-tokenizer-core = { version = "0.1.0", default-features = false }
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

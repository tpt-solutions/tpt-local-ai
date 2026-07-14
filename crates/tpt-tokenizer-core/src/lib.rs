//! # tpt-tokenizer-core
//!
//! A small, **dependency-free** implementation of the two tokenization schemes
//! used by most open-weight LLMs:
//!
//! * [`BpeTokenizer`] — Byte-Pair Encoding (GPT-2 / Llama style).
//! * [`WordPieceTokenizer`] — WordPiece (BERT style).
//!
//! Both implement the shared [`Tokenizer`] trait with `encode` / `decode`.
//!
//! The crate is `#![no_std]` compatible: all tokenization logic lives behind
//! `alloc` and never touches the standard library. The `std` feature (enabled
//! by default) only adds convenience constructors that load vocabularies and
//! merge tables from disk.
//!
//! ## Example
//!
//! ```
//! use tpt_tokenizer_core::{BpeTokenizer, Tokenizer};
//!
//! let mut vocab = std::collections::BTreeMap::new();
//! vocab.insert("l".to_string(), 0u32);
//! vocab.insert("o".to_string(), 1u32);
//! vocab.insert("w".to_string(), 2u32);
//! vocab.insert("lo".to_string(), 3u32);
//! vocab.insert("low".to_string(), 4u32);
//! let merges = vec![
//!     ("l".to_string(), "o".to_string()),
//!     ("lo".to_string(), "w".to_string()),
//! ];
//! let tok = BpeTokenizer::from_vocab_merges(vocab, merges);
//! let ids = tok.encode("low").unwrap();
//! assert_eq!(ids, vec![4]);
//! ```
#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]

extern crate alloc;

pub mod error;
pub mod tokenizer;

mod bpe;
mod wordpiece;

pub use bpe::BpeTokenizer;
pub use error::TokenizerError;
pub use tokenizer::{TokenId, Tokenizer};
pub use wordpiece::WordPieceTokenizer;

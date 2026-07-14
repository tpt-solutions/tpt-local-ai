//! Shared tokenizer trait and token types.

use alloc::string::String;
use alloc::vec::Vec;

use crate::error::TokenizerError;

/// A single token id — the integer a model actually consumes.
pub type TokenId = u32;

/// A tokenizer that maps text to a sequence of [`TokenId`]s and back.
///
/// Both [`BpeTokenizer`](crate::BpeTokenizer) and
/// [`WordPieceTokenizer`](crate::WordPieceTokenizer) implement this trait.
pub trait Tokenizer {
    /// Encode `text` into a sequence of token ids.
    ///
    /// # Errors
    /// Returns a [`TokenizerError`] if a sub-word cannot be represented by the
    /// vocabulary and no `<unk>` / `[UNK]` fallback is configured.
    fn encode(&self, text: &str) -> Result<Vec<TokenId>, TokenizerError>;

    /// Decode `ids` back into a human-readable string.
    ///
    /// # Errors
    /// Returns a [`TokenizerError`] if an id is missing from the inverse
    /// vocabulary.
    fn decode(&self, ids: &[TokenId]) -> Result<String, TokenizerError>;
}

/// Splits `text` into whitespace-delimited words, skipping empties. Shared by
/// both tokenizer `encode` implementations so tokenization policy cannot
/// silently drift between schemes.
pub(crate) fn split_words(text: &str) -> impl Iterator<Item = &str> {
    text.split_whitespace().filter(|w| !w.is_empty())
}

/// Parses a one-token-per-line vocabulary file (line index = token id),
/// skipping blank lines. Shared by the `std` file-loading constructors.
#[cfg(feature = "std")]
pub(crate) fn parse_vocab_lines(text: &str) -> alloc::collections::BTreeMap<String, TokenId> {
    let mut vocab = alloc::collections::BTreeMap::new();
    for (i, line) in text.lines().enumerate() {
        let token = line.trim_end();
        if token.is_empty() {
            continue;
        }
        vocab.insert(token.to_string(), i as TokenId);
    }
    vocab
}

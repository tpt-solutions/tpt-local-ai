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

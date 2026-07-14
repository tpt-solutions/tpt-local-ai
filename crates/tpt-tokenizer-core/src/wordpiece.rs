//! WordPiece tokenizer (BERT style).

use alloc::collections::BTreeMap;
use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

use crate::error::TokenizerError;
use crate::tokenizer::{TokenId, Tokenizer};

/// Prefix attached to continuation sub-words in a WordPiece vocabulary.
const CONTINUATION: &str = "##";

/// A WordPiece tokenizer built from a vocabulary.
///
/// Words are split greedily into the longest matching sub-words; a word that
/// cannot be split at all is replaced by the `[UNK]` token.
#[derive(Debug, Clone)]
pub struct WordPieceTokenizer {
    vocab: BTreeMap<String, TokenId>,
    id_to_token: BTreeMap<TokenId, String>,
    unk_id: TokenId,
    max_input_chars_per_word: usize,
}

impl WordPieceTokenizer {
    /// Builds a tokenizer from a vocabulary (token string → id). `unk_token`
    /// names the fallback token (typically `"[UNK]"`); it must be present in
    /// `vocab`.
    ///
    /// # Errors
    /// Returns [`TokenizerError::UnknownToken`] if `unk_token` is missing.
    pub fn from_vocab(
        vocab: BTreeMap<String, TokenId>,
        unk_token: &str,
    ) -> Result<Self, TokenizerError> {
        let unk_id = *vocab
            .get(unk_token)
            .ok_or_else(|| TokenizerError::UnknownToken(unk_token.to_string()))?;
        let id_to_token = vocab
            .iter()
            .map(|(token, &id)| (id, token.clone()))
            .collect();
        Ok(Self {
            vocab,
            id_to_token,
            unk_id,
            max_input_chars_per_word: 100,
        })
    }

    /// Number of tokens in the vocabulary.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Tokenizes a single word into sub-word strings using the WordPiece
    /// longest-match algorithm.
    ///
    /// Returns `None` when the word cannot be split at all (it should be
    /// replaced by the `[UNK]` token by the caller).
    fn tokenize_word(&self, word: &str) -> Option<Vec<String>> {
        if self.vocab.contains_key(word) {
            return Some(vec![word.to_string()]);
        }
        let chars: Vec<char> = word.chars().collect();
        if chars.len() > self.max_input_chars_per_word {
            return None;
        }

        let mut sub_tokens = Vec::new();
        let mut start = 0usize;
        while start < chars.len() {
            let mut end = chars.len();
            let mut cur_substr: Option<String> = None;
            while start < end {
                let substr: String = chars[start..end].iter().collect();
                let candidate = if start > 0 {
                    format!("{CONTINUATION}{substr}")
                } else {
                    substr.clone()
                };
                if self.vocab.contains_key(&candidate) {
                    cur_substr = Some(candidate);
                    break;
                }
                end -= 1;
            }
            let sub = cur_substr?;
            sub_tokens.push(sub);
            start = end;
        }

        Some(sub_tokens)
    }

    /// Loads a BERT style vocabulary from disk: one token per line (line number
    /// = id).
    ///
    /// # Errors
    /// Returns [`TokenizerError::Io`] on a read failure or
    /// [`TokenizerError::UnknownToken`] if `unk_token` is absent.
    #[cfg(feature = "std")]
    pub fn from_file(vocab_path: &str, unk_token: &str) -> Result<Self, TokenizerError> {
        let text = std::fs::read_to_string(vocab_path)?;
        let mut vocab = BTreeMap::new();
        for (i, line) in text.lines().enumerate() {
            let token = line.trim_end().to_string();
            if token.is_empty() {
                continue;
            }
            vocab.insert(token, i as TokenId);
        }
        Self::from_vocab(vocab, unk_token)
    }
}

impl Tokenizer for WordPieceTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<TokenId>, TokenizerError> {
        let mut ids = Vec::new();
        for word in text.split_whitespace() {
            if word.is_empty() {
                continue;
            }
            match self.tokenize_word(word) {
                Some(subs) => {
                    for sub in subs {
                        ids.push(
                            *self
                                .vocab
                                .get(&sub)
                                .ok_or(TokenizerError::UnknownToken(sub))?,
                        );
                    }
                }
                None => ids.push(self.unk_id),
            }
        }
        Ok(ids)
    }

    fn decode(&self, ids: &[TokenId]) -> Result<String, TokenizerError> {
        let mut out = String::new();
        for &id in ids {
            let token = self
                .id_to_token
                .get(&id)
                .ok_or_else(|| TokenizerError::UnknownToken(id.to_string()))?;
            if let Some(stripped) = token.strip_prefix(CONTINUATION) {
                out.push_str(stripped);
            } else {
                out.push_str(token);
            }
        }
        Ok(out)
    }
}

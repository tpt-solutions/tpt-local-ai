//! WordPiece tokenizer (BERT style).

use alloc::collections::BTreeMap;
use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

use crate::error::TokenizerError;
use crate::pretokenize::bert_basic;
use crate::tokenizer::{TokenId, Tokenizer};

/// Prefix attached to continuation sub-words in a WordPiece vocabulary.
const CONTINUATION: &str = "##";

/// A WordPiece tokenizer built from a vocabulary.
///
/// Words are split greedily into the longest matching sub-words; a word that
/// cannot be split at all is replaced by the `[UNK]` token.
///
/// Input is pre-tokenized with a BERT-style basic tokenizer that splits on
/// whitespace and punctuation and isolates CJK characters. Enable
/// [`with_lowercase`](Self::with_lowercase) to match `*-uncased` models.
#[derive(Debug, Clone)]
pub struct WordPieceTokenizer {
    vocab: BTreeMap<String, TokenId>,
    id_to_token: BTreeMap<TokenId, String>,
    unk_id: TokenId,
    max_input_chars_per_word: usize,
    lowercase: bool,
    #[cfg(feature = "normalization")]
    normalize: Option<crate::normalize::NormalizationForm>,
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
            lowercase: false,
            #[cfg(feature = "normalization")]
            normalize: None,
        })
    }

    /// Enables lowercasing during pre-tokenization, matching `*-uncased`
    /// BERT models.
    #[must_use]
    pub fn with_lowercase(mut self) -> Self {
        self.lowercase = true;
        self
    }

    /// Applies the given Unicode normalization form to text before encoding.
    ///
    /// Requires the `normalization` Cargo feature.
    #[cfg(feature = "normalization")]
    #[must_use]
    pub fn with_normalization(mut self, form: crate::normalize::NormalizationForm) -> Self {
        self.normalize = Some(form);
        self
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
        let vocab = crate::tokenizer::parse_vocab_lines(&text);
        Self::from_vocab(vocab, unk_token)
    }
}

impl Tokenizer for WordPieceTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<TokenId>, TokenizerError> {
        #[cfg(feature = "normalization")]
        let normalized;
        #[cfg(feature = "normalization")]
        let text = if let Some(form) = self.normalize {
            normalized = crate::normalize::normalize(text, form);
            normalized.as_str()
        } else {
            text
        };

        let mut ids = Vec::new();
        for word in bert_basic(text, self.lowercase) {
            match self.tokenize_word(&word) {
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

    /// Decode `ids` back into a string, mirroring BERT's
    /// `convert_tokens_to_string`: continuation tokens (those prefixed with
    /// `##`) are concatenated directly, while each new word is preceded by a
    /// space. This preserves inter-word spacing that `encode` discards.
    ///
    /// # Errors
    /// Returns a [`TokenizerError`] if an id is missing from the inverse
    /// vocabulary.
    fn decode(&self, ids: &[TokenId]) -> Result<String, TokenizerError> {
        let mut out = String::new();
        for (i, &id) in ids.iter().enumerate() {
            let token = self
                .id_to_token
                .get(&id)
                .ok_or_else(|| TokenizerError::UnknownToken(id.to_string()))?;
            if let Some(stripped) = token.strip_prefix(CONTINUATION) {
                out.push_str(stripped);
            } else {
                if i > 0 {
                    out.push(' ');
                }
                out.push_str(token);
            }
        }
        Ok(out)
    }
}

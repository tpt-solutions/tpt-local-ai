//! Byte-Pair Encoding (BPE) tokenizer.

use alloc::collections::BTreeMap;
use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

use crate::error::TokenizerError;
use crate::tokenizer::{parse_vocab_lines, split_words, TokenId, Tokenizer};

/// The fallback token name used when a sub-word is not present in the vocab.
const UNK: &str = "<unk>";

/// A Byte-Pair Encoding tokenizer built from a vocabulary and an ordered list
/// of merge rules.
///
/// The merge list is applied greedily: at each step the adjacent symbol pair
/// with the lowest rank (earliest in the list) is merged, until no ranked pair
/// remains.
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    vocab: BTreeMap<String, TokenId>,
    id_to_token: BTreeMap<TokenId, String>,
    merge_ranks: BTreeMap<(String, String), usize>,
}

impl BpeTokenizer {
    /// Builds a tokenizer from a vocabulary (token string → id) and an ordered
    /// list of merge rules. The index of a rule in `merges` is its rank, so
    /// earlier rules win ties.
    #[must_use]
    pub fn from_vocab_merges(
        vocab: BTreeMap<String, TokenId>,
        merges: Vec<(String, String)>,
    ) -> Self {
        let merge_ranks = merges
            .into_iter()
            .enumerate()
            .map(|(rank, pair)| (pair, rank))
            .collect();
        let id_to_token = vocab
            .iter()
            .map(|(token, &id)| (id, token.clone()))
            .collect();
        Self {
            vocab,
            id_to_token,
            merge_ranks,
        }
    }

    /// Number of tokens in the vocabulary.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Applies the BPE merge algorithm to a single whitespace-delimited word,
    /// returning the resulting sub-word symbols (before id lookup).
    fn tokenize_word(&self, word: &str) -> Vec<String> {
        if word.is_empty() {
            return vec![];
        }
        let mut symbols: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        loop {
            let mut best_rank: Option<usize> = None;
            let mut best_idx: Option<usize> = None;
            for i in 0..symbols.len().saturating_sub(1) {
                let pair = (symbols[i].clone(), symbols[i + 1].clone());
                if let Some(&rank) = self.merge_ranks.get(&pair) {
                    if best_rank.map_or(true, |r| rank < r) {
                        best_rank = Some(rank);
                        best_idx = Some(i);
                    }
                }
            }
            let idx = match best_idx {
                Some(i) => i,
                None => break,
            };
            let merged = format!("{}{}", symbols[idx], symbols[idx + 1]);
            symbols[idx] = merged;
            symbols.remove(idx + 1);
        }
        symbols
    }

    /// Loads a GPT-2 style tokenizer from disk: `vocab_path` is one token per
    /// line (line number = id), `merges_path` is one `A B` pair per line
    /// (optionally prefixed by a single count header line).
    ///
    /// # Errors
    /// Returns [`TokenizerError::Io`] on a read failure, or
    /// [`TokenizerError::MalformedFile`] if a line cannot be parsed.
    #[cfg(feature = "std")]
    pub fn from_files(vocab_path: &str, merges_path: &str) -> Result<Self, TokenizerError> {
        let vocab_text = std::fs::read_to_string(vocab_path)?;
        let merges_text = std::fs::read_to_string(merges_path)?;

        let vocab = parse_vocab_lines(&vocab_text);

        let mut merges = Vec::new();
        for line in merges_text.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() != 2 {
                continue; // skip the optional count header line
            }
            merges.push((parts[0].to_string(), parts[1].to_string()));
        }

        Ok(Self::from_vocab_merges(vocab, merges))
    }
}

impl Tokenizer for BpeTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<TokenId>, TokenizerError> {
        let mut ids = Vec::new();
        for word in split_words(text) {
            for sub in self.tokenize_word(word) {
                match self.vocab.get(&sub) {
                    Some(&id) => ids.push(id),
                    None => {
                        if let Some(&unk) = self.vocab.get(UNK) {
                            ids.push(unk);
                        } else {
                            return Err(TokenizerError::UnknownToken(sub));
                        }
                    }
                }
            }
        }
        Ok(ids)
    }

    /// Decode `ids` back into a string by concatenating token text.
    ///
    /// Note: `encode` discards whitespace, so this is **lossy** for
    /// multi-word input — `decode(encode(s))` will not reproduce `s`'s spaces.
    /// (GPT-2-style tokenizers recover spacing via a `Ġ` marker, which this
    /// minimal implementation does not emit.)
    fn decode(&self, ids: &[TokenId]) -> Result<String, TokenizerError> {
        let mut out = String::new();
        for &id in ids {
            match self.id_to_token.get(&id) {
                Some(token) => out.push_str(token),
                None => return Err(TokenizerError::UnknownToken(id.to_string())),
            }
        }
        Ok(out)
    }
}

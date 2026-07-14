//! Byte-Pair Encoding (BPE) tokenizer.

use alloc::collections::BTreeMap;
use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

use crate::error::TokenizerError;
use crate::pretokenize::{decode_byte_level, encode_byte_level, gpt2_split};
use crate::tokenizer::{split_words, TokenId, Tokenizer};

/// The fallback token name used when a sub-word is not present in the vocab.
const UNK: &str = "<unk>";

/// A Byte-Pair Encoding tokenizer built from a vocabulary and an ordered list
/// of merge rules.
///
/// The merge list is applied greedily: at each step the adjacent symbol pair
/// with the lowest rank (earliest in the list) is merged, until no ranked pair
/// remains.
///
/// # Byte-level mode
///
/// Enable [`with_byte_level`](Self::with_byte_level) for GPT-2 / RoBERTa /
/// Llama style tokenizers. In this mode the input is pre-tokenized with a
/// GPT-2 word-splitting pass and each piece's raw bytes are mapped into the
/// reversible byte-level unicode alphabet before BPE is applied, so encoding
/// **never fails** on arbitrary UTF-8 and `decode(encode(s)) == s`.
///
/// # Special tokens
///
/// Register atomic markers (e.g. `<|endoftext|>`, `<s>`) with
/// [`with_special_tokens`](Self::with_special_tokens). They are matched
/// verbatim (longest-match-wins) before pre-tokenization and never split.
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    vocab: BTreeMap<String, TokenId>,
    id_to_token: BTreeMap<TokenId, String>,
    merge_ranks: BTreeMap<(String, String), usize>,
    byte_level: bool,
    special_tokens: BTreeMap<String, TokenId>,
    special_id_to_token: BTreeMap<TokenId, String>,
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
            byte_level: false,
            special_tokens: BTreeMap::new(),
            special_id_to_token: BTreeMap::new(),
        }
    }

    /// Enables GPT-2 byte-level pre-tokenization and byte-fallback encoding.
    #[must_use]
    pub fn with_byte_level(mut self) -> Self {
        self.byte_level = true;
        self
    }

    /// Registers special tokens (token string → id) that are matched atomically
    /// before pre-tokenization and never split by BPE.
    #[must_use]
    pub fn with_special_tokens(mut self, specials: BTreeMap<String, TokenId>) -> Self {
        self.special_id_to_token = specials.iter().map(|(t, &id)| (id, t.clone())).collect();
        self.special_tokens = specials;
        self
    }

    /// Number of tokens in the vocabulary (excluding special tokens that are
    /// not also present in the base vocabulary).
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Applies the BPE merge algorithm to a single pre-token, returning the
    /// resulting sub-word symbols (before id lookup).
    fn tokenize_word(&self, word: &str) -> Vec<String> {
        if word.is_empty() {
            return vec![];
        }
        let mut symbols: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        if symbols.len() < 2 {
            return symbols;
        }
        loop {
            let mut best_rank: Option<usize> = None;
            let mut best_idx: Option<usize> = None;
            for i in 0..symbols.len() - 1 {
                if let Some(&rank) = self.merge_ranks.get(&(
                    // Cheap: BTreeMap requires owned keys, but pre-tokens are
                    // short so these clones stay small.
                    symbols[i].clone(),
                    symbols[i + 1].clone(),
                )) {
                    if best_rank.map_or(true, |r| rank < r) {
                        best_rank = Some(rank);
                        best_idx = Some(i);
                    }
                }
            }
            let Some(idx) = best_idx else { break };
            let merged = format!("{}{}", symbols[idx], symbols[idx + 1]);
            symbols[idx] = merged;
            symbols.remove(idx + 1);
            if symbols.len() < 2 {
                break;
            }
        }
        symbols
    }

    /// Looks up a sub-word symbol, falling back to `<unk>` when configured.
    fn push_symbol(&self, sub: String, ids: &mut Vec<TokenId>) -> Result<(), TokenizerError> {
        match self.vocab.get(&sub) {
            Some(&id) => ids.push(id),
            None => match self.vocab.get(UNK) {
                Some(&unk) => ids.push(unk),
                None => return Err(TokenizerError::UnknownToken(sub)),
            },
        }
        Ok(())
    }

    /// Encodes a run of ordinary (non-special) text into `ids`.
    fn encode_text(&self, text: &str, ids: &mut Vec<TokenId>) -> Result<(), TokenizerError> {
        if self.byte_level {
            for piece in gpt2_split(text) {
                let encoded = encode_byte_level(&piece);
                for sub in self.tokenize_word(&encoded) {
                    self.push_symbol(sub, ids)?;
                }
            }
        } else {
            for word in split_words(text) {
                for sub in self.tokenize_word(word) {
                    self.push_symbol(sub, ids)?;
                }
            }
        }
        Ok(())
    }

    /// Splits `text` into `(special_id?, text_run)` segments, matching the
    /// longest registered special token starting at each position.
    fn split_specials<'a>(&self, text: &'a str) -> Vec<Segment<'a>> {
        if self.special_tokens.is_empty() {
            return vec![Segment::Text(text)];
        }
        // Longest tokens first so overlapping markers prefer the longest match.
        let mut specials: Vec<(&String, &TokenId)> = self.special_tokens.iter().collect();
        specials.sort_by_key(|s| core::cmp::Reverse(s.0.len()));

        let mut segments = Vec::new();
        let mut cursor = 0usize;
        let mut run_start = 0usize;
        while cursor < text.len() {
            let mut matched = None;
            for (tok, &id) in &specials {
                if text[cursor..].starts_with(tok.as_str()) {
                    matched = Some((tok.len(), id));
                    break;
                }
            }
            if let Some((len, id)) = matched {
                if run_start < cursor {
                    segments.push(Segment::Text(&text[run_start..cursor]));
                }
                segments.push(Segment::Special(id));
                cursor += len;
                run_start = cursor;
            } else {
                // Advance by one full char to keep byte offsets on boundaries.
                cursor += text[cursor..].chars().next().map_or(1, char::len_utf8);
            }
        }
        if run_start < text.len() {
            segments.push(Segment::Text(&text[run_start..]));
        }
        segments
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

        let vocab = crate::tokenizer::parse_vocab_lines(&vocab_text);

        let mut merges = Vec::new();
        for line in merges_text.lines() {
            if line.starts_with("#version") || line.trim().is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() != 2 {
                continue; // skip the optional count header line
            }
            merges.push((parts[0].to_string(), parts[1].to_string()));
        }

        Ok(Self::from_vocab_merges(vocab, merges))
    }
}

/// A run of input classified during special-token splitting.
enum Segment<'a> {
    /// Ordinary text to be pre-tokenized and BPE-encoded.
    Text(&'a str),
    /// A registered special token, emitted atomically.
    Special(TokenId),
}

impl Tokenizer for BpeTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<TokenId>, TokenizerError> {
        let mut ids = Vec::new();
        for segment in self.split_specials(text) {
            match segment {
                Segment::Special(id) => ids.push(id),
                Segment::Text(run) => self.encode_text(run, &mut ids)?,
            }
        }
        Ok(ids)
    }

    /// Decode `ids` back into a string.
    ///
    /// In byte-level mode this is exact: `decode(encode(s)) == s`. In plain
    /// mode `encode` discards whitespace, so decoding is **lossy** for
    /// multi-word input.
    fn decode(&self, ids: &[TokenId]) -> Result<String, TokenizerError> {
        let mut out = String::new();
        // Byte-level tokens must be decoded together (a single UTF-8 char may
        // span several tokens), so buffer them until interrupted by a special.
        let mut buf = String::new();
        let flush = |buf: &mut String, out: &mut String| -> Result<(), TokenizerError> {
            if buf.is_empty() {
                return Ok(());
            }
            if self.byte_level {
                let decoded = decode_byte_level(buf).ok_or_else(|| {
                    TokenizerError::MalformedFile("invalid byte-level sequence".to_string())
                })?;
                out.push_str(&decoded);
            } else {
                out.push_str(buf);
            }
            buf.clear();
            Ok(())
        };

        for &id in ids {
            if let Some(special) = self.special_id_to_token.get(&id) {
                flush(&mut buf, &mut out)?;
                out.push_str(special);
                continue;
            }
            match self.id_to_token.get(&id) {
                Some(token) => buf.push_str(token),
                None => return Err(TokenizerError::UnknownToken(id.to_string())),
            }
        }
        flush(&mut buf, &mut out)?;
        Ok(out)
    }
}

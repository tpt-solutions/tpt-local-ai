//! Loader for the modern unified Hugging Face `tokenizer.json` format.
//!
//! Most current Hub repositories ship a single `tokenizer.json` instead of the
//! legacy `vocab.txt` + `merges.txt` pair. This module parses that file with the
//! crate's internal [JSON parser](crate::json) (no `serde`) and produces the
//! appropriate concrete tokenizer.
//!
//! Supported `model.type` values: `"BPE"` and `"WordPiece"`. The loader also
//! honours `added_tokens` (registered as atomic special tokens), detects
//! GPT-2 byte-level BPE from the pre-tokenizer, and detects lowercasing from a
//! BERT normalizer.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::bpe::BpeTokenizer;
use crate::error::TokenizerError;
use crate::json::{self, JsonValue};
use crate::tokenizer::TokenId;
use crate::wordpiece::WordPieceTokenizer;

/// A tokenizer loaded from a `tokenizer.json`, tagged by its underlying scheme.
///
/// Both variants implement [`Tokenizer`](crate::Tokenizer); match on this to
/// recover the concrete type, or call [`LoadedTokenizer::as_tokenizer`] for a
/// trait object.
#[derive(Debug, Clone)]
pub enum LoadedTokenizer {
    /// A Byte-Pair Encoding tokenizer.
    Bpe(BpeTokenizer),
    /// A WordPiece tokenizer.
    WordPiece(WordPieceTokenizer),
}

impl LoadedTokenizer {
    /// Borrow the loaded tokenizer as a [`Tokenizer`](crate::Tokenizer) trait
    /// object.
    #[must_use]
    pub fn as_tokenizer(&self) -> &dyn crate::Tokenizer {
        match self {
            LoadedTokenizer::Bpe(t) => t,
            LoadedTokenizer::WordPiece(t) => t,
        }
    }
}

/// Parse a `tokenizer.json` document from a string.
///
/// # Errors
/// Returns [`TokenizerError::MalformedFile`] if the JSON is invalid, the model
/// type is unsupported, or a required field is missing.
pub fn from_tokenizer_json_str(text: &str) -> Result<LoadedTokenizer, TokenizerError> {
    let root = json::parse(text).map_err(TokenizerError::MalformedFile)?;
    let model = root
        .get("model")
        .ok_or_else(|| malformed("missing \"model\" object"))?;
    let model_type = model
        .get("type")
        .and_then(JsonValue::as_str)
        .ok_or_else(|| malformed("missing \"model.type\""))?;

    match model_type {
        "BPE" => load_bpe(&root, model).map(LoadedTokenizer::Bpe),
        "WordPiece" => load_wordpiece(&root, model).map(LoadedTokenizer::WordPiece),
        other => Err(malformed(&alloc::format!(
            "unsupported model.type {other:?} (only BPE and WordPiece are supported)"
        ))),
    }
}

/// Load a `tokenizer.json` from disk.
///
/// # Errors
/// Returns [`TokenizerError::Io`] on a read failure, or the same errors as
/// [`from_tokenizer_json_str`] on a parse failure.
#[cfg(feature = "std")]
pub fn from_tokenizer_json_file(path: &str) -> Result<LoadedTokenizer, TokenizerError> {
    let text = std::fs::read_to_string(path)?;
    from_tokenizer_json_str(&text)
}

fn malformed(msg: &str) -> TokenizerError {
    TokenizerError::MalformedFile(msg.to_string())
}

/// Extract a `token -> id` map from a JSON object.
fn parse_vocab(value: &JsonValue) -> Result<BTreeMap<String, TokenId>, TokenizerError> {
    let obj = value
        .as_object()
        .ok_or_else(|| malformed("\"model.vocab\" must be an object"))?;
    let mut vocab = BTreeMap::new();
    for (token, id) in obj {
        let id = id
            .as_u32()
            .ok_or_else(|| malformed("vocab id is not a non-negative integer"))?;
        vocab.insert(token.clone(), id);
    }
    Ok(vocab)
}

/// Collect `(content, id, is_special)` for every entry in a top-level
/// `added_tokens` array.
fn parse_added_tokens(root: &JsonValue) -> Vec<(String, TokenId, bool)> {
    let Some(added) = root.get("added_tokens").and_then(JsonValue::as_array) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in added {
        let (Some(content), Some(id)) = (
            entry.get("content").and_then(JsonValue::as_str),
            entry.get("id").and_then(JsonValue::as_u32),
        ) else {
            continue;
        };
        let special = entry
            .get("special")
            .is_some_and(|v| matches!(v, JsonValue::Bool(true)));
        out.push((content.to_string(), id, special));
    }
    out
}

/// Detect whether a pre-tokenizer (possibly a `Sequence`) uses GPT-2 byte-level
/// splitting.
fn detect_byte_level(root: &JsonValue) -> bool {
    fn contains_byte_level(v: &JsonValue) -> bool {
        if v.get("type").and_then(JsonValue::as_str) == Some("ByteLevel") {
            return true;
        }
        if let Some(list) = v.get("pretokenizers").and_then(JsonValue::as_array) {
            return list.iter().any(contains_byte_level);
        }
        false
    }
    root.get("pre_tokenizer").is_some_and(contains_byte_level)
}

/// Detect a lowercasing normalizer (BERT `lowercase: true`, or a `Lowercase`
/// normalizer, possibly inside a `Sequence`).
fn detect_lowercase(root: &JsonValue) -> bool {
    fn is_lower(v: &JsonValue) -> bool {
        match v.get("type").and_then(JsonValue::as_str) {
            Some("Lowercase") => return true,
            Some("BertNormalizer") => {
                if matches!(v.get("lowercase"), Some(JsonValue::Bool(true))) {
                    return true;
                }
            }
            _ => {}
        }
        if let Some(list) = v.get("normalizers").and_then(JsonValue::as_array) {
            return list.iter().any(is_lower);
        }
        false
    }
    root.get("normalizer").is_some_and(is_lower)
}

fn load_bpe(root: &JsonValue, model: &JsonValue) -> Result<BpeTokenizer, TokenizerError> {
    let mut vocab = parse_vocab(
        model
            .get("vocab")
            .ok_or_else(|| malformed("missing \"model.vocab\""))?,
    )?;

    let merges_val = model
        .get("merges")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| malformed("missing \"model.merges\" array"))?;
    let mut merges = Vec::with_capacity(merges_val.len());
    for entry in merges_val {
        let pair = match entry {
            // Newer format: ["a", "b"].
            JsonValue::Array(parts) if parts.len() == 2 => {
                let a = parts[0]
                    .as_str()
                    .ok_or_else(|| malformed("merge entry element is not a string"))?;
                let b = parts[1]
                    .as_str()
                    .ok_or_else(|| malformed("merge entry element is not a string"))?;
                (a.to_string(), b.to_string())
            }
            // Legacy format: "a b".
            JsonValue::String(s) => {
                let mut it = s.splitn(2, ' ');
                match (it.next(), it.next()) {
                    (Some(a), Some(b)) => (a.to_string(), b.to_string()),
                    _ => return Err(malformed("merge string is not a space-separated pair")),
                }
            }
            _ => return Err(malformed("unrecognised merge entry")),
        };
        merges.push(pair);
    }

    // Fold added tokens into the vocab and collect the special ones.
    let mut specials = BTreeMap::new();
    for (content, id, special) in parse_added_tokens(root) {
        vocab.entry(content.clone()).or_insert(id);
        if special {
            specials.insert(content, id);
        }
    }

    let mut tok = BpeTokenizer::from_vocab_merges(vocab, merges);
    if detect_byte_level(root) {
        tok = tok.with_byte_level();
    }
    if !specials.is_empty() {
        tok = tok.with_special_tokens(specials);
    }
    Ok(tok)
}

fn load_wordpiece(
    root: &JsonValue,
    model: &JsonValue,
) -> Result<WordPieceTokenizer, TokenizerError> {
    let mut vocab = parse_vocab(
        model
            .get("vocab")
            .ok_or_else(|| malformed("missing \"model.vocab\""))?,
    )?;

    // Fold in any added tokens so their ids are decodable.
    for (content, id, _special) in parse_added_tokens(root) {
        vocab.entry(content).or_insert(id);
    }

    let unk = model
        .get("unk_token")
        .and_then(JsonValue::as_str)
        .unwrap_or("[UNK]")
        .to_string();

    let mut tok = WordPieceTokenizer::from_vocab(vocab, &unk)?;
    if detect_lowercase(root) {
        tok = tok.with_lowercase();
    }
    Ok(tok)
}

//! WebAssembly bindings that expose [`tpt-jinja-chat`] chat-template rendering
//! and [`tpt-tokenizer-core`] BPE/WordPiece tokenization to a browser
//! playground.
//!
//! Both underlying crates are pure Rust with zero/minimal dependencies, so the
//! resulting `.wasm` is small and self-contained — the demo doubles as a
//! "zero-dependency" proof point. Build it with `wasm-pack` (see `README.md`).

use std::collections::BTreeMap;

use tpt_jinja_chat::{ChatTemplate, Context};
use tpt_tokenizer_core::{BpeTokenizer, Tokenizer, WordPieceTokenizer};
use wasm_bindgen::prelude::*;

/// Renders a Jinja chat template against a JSON context object.
///
/// `context_json` must be a JSON object, e.g.
/// `{"messages": [{"role": "user", "content": "Hi"}]}`.
#[wasm_bindgen]
pub fn render_chat_template(template: &str, context_json: &str) -> Result<String, JsError> {
    let tmpl = ChatTemplate::parse(template).map_err(js_err)?;
    let ctx = Context::from_json_str(context_json).map_err(js_err)?;
    tmpl.render(&ctx).map_err(js_err)
}

/// Encodes `text` with a BPE tokenizer built from the supplied vocab and merges.
///
/// * `vocab_json` — a JSON object mapping token string → integer id.
/// * `merges_text` — newline-separated `"a b"` merge rules (GPT-2 `merges.txt`
///   style); a leading `#version` comment line is ignored.
/// * `byte_level` — enable GPT-2 byte-level pre-tokenization / byte-fallback.
#[wasm_bindgen]
pub fn bpe_encode(
    text: &str,
    vocab_json: &str,
    merges_text: &str,
    byte_level: bool,
) -> Result<Vec<u32>, JsError> {
    let tok = build_bpe(vocab_json, merges_text, byte_level)?;
    tok.encode(text).map_err(js_err)
}

/// Decodes BPE token ids back into text, mirroring [`bpe_encode`]'s config.
#[wasm_bindgen]
pub fn bpe_decode(
    ids: Vec<u32>,
    vocab_json: &str,
    merges_text: &str,
    byte_level: bool,
) -> Result<String, JsError> {
    let tok = build_bpe(vocab_json, merges_text, byte_level)?;
    tok.decode(&ids).map_err(js_err)
}

/// Encodes `text` with a WordPiece (BERT-style) tokenizer.
///
/// * `vocab_json` — JSON object mapping token string → id (must contain
///   `unk_token`).
/// * `lowercase` — run the `*-uncased` lowercasing basic-tokenizer pass.
#[wasm_bindgen]
pub fn wordpiece_encode(
    text: &str,
    vocab_json: &str,
    unk_token: &str,
    lowercase: bool,
) -> Result<Vec<u32>, JsError> {
    let vocab = parse_vocab(vocab_json)?;
    let mut tok = WordPieceTokenizer::from_vocab(vocab, unk_token).map_err(js_err)?;
    if lowercase {
        tok = tok.with_lowercase();
    }
    tok.encode(text).map_err(js_err)
}

fn build_bpe(
    vocab_json: &str,
    merges_text: &str,
    byte_level: bool,
) -> Result<BpeTokenizer, JsError> {
    let vocab = parse_vocab(vocab_json)?;
    let merges = parse_merges(merges_text);
    let mut tok = BpeTokenizer::from_vocab_merges(vocab, merges);
    if byte_level {
        tok = tok.with_byte_level();
    }
    Ok(tok)
}

/// Parses a `{ "token": id }` JSON object into a vocabulary map.
fn parse_vocab(vocab_json: &str) -> Result<BTreeMap<String, u32>, JsError> {
    let raw: BTreeMap<String, serde_json::Value> =
        serde_json::from_str(vocab_json).map_err(|e| JsError::new(&format!("invalid vocab JSON: {e}")))?;
    let mut vocab = BTreeMap::new();
    for (token, id) in raw {
        let id = id
            .as_u64()
            .ok_or_else(|| JsError::new(&format!("vocab id for {token:?} is not an integer")))?;
        let id = u32::try_from(id)
            .map_err(|_| JsError::new(&format!("vocab id for {token:?} exceeds u32")))?;
        vocab.insert(token, id);
    }
    Ok(vocab)
}

/// Parses GPT-2 `merges.txt`-style text into ordered merge pairs.
fn parse_merges(merges_text: &str) -> Vec<(String, String)> {
    merges_text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .filter_map(|l| {
            let mut it = l.split_whitespace();
            match (it.next(), it.next()) {
                (Some(a), Some(b)) => Some((a.to_string(), b.to_string())),
                _ => None,
            }
        })
        .collect()
}

/// Converts any `Display` error into a JS-visible error.
fn js_err<E: std::fmt::Display>(e: E) -> JsError {
    JsError::new(&e.to_string())
}

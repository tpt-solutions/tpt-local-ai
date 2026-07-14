//! Unit tests for `tpt-tokenizer-core` against hand-built vocab/merge fixtures.

use std::collections::BTreeMap;

use tpt_tokenizer_core::encoding::{Padding, Truncation};
use tpt_tokenizer_core::{
    BpeTokenizer, EncodeConfig, LoadedTokenizer, Tokenizer, TokenizerExt, WordPieceTokenizer,
};

fn bpe_fixture() -> BpeTokenizer {
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
    BpeTokenizer::from_vocab_merges(vocab, merges)
}

#[test]
fn bpe_merges_to_single_token() {
    let tok = bpe_fixture();
    assert_eq!(tok.encode("low").unwrap(), vec![4]);
    assert_eq!(tok.encode("lo").unwrap(), vec![3]);
}

#[test]
fn bpe_round_trip() {
    let tok = bpe_fixture();
    let text = "low lo l o w";
    let ids = tok.encode(text).unwrap();
    assert_eq!(ids, vec![4, 3, 0, 1, 2]);
    assert_eq!(tok.decode(&ids).unwrap(), "lowlolow");
}

#[test]
fn bpe_unknown_without_unk_errors() {
    let tok = bpe_fixture();
    // 'x' is not in the vocab and there is no "<unk>" entry.
    assert!(tok.encode("x").is_err());
}

fn wordpiece_fixture() -> WordPieceTokenizer {
    let mut vocab = BTreeMap::new();
    vocab.insert("[UNK]".to_string(), 0u32);
    vocab.insert("un".to_string(), 1u32);
    vocab.insert("##aff".to_string(), 2u32);
    vocab.insert("##able".to_string(), 3u32);
    vocab.insert("##a".to_string(), 4u32);
    vocab.insert("##ble".to_string(), 5u32);
    WordPieceTokenizer::from_vocab(vocab, "[UNK]").unwrap()
}

#[test]
fn wordpiece_longest_match() {
    let tok = wordpiece_fixture();
    // "un" + "##aff" + "##able"
    assert_eq!(tok.encode("unaffable").unwrap(), vec![1, 2, 3]);
    assert_eq!(tok.decode(&[1, 2, 3]).unwrap(), "unaffable");
}

#[test]
fn wordpiece_unknown_word_is_unk() {
    let tok = wordpiece_fixture();
    assert_eq!(tok.encode("zzz").unwrap(), vec![0]);
    assert_eq!(tok.decode(&[0]).unwrap(), "[UNK]");
}

#[test]
fn wordpiece_whole_word_in_vocab() {
    let tok = wordpiece_fixture();
    // "un" is a whole-word token.
    assert_eq!(tok.encode("un").unwrap(), vec![1]);
}

/// Builds a small byte-level BPE that can encode any ASCII text via the
/// byte-level unicode alphabet, with a couple of merges.
fn byte_level_fixture() -> BpeTokenizer {
    let mut vocab = BTreeMap::new();
    // Byte-level unicode for printable ASCII maps to the char itself, and the
    // leading-space marker 'Ġ' is code point 0x120.
    for c in "helo wrdĠ".chars() {
        let id = vocab.len() as u32;
        vocab.insert(c.to_string(), id);
    }
    // Merges to build "he" and "Ġworld"-ish pieces.
    vocab.insert("he".to_string(), vocab.len() as u32);
    vocab.insert("Ġw".to_string(), vocab.len() as u32);
    let merges = vec![
        ("h".to_string(), "e".to_string()),
        ("Ġ".to_string(), "w".to_string()),
    ];
    BpeTokenizer::from_vocab_merges(vocab, merges).with_byte_level()
}

#[test]
fn byte_level_round_trip_is_lossless() {
    let tok = byte_level_fixture();
    let text = "hello world";
    let ids = tok.encode(text).unwrap();
    // Byte-level mode must reproduce the original exactly, spaces included.
    assert_eq!(tok.decode(&ids).unwrap(), text);
}

#[test]
fn byte_level_never_fails_on_unicode() {
    // Even without vocab entries for multi-byte chars, byte fallback + <unk>
    // guarantees encoding succeeds when <unk> exists.
    let mut vocab = BTreeMap::new();
    vocab.insert("<unk>".to_string(), 0u32);
    let tok = BpeTokenizer::from_vocab_merges(vocab, vec![]).with_byte_level();
    // Emoji is several UTF-8 bytes; each maps to a byte-level symbol.
    assert!(tok.encode("é🎉").is_ok());
}

#[test]
fn special_tokens_are_atomic() {
    let mut vocab = BTreeMap::new();
    vocab.insert("a".to_string(), 0u32);
    vocab.insert("b".to_string(), 1u32);
    let mut specials = BTreeMap::new();
    specials.insert("<|endoftext|>".to_string(), 100u32);
    let tok = BpeTokenizer::from_vocab_merges(vocab, vec![]).with_special_tokens(specials);

    let ids = tok.encode("a<|endoftext|>b").unwrap();
    assert_eq!(ids, vec![0, 100, 1]);
    // Special token id decodes back to its verbatim string, surrounded by text.
    assert_eq!(tok.decode(&[0, 100, 1]).unwrap(), "a<|endoftext|>b");
    assert_eq!(tok.decode(&[100]).unwrap(), "<|endoftext|>");
}

#[test]
fn wordpiece_splits_punctuation() {
    let mut vocab = BTreeMap::new();
    vocab.insert("[UNK]".to_string(), 0u32);
    vocab.insert("hi".to_string(), 1u32);
    vocab.insert("!".to_string(), 2u32);
    let tok = WordPieceTokenizer::from_vocab(vocab, "[UNK]").unwrap();
    // BERT basic tokenization isolates the trailing "!" as its own token.
    assert_eq!(tok.encode("hi!").unwrap(), vec![1, 2]);
}

#[test]
fn wordpiece_lowercase_option() {
    let mut vocab = BTreeMap::new();
    vocab.insert("[UNK]".to_string(), 0u32);
    vocab.insert("hi".to_string(), 1u32);
    let tok = WordPieceTokenizer::from_vocab(vocab, "[UNK]")
        .unwrap()
        .with_lowercase();
    assert_eq!(tok.encode("HI").unwrap(), vec![1]);
}

#[test]
fn empty_input_encodes_to_nothing() {
    let bpe = bpe_fixture();
    assert_eq!(bpe.encode("").unwrap(), Vec::<u32>::new());
    assert_eq!(bpe.encode("   ").unwrap(), Vec::<u32>::new());

    let wp = wordpiece_fixture();
    assert_eq!(wp.encode("").unwrap(), Vec::<u32>::new());
}

#[test]
fn bpe_from_files_loads_vocab_and_merges() {
    let dir = std::env::temp_dir().join(format!("tpt_tok_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let vocab_path = dir.join("vocab.txt");
    let merges_path = dir.join("merges.txt");
    // line index = id: l=0, o=1, w=2, lo=3, low=4
    std::fs::write(&vocab_path, "l\no\nw\nlo\nlow\n").unwrap();
    // A version header line must be ignored, then the merge rules.
    std::fs::write(&merges_path, "#version: 0.2\nl o\nlo w\n").unwrap();

    let tok = BpeTokenizer::from_files(vocab_path.to_str().unwrap(), merges_path.to_str().unwrap())
        .unwrap();
    assert_eq!(tok.encode("low").unwrap(), vec![4]);
    assert_eq!(tok.vocab_size(), 5);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn wordpiece_from_file_errors_on_missing_unk() {
    let dir = std::env::temp_dir().join(format!("tpt_tok_test_wp_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let vocab_path = dir.join("vocab.txt");
    std::fs::write(&vocab_path, "hello\nworld\n").unwrap();

    // "[UNK]" is not present, so construction must fail.
    let result = WordPieceTokenizer::from_file(vocab_path.to_str().unwrap(), "[UNK]");
    assert!(result.is_err());

    std::fs::remove_dir_all(&dir).ok();
}

// --- L121: heap-based BPE merge correctness on longer inputs ---

#[test]
fn bpe_heap_merges_long_word_correctly() {
    // Vocab/merges that fully reduce a repeated pattern; exercises the
    // linked-list + heap merge path over many symbols.
    let mut vocab = BTreeMap::new();
    vocab.insert("a".to_string(), 0u32);
    vocab.insert("b".to_string(), 1u32);
    vocab.insert("ab".to_string(), 2u32);
    vocab.insert("abab".to_string(), 3u32);
    let merges = vec![
        ("a".to_string(), "b".to_string()),
        ("ab".to_string(), "ab".to_string()),
    ];
    let tok = BpeTokenizer::from_vocab_merges(vocab, merges);
    // "abababab" -> ab ab ab ab -> abab abab
    assert_eq!(tok.encode("abababab").unwrap(), vec![3, 3]);
}

#[test]
fn bpe_heap_respects_merge_rank_order() {
    // Lower-rank merges must win: "l"+"o" (rank 0) before "o"+"w" (rank 1).
    let mut vocab = BTreeMap::new();
    for (i, t) in ["l", "o", "w", "lo", "low"].iter().enumerate() {
        vocab.insert(t.to_string(), i as u32);
    }
    let merges = vec![
        ("l".to_string(), "o".to_string()),
        ("lo".to_string(), "w".to_string()),
    ];
    let tok = BpeTokenizer::from_vocab_merges(vocab, merges);
    assert_eq!(tok.encode("low").unwrap(), vec![4]);
}

// --- L126: special-token insertion, padding, truncation, batch API ---

#[test]
fn encode_with_adds_bos_eos() {
    let tok = bpe_fixture();
    let cfg = EncodeConfig::new().with_bos(90).with_eos(91);
    let enc = tok.encode_with("low", &cfg).unwrap();
    assert_eq!(enc.ids, vec![90, 4, 91]);
    assert_eq!(enc.attention_mask, vec![1, 1, 1]);
}

#[test]
fn encode_with_fixed_padding() {
    let tok = bpe_fixture();
    let cfg = EncodeConfig::new().with_pad(0).padding(Padding::Fixed(4));
    let enc = tok.encode_with("low", &cfg).unwrap();
    // "low" -> [4], padded to length 4 with pad id 0.
    assert_eq!(enc.ids, vec![4, 0, 0, 0]);
    assert_eq!(enc.attention_mask, vec![1, 0, 0, 0]);
}

#[test]
fn encode_with_truncation_includes_specials() {
    let tok = bpe_fixture();
    let cfg = EncodeConfig::new()
        .with_bos(90)
        .with_eos(91)
        .truncation(Truncation::Fixed(3));
    // Content [4,3,0,1,2] truncated to room=1 (3 - 2 specials), then bos/eos.
    let enc = tok.encode_with("low lo l o w", &cfg).unwrap();
    assert_eq!(enc.ids, vec![90, 4, 91]);
}

#[test]
fn encode_batch_pads_to_longest() {
    let tok = bpe_fixture();
    let cfg = EncodeConfig::new().with_pad(0).padding(Padding::Longest);
    let batch = tok.encode_batch(&["low", "low lo l"], &cfg).unwrap();
    assert_eq!(batch[0].ids.len(), batch[1].ids.len());
    assert_eq!(batch[0].ids, vec![4, 0, 0]);
    assert_eq!(batch[0].attention_mask, vec![1, 0, 0]);
    assert_eq!(batch[1].ids, vec![4, 3, 0]);
    assert_eq!(batch[1].attention_mask, vec![1, 1, 1]);
}

#[test]
fn padding_without_pad_id_errors() {
    let tok = bpe_fixture();
    let cfg = EncodeConfig::new().padding(Padding::Fixed(4));
    assert!(tok.encode_with("low", &cfg).is_err());
}

// --- L125/177: tokenizer.json loader ---

#[test]
fn load_bpe_tokenizer_json() {
    let json = r#"{
        "model": {
            "type": "BPE",
            "vocab": { "l": 0, "o": 1, "w": 2, "lo": 3, "low": 4 },
            "merges": ["l o", ["lo", "w"]]
        },
        "added_tokens": [
            { "id": 100, "content": "<|endoftext|>", "special": true }
        ],
        "pre_tokenizer": { "type": "Whitespace" }
    }"#;
    let loaded = tpt_tokenizer_core::from_tokenizer_json_str(json).unwrap();
    let LoadedTokenizer::Bpe(tok) = loaded else {
        panic!("expected BPE");
    };
    assert_eq!(tok.encode("low").unwrap(), vec![4]);
    // The special added token is matched atomically.
    assert_eq!(tok.encode("low<|endoftext|>").unwrap(), vec![4, 100]);
}

#[test]
fn load_wordpiece_tokenizer_json_lowercase() {
    let json = r#"{
        "model": {
            "type": "WordPiece",
            "unk_token": "[UNK]",
            "vocab": { "[UNK]": 0, "hi": 1 }
        },
        "normalizer": { "type": "BertNormalizer", "lowercase": true }
    }"#;
    let loaded = tpt_tokenizer_core::from_tokenizer_json_str(json).unwrap();
    let LoadedTokenizer::WordPiece(tok) = loaded else {
        panic!("expected WordPiece");
    };
    assert_eq!(tok.encode("HI").unwrap(), vec![1]);
}

#[test]
fn load_tokenizer_json_rejects_unknown_model() {
    let json = r#"{ "model": { "type": "Unigram", "vocab": [] } }"#;
    assert!(tpt_tokenizer_core::from_tokenizer_json_str(json).is_err());
}

#[test]
fn load_tokenizer_json_rejects_invalid_json() {
    assert!(tpt_tokenizer_core::from_tokenizer_json_str("{ not json").is_err());
}

#[test]
fn json_parses_unicode_escapes() {
    use tpt_tokenizer_core::json::{self, JsonValue};
    let v = json::parse(r#"{ "k": "caf\u00e9 \ud83c\udf89" }"#).unwrap();
    assert_eq!(v.get("k").and_then(JsonValue::as_str), Some("café 🎉"));
}

// --- L127: Unicode normalization (feature-gated) ---

#[cfg(feature = "normalization")]
#[test]
fn nfc_normalization_before_encoding() {
    use tpt_tokenizer_core::NormalizationForm;
    // "é" as base 'e' + combining acute (NFD) should normalize to the single
    // precomposed "é" (NFC) and match the vocab token.
    let mut vocab = BTreeMap::new();
    vocab.insert("é".to_string(), 0u32);
    let tok =
        BpeTokenizer::from_vocab_merges(vocab, vec![]).with_normalization(NormalizationForm::Nfc);
    let decomposed = "e\u{0301}"; // e + combining acute accent
    assert_eq!(tok.encode(decomposed).unwrap(), vec![0]);
}

#[cfg(feature = "normalization")]
#[test]
fn nfkc_normalization_folds_compatibility_chars() {
    use tpt_tokenizer_core::{normalize, NormalizationForm};
    // Fullwidth 'Ａ' (U+FF21) folds to ASCII 'A' under NFKC.
    assert_eq!(normalize("\u{FF21}", NormalizationForm::Nfkc), "A");
}

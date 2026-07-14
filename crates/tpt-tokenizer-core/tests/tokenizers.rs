//! Unit tests for `tpt-tokenizer-core` against hand-built vocab/merge fixtures.

use std::collections::BTreeMap;

use tpt_tokenizer_core::{BpeTokenizer, Tokenizer, WordPieceTokenizer};

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

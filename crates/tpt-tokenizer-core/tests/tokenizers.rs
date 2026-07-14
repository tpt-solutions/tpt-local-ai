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

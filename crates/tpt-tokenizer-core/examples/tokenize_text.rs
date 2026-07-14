//! Tokenizes a small built-in sample with both BPE and WordPiece.
//!
//! Run with:
//!
//! ```sh
//! cargo run -p tpt-tokenizer-core --example tokenize_text
//! ```

use std::collections::BTreeMap;

use tpt_tokenizer_core::{BpeTokenizer, Tokenizer, WordPieceTokenizer};

fn main() {
    // --- BPE (GPT-2 style) ---
    let mut bpe_vocab = BTreeMap::new();
    for (i, t) in ["l", "o", "w", "lo", "low", "e", "r", "er", "<unk>"]
        .into_iter()
        .enumerate()
    {
        bpe_vocab.insert(t.to_string(), i as u32);
    }
    let bpe_merges = vec![
        ("l".to_string(), "o".to_string()),
        ("lo".to_string(), "w".to_string()),
        ("e".to_string(), "r".to_string()),
    ];
    let bpe = BpeTokenizer::from_vocab_merges(bpe_vocab, bpe_merges);

    let text = "low lower";
    let bpe_ids = bpe.encode(text).unwrap();
    println!("BPE   encode({text:?}) = {bpe_ids:?}");
    println!(
        "BPE   decode({bpe_ids:?}) = {:?}",
        bpe.decode(&bpe_ids).unwrap()
    );

    // --- WordPiece (BERT style) ---
    let mut wp_vocab = BTreeMap::new();
    for (i, t) in ["[UNK]", "un", "##aff", "##able", "play", "##ing"]
        .into_iter()
        .enumerate()
    {
        wp_vocab.insert(t.to_string(), i as u32);
    }
    let wp = WordPieceTokenizer::from_vocab(wp_vocab, "[UNK]").unwrap();

    let text2 = "unaffable playing";
    let wp_ids = wp.encode(text2).unwrap();
    println!("WP    encode({text2:?}) = {wp_ids:?}");
    println!(
        "WP    decode({wp_ids:?}) = {:?}",
        wp.decode(&wp_ids).unwrap()
    );
}

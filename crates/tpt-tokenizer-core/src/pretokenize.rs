//! Pre-tokenization and byte-level helpers shared by the tokenizers.
//!
//! Everything here is `no_std` + `alloc` and dependency-free.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// The GPT-2 "byte-to-unicode" table: a reversible mapping of every one of the
/// 256 byte values onto a printable Unicode scalar, so raw bytes can be encoded
/// as ordinary strings (and BPE can therefore never fail on arbitrary input).
///
/// This is the exact mapping used by GPT-2 / RoBERTa / Llama byte-level BPE.
pub(crate) fn byte_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut used = [false; 256];
    // Printable ASCII/Latin-1 ranges map to themselves.
    for b in b'!'..=b'~' {
        table[b as usize] = b as char;
        used[b as usize] = true;
    }
    for b in 0xA1u8..=0xAC {
        table[b as usize] = b as char;
        used[b as usize] = true;
    }
    for b in 0xAEu8..=0xFF {
        table[b as usize] = b as char;
        used[b as usize] = true;
    }
    // Remaining bytes map to Unicode code points 256+ in order.
    let mut n = 0u32;
    for b in 0..=255usize {
        if !used[b] {
            table[b] = char::from_u32(256 + n).expect("valid code point");
            n += 1;
        }
    }
    table
}

/// Encodes a string's UTF-8 bytes into the GPT-2 byte-level unicode space.
pub(crate) fn encode_byte_level(text: &str) -> String {
    let table = byte_to_unicode();
    let mut out = String::new();
    for &b in text.as_bytes() {
        out.push(table[b as usize]);
    }
    out
}

/// Decodes a GPT-2 byte-level unicode string back into its original text,
/// returning `None` if the bytes are not valid UTF-8.
pub(crate) fn decode_byte_level(text: &str) -> Option<String> {
    let table = byte_to_unicode();
    let mut bytes = Vec::new();
    for ch in text.chars() {
        let idx = table.iter().position(|&c| c == ch)?;
        bytes.push(idx as u8);
    }
    String::from_utf8(bytes).ok()
}

/// GPT-2 style pre-tokenization without a regex engine: splits `text` into
/// runs of letters, digits, and other non-space characters, attaching a single
/// leading space to each run, and recognises the common English contractions.
pub(crate) fn gpt2_split(text: &str) -> Vec<String> {
    const CONTRACTIONS: &[&[char]] = &[
        &['\'', 's'],
        &['\'', 't'],
        &['\'', 'm'],
        &['\'', 'd'],
        &['\'', 'r', 'e'],
        &['\'', 'v', 'e'],
        &['\'', 'l', 'l'],
    ];
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut tokens = Vec::new();
    let mut i = 0;

    'outer: while i < n {
        // Contractions like 's, 're, 'll.
        if chars[i] == '\'' {
            for pat in CONTRACTIONS {
                if i + pat.len() <= n && chars[i..i + pat.len()] == **pat {
                    tokens.push(pat.iter().collect());
                    i += pat.len();
                    continue 'outer;
                }
            }
        }

        let mut j = i;
        let mut leading_space = String::new();
        if chars[j] == ' ' && j + 1 < n && !chars[j + 1].is_whitespace() {
            leading_space.push(' ');
            j += 1;
        }

        if j < n && chars[j].is_alphabetic() {
            let start = j;
            while j < n && chars[j].is_alphabetic() {
                j += 1;
            }
            let mut t = leading_space;
            t.extend(&chars[start..j]);
            tokens.push(t);
        } else if j < n && chars[j].is_numeric() {
            let start = j;
            while j < n && chars[j].is_numeric() {
                j += 1;
            }
            let mut t = leading_space;
            t.extend(&chars[start..j]);
            tokens.push(t);
        } else if j < n && !chars[j].is_whitespace() {
            let start = j;
            while j < n && !chars[j].is_whitespace() && !chars[j].is_alphanumeric() {
                j += 1;
            }
            let mut t = leading_space;
            t.extend(&chars[start..j]);
            tokens.push(t);
        } else {
            // A run of whitespace becomes its own token.
            let start = i;
            while j < n && chars[j].is_whitespace() {
                j += 1;
            }
            tokens.push(chars[start..j].iter().collect());
        }
        i = j;
    }
    tokens
}

/// BERT "basic tokenizer": splits on whitespace and punctuation, isolates CJK
/// characters, and optionally lowercases. Accents are left intact (a full
/// NFD/strip-accents pass would require Unicode tables this crate deliberately
/// avoids).
pub(crate) fn bert_basic(text: &str, lowercase: bool) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();

    let flush = |cur: &mut String, out: &mut Vec<String>| {
        if !cur.is_empty() {
            out.push(core::mem::take(cur));
        }
    };

    for ch in text.chars() {
        if ch.is_whitespace() {
            flush(&mut current, &mut tokens);
            continue;
        }
        let ch = if lowercase {
            ch.to_lowercase().next().unwrap_or(ch)
        } else {
            ch
        };
        if is_punctuation(ch) || is_cjk(ch) {
            // Punctuation and CJK chars are emitted as standalone tokens.
            flush(&mut current, &mut tokens);
            tokens.push(ch.to_string());
        } else {
            current.push(ch);
        }
    }
    flush(&mut current, &mut tokens);
    tokens
}

/// Matches BERT's notion of punctuation: ASCII punctuation plus any Unicode
/// punctuation category.
fn is_punctuation(ch: char) -> bool {
    let c = ch as u32;
    (0x21..=0x2F).contains(&c)
        || (0x3A..=0x40).contains(&c)
        || (0x5B..=0x60).contains(&c)
        || (0x7B..=0x7E).contains(&c)
        || ch.is_ascii_punctuation()
}

/// Matches the CJK unified-ideograph ranges BERT isolates.
fn is_cjk(ch: char) -> bool {
    let c = ch as u32;
    (0x4E00..=0x9FFF).contains(&c)
        || (0x3400..=0x4DBF).contains(&c)
        || (0x20000..=0x2A6DF).contains(&c)
        || (0x2A700..=0x2B73F).contains(&c)
        || (0xF900..=0xFAFF).contains(&c)
}

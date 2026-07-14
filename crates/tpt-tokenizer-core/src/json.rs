//! A minimal, dependency-free JSON parser.
//!
//! Just enough of [RFC 8259](https://www.rfc-editor.org/rfc/rfc8259) to read a
//! Hugging Face `tokenizer.json` file: objects, arrays, strings (with `\uXXXX`
//! escapes and surrogate pairs), numbers, booleans, and null. It is `no_std` +
//! `alloc` and allocates owned [`String`]/[`Vec`] values.
//!
//! This exists so [`crate::loader`] can parse `tokenizer.json` without pulling
//! in `serde`/`serde_json`, keeping the crate's zero-dependency promise.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// A parsed JSON value.
#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    /// JSON `null`.
    Null,
    /// A boolean.
    Bool(bool),
    /// A number, kept as `f64` (sufficient for token ids up to 2^53).
    Number(f64),
    /// A string.
    String(String),
    /// An array.
    Array(Vec<JsonValue>),
    /// An object. Insertion order is not preserved (keys are sorted), which is
    /// fine for `tokenizer.json` where lookups are by name.
    Object(BTreeMap<String, JsonValue>),
}

impl JsonValue {
    /// Borrow this value as an object, if it is one.
    #[must_use]
    pub fn as_object(&self) -> Option<&BTreeMap<String, JsonValue>> {
        match self {
            JsonValue::Object(m) => Some(m),
            _ => None,
        }
    }

    /// Borrow this value as an array, if it is one.
    #[must_use]
    pub fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            JsonValue::Array(a) => Some(a),
            _ => None,
        }
    }

    /// Borrow this value as a string, if it is one.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// Read this value as an `f64`, if it is a number.
    #[must_use]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            JsonValue::Number(n) => Some(*n),
            _ => None,
        }
    }

    /// Read this value as a `u32` token id, if it is a non-negative integer.
    #[must_use]
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            // Avoid `f64::fract` (std-only); round-trip through `u32` instead so
            // this stays `no_std`.
            JsonValue::Number(n) if *n >= 0.0 && *n <= f64::from(u32::MAX) => {
                let truncated = *n as u32;
                (f64::from(truncated) == *n).then_some(truncated)
            }
            _ => None,
        }
    }

    /// Look up a key in this value, if it is an object.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        self.as_object().and_then(|m| m.get(key))
    }
}

/// Parse a JSON document from `input`.
///
/// # Errors
/// Returns a human-readable error message describing the first parse failure.
pub fn parse(input: &str) -> Result<JsonValue, String> {
    let mut p = Parser {
        chars: input.chars().collect(),
        pos: 0,
    };
    p.skip_ws();
    let value = p.parse_value()?;
    p.skip_ws();
    if p.pos != p.chars.len() {
        return Err(format_err("trailing data after JSON value", p.pos));
    }
    Ok(value)
}

fn format_err(msg: &str, pos: usize) -> String {
    let mut s = String::from(msg);
    s.push_str(" (at char ");
    s.push_str(&pos.to_string());
    s.push(')');
    s
}

struct Parser {
    chars: Vec<char>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.chars.get(self.pos).copied();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }

    fn skip_ws(&mut self) {
        while let Some(c) = self.peek() {
            if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn parse_value(&mut self) -> Result<JsonValue, String> {
        self.skip_ws();
        match self.peek() {
            Some('{') => self.parse_object(),
            Some('[') => self.parse_array(),
            Some('"') => Ok(JsonValue::String(self.parse_string()?)),
            Some('t') | Some('f') => self.parse_bool(),
            Some('n') => self.parse_null(),
            Some(c) if c == '-' || c.is_ascii_digit() => self.parse_number(),
            _ => Err(format_err("unexpected character", self.pos)),
        }
    }

    fn parse_object(&mut self) -> Result<JsonValue, String> {
        self.bump(); // '{'
        let mut map = BTreeMap::new();
        self.skip_ws();
        if self.peek() == Some('}') {
            self.bump();
            return Ok(JsonValue::Object(map));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some('"') {
                return Err(format_err("expected object key string", self.pos));
            }
            let key = self.parse_string()?;
            self.skip_ws();
            if self.bump() != Some(':') {
                return Err(format_err("expected ':' after object key", self.pos));
            }
            let value = self.parse_value()?;
            map.insert(key, value);
            self.skip_ws();
            match self.bump() {
                Some(',') => continue,
                Some('}') => break,
                _ => return Err(format_err("expected ',' or '}' in object", self.pos)),
            }
        }
        Ok(JsonValue::Object(map))
    }

    fn parse_array(&mut self) -> Result<JsonValue, String> {
        self.bump(); // '['
        let mut arr = Vec::new();
        self.skip_ws();
        if self.peek() == Some(']') {
            self.bump();
            return Ok(JsonValue::Array(arr));
        }
        loop {
            let value = self.parse_value()?;
            arr.push(value);
            self.skip_ws();
            match self.bump() {
                Some(',') => continue,
                Some(']') => break,
                _ => return Err(format_err("expected ',' or ']' in array", self.pos)),
            }
        }
        Ok(JsonValue::Array(arr))
    }

    fn parse_string(&mut self) -> Result<String, String> {
        self.bump(); // opening '"'
        let mut out = String::new();
        loop {
            match self.bump() {
                None => return Err(format_err("unterminated string", self.pos)),
                Some('"') => break,
                Some('\\') => {
                    let esc = self
                        .bump()
                        .ok_or_else(|| format_err("unterminated escape", self.pos))?;
                    match esc {
                        '"' => out.push('"'),
                        '\\' => out.push('\\'),
                        '/' => out.push('/'),
                        'b' => out.push('\u{0008}'),
                        'f' => out.push('\u{000C}'),
                        'n' => out.push('\n'),
                        'r' => out.push('\r'),
                        't' => out.push('\t'),
                        'u' => {
                            let cp = self.parse_hex4()?;
                            // Handle UTF-16 surrogate pairs.
                            if (0xD800..=0xDBFF).contains(&cp) {
                                if self.peek() == Some('\\') {
                                    self.bump();
                                    if self.bump() != Some('u') {
                                        return Err(format_err(
                                            "expected low surrogate escape",
                                            self.pos,
                                        ));
                                    }
                                    let low = self.parse_hex4()?;
                                    if !(0xDC00..=0xDFFF).contains(&low) {
                                        return Err(format_err("invalid low surrogate", self.pos));
                                    }
                                    let c = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                                    out.push(char::from_u32(c).ok_or_else(|| {
                                        format_err("invalid surrogate pair", self.pos)
                                    })?);
                                } else {
                                    return Err(format_err("lone high surrogate", self.pos));
                                }
                            } else {
                                out.push(
                                    char::from_u32(cp).ok_or_else(|| {
                                        format_err("invalid \\u escape", self.pos)
                                    })?,
                                );
                            }
                        }
                        _ => return Err(format_err("invalid escape character", self.pos)),
                    }
                }
                Some(c) => out.push(c),
            }
        }
        Ok(out)
    }

    fn parse_hex4(&mut self) -> Result<u32, String> {
        let mut value = 0u32;
        for _ in 0..4 {
            let c = self
                .bump()
                .ok_or_else(|| format_err("truncated \\u escape", self.pos))?;
            let digit = c
                .to_digit(16)
                .ok_or_else(|| format_err("invalid hex digit in \\u escape", self.pos))?;
            value = value * 16 + digit;
        }
        Ok(value)
    }

    fn parse_bool(&mut self) -> Result<JsonValue, String> {
        if self.matches("true") {
            Ok(JsonValue::Bool(true))
        } else if self.matches("false") {
            Ok(JsonValue::Bool(false))
        } else {
            Err(format_err("invalid literal", self.pos))
        }
    }

    fn parse_null(&mut self) -> Result<JsonValue, String> {
        if self.matches("null") {
            Ok(JsonValue::Null)
        } else {
            Err(format_err("invalid literal", self.pos))
        }
    }

    fn matches(&mut self, literal: &str) -> bool {
        let lit: Vec<char> = literal.chars().collect();
        if self.pos + lit.len() > self.chars.len() {
            return false;
        }
        if self.chars[self.pos..self.pos + lit.len()] == lit[..] {
            self.pos += lit.len();
            true
        } else {
            false
        }
    }

    fn parse_number(&mut self) -> Result<JsonValue, String> {
        let start = self.pos;
        if self.peek() == Some('-') {
            self.bump();
        }
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-' {
                self.bump();
            } else {
                break;
            }
        }
        let s: String = self.chars[start..self.pos].iter().collect();
        s.parse::<f64>()
            .map(JsonValue::Number)
            .map_err(|_| format_err("invalid number", start))
    }
}

//! Dynamic values used inside chat-template contexts.
//!
//! [`Value`] is a minimal, JSON-like value tree. It intentionally avoids any
//! external dependency so the whole crate stays `no_deps`. A small hand-rolled
//! JSON parser is provided so contexts can be built directly from JSON text.

use std::collections::HashMap;

use crate::error::TemplateError;

/// A dynamically-typed value that flows through a chat template.
///
/// Scalars are [`Value::Bool`], [`Value::Number`] and [`Value::String`]; compound
/// values are [`Value::Array`] and [`Value::Object`]. [`Value::Null`] marks the
/// absence of a value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// Absence of a value (maps to JSON `null`).
    Null,
    /// A boolean.
    Bool(bool),
    /// A floating-point number (JSON numbers are decoded as `f64`).
    Number(f64),
    /// A UTF-8 string.
    String(String),
    /// An ordered list of values.
    Array(Vec<Value>),
    /// A string-keyed map of values.
    Object(HashMap<String, Value>),
}

impl Value {
    /// Returns `true` if the value is "truthy" following Jinja/Python semantics.
    ///
    /// `Null`, `false`, `0`, `""`, `[]` and `{}` are falsy; everything else is truthy.
    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Null => false,
            Value::Bool(b) => *b,
            Value::Number(n) => *n != 0.0 && !n.is_nan(),
            Value::String(s) => !s.is_empty(),
            Value::Array(a) => !a.is_empty(),
            Value::Object(o) => !o.is_empty(),
        }
    }

    /// Render the value to a string for template output.
    pub fn to_rendered(&self) -> String {
        match self {
            Value::Null => String::new(),
            Value::Bool(b) => b.to_string(),
            Value::Number(n) => {
                if n.fract() == 0.0 && n.is_finite() {
                    format!("{:.0}", n)
                } else {
                    n.to_string()
                }
            }
            Value::String(s) => s.clone(),
            Value::Array(_) | Value::Object(_) => json_stringify(self),
        }
    }
}

/// Build a [`Context`] from structured values and render chat templates against it.
///
/// A `Context` is essentially a named map of [`Value`]s (the top-level template
/// variables such as `messages`, `bos_token`, ...).
#[derive(Debug, Clone, Default)]
pub struct Context {
    vars: HashMap<String, Value>,
}

impl Context {
    /// Create an empty context.
    pub fn new() -> Self {
        Context {
            vars: HashMap::new(),
        }
    }

    /// Insert a top-level variable into the context.
    pub fn insert(&mut self, key: impl Into<String>, value: Value) {
        self.vars.insert(key.into(), value);
    }

    /// Look up a top-level variable by name.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.vars.get(key)
    }

    /// Consume the context, returning the underlying variable map.
    pub fn into_vars(self) -> HashMap<String, Value> {
        self.vars
    }

    /// Build a context from an object [`Value`].
    pub fn from_value(value: Value) -> Result<Self, TemplateError> {
        match value {
            Value::Object(map) => Ok(Context { vars: map }),
            other => Err(TemplateError::type_error(format!(
                "context root must be an object, got {other:?}"
            ))),
        }
    }

    /// Parse a JSON string and build a context from the resulting object.
    pub fn from_json_str(input: &str) -> Result<Self, TemplateError> {
        let value = json::parse(input)?;
        Context::from_value(value)
    }
}

// ---------------------------------------------------------------------------
// Minimal JSON parser (no external dependencies)
// ---------------------------------------------------------------------------

mod json {
    use super::*;

    pub(super) fn parse(input: &str) -> Result<Value, TemplateError> {
        let bytes = input.as_bytes();
        let mut p = Parser { bytes, pos: 0 };
        p.skip_ws();
        let v = p.parse_value()?;
        p.skip_ws();
        if p.pos != p.bytes.len() {
            return Err(TemplateError::json(format!(
                "trailing characters at byte {}",
                p.pos
            )));
        }
        Ok(v)
    }

    struct Parser<'a> {
        bytes: &'a [u8],
        pos: usize,
    }

    impl<'a> Parser<'a> {
        fn skip_ws(&mut self) {
            while self.pos < self.bytes.len() {
                match self.bytes[self.pos] {
                    b' ' | b'\t' | b'\n' | b'\r' => self.pos += 1,
                    _ => break,
                }
            }
        }

        fn peek(&self) -> Option<u8> {
            self.bytes.get(self.pos).copied()
        }

        fn parse_value(&mut self) -> Result<Value, TemplateError> {
            self.skip_ws();
            match self.peek() {
                Some(b'{') => self.parse_object(),
                Some(b'[') => self.parse_array(),
                Some(b'"') => Ok(Value::String(self.parse_string()?)),
                Some(b't') | Some(b'f') => self.parse_bool(),
                Some(b'n') => self.parse_null(),
                Some(c) if c == b'-' || c.is_ascii_digit() => self.parse_number(),
                Some(c) => Err(TemplateError::json(format!(
                    "unexpected character '{}' at byte {}",
                    c as char, self.pos
                ))),
                None => Err(TemplateError::json("unexpected end of input")),
            }
        }

        fn parse_object(&mut self) -> Result<Value, TemplateError> {
            self.pos += 1; // consume '{'
            let mut map = HashMap::new();
            self.skip_ws();
            if self.peek() == Some(b'}') {
                self.pos += 1;
                return Ok(Value::Object(map));
            }
            loop {
                self.skip_ws();
                if self.peek() != Some(b'"') {
                    return Err(TemplateError::json(format!(
                        "expected object key string at byte {}",
                        self.pos
                    )));
                }
                let key = self.parse_string()?;
                self.skip_ws();
                if self.peek() != Some(b':') {
                    return Err(TemplateError::json(format!(
                        "expected ':' at byte {}",
                        self.pos
                    )));
                }
                self.pos += 1;
                let value = self.parse_value()?;
                map.insert(key, value);
                self.skip_ws();
                match self.peek() {
                    Some(b',') => {
                        self.pos += 1;
                        continue;
                    }
                    Some(b'}') => {
                        self.pos += 1;
                        break;
                    }
                    _ => {
                        return Err(TemplateError::json(format!(
                            "expected ',' or '}}' at byte {}",
                            self.pos
                        )))
                    }
                }
            }
            Ok(Value::Object(map))
        }

        fn parse_array(&mut self) -> Result<Value, TemplateError> {
            self.pos += 1; // consume '['
            let mut arr = Vec::new();
            self.skip_ws();
            if self.peek() == Some(b']') {
                self.pos += 1;
                return Ok(Value::Array(arr));
            }
            loop {
                let value = self.parse_value()?;
                arr.push(value);
                self.skip_ws();
                match self.peek() {
                    Some(b',') => {
                        self.pos += 1;
                        continue;
                    }
                    Some(b']') => {
                        self.pos += 1;
                        break;
                    }
                    _ => {
                        return Err(TemplateError::json(format!(
                            "expected ',' or ']' at byte {}",
                            self.pos
                        )))
                    }
                }
            }
            Ok(Value::Array(arr))
        }

        fn parse_string(&mut self) -> Result<String, TemplateError> {
            self.pos += 1; // consume opening quote
            let mut out = String::new();
            while self.pos < self.bytes.len() {
                let c = self.bytes[self.pos];
                self.pos += 1;
                match c {
                    b'"' => return Ok(out),
                    b'\\' => {
                        if self.pos >= self.bytes.len() {
                            return Err(TemplateError::json("unterminated escape"));
                        }
                        let e = self.bytes[self.pos];
                        self.pos += 1;
                        match e {
                            b'"' => out.push('"'),
                            b'\\' => out.push('\\'),
                            b'/' => out.push('/'),
                            b'b' => out.push('\u{0008}'),
                            b'f' => out.push('\u{000C}'),
                            b'n' => out.push('\n'),
                            b'r' => out.push('\r'),
                            b't' => out.push('\t'),
                            b'u' => {
                                let cp = self.parse_hex4()?;
                                if (0xD800..=0xDBFF).contains(&cp) {
                                    // high surrogate; expect low surrogate
                                    if self.pos + 1 < self.bytes.len()
                                        && self.bytes[self.pos] == b'\\'
                                        && self.bytes[self.pos + 1] == b'u'
                                    {
                                        self.pos += 2;
                                        let low = self.parse_hex4()?;
                                        if !(0xDC00..=0xDFFF).contains(&low) {
                                            return Err(TemplateError::json(
                                                "invalid low surrogate",
                                            ));
                                        }
                                        let c = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                                        match char::from_u32(c) {
                                            Some(ch) => out.push(ch),
                                            None => {
                                                return Err(TemplateError::json(
                                                    "invalid surrogate pair",
                                                ))
                                            }
                                        }
                                    } else {
                                        return Err(TemplateError::json("expected low surrogate"));
                                    }
                                } else {
                                    match char::from_u32(cp) {
                                        Some(ch) => out.push(ch),
                                        None => {
                                            return Err(TemplateError::json(
                                                "invalid unicode escape",
                                            ))
                                        }
                                    }
                                }
                            }
                            other => {
                                return Err(TemplateError::json(format!(
                                    "invalid escape '\\{}'",
                                    other as char
                                )))
                            }
                        }
                    }
                    _ => {
                        // Raw byte; collect a UTF-8 sequence.
                        if c < 0x80 {
                            out.push(c as char);
                        } else {
                            let start = self.pos - 1;
                            let len = if c >= 0xF0 {
                                4
                            } else if c >= 0xE0 {
                                3
                            } else {
                                2
                            };
                            if start + len > self.bytes.len() {
                                return Err(TemplateError::json("invalid utf-8"));
                            }
                            let slice = &self.bytes[start..start + len];
                            match std::str::from_utf8(slice) {
                                Ok(s) => out.push_str(s),
                                Err(_) => return Err(TemplateError::json("invalid utf-8")),
                            }
                            self.pos = start + len;
                        }
                    }
                }
            }
            Err(TemplateError::json("unterminated string"))
        }

        fn parse_hex4(&mut self) -> Result<u32, TemplateError> {
            if self.pos + 4 > self.bytes.len() {
                return Err(TemplateError::json("invalid \\u escape"));
            }
            let mut val: u32 = 0;
            for _ in 0..4 {
                let c = self.bytes[self.pos];
                self.pos += 1;
                let d = match c {
                    b'0'..=b'9' => (c - b'0') as u32,
                    b'a'..=b'f' => (c - b'a' + 10) as u32,
                    b'A'..=b'F' => (c - b'A' + 10) as u32,
                    _ => return Err(TemplateError::json("invalid \\u escape")),
                };
                val = val * 16 + d;
            }
            Ok(val)
        }

        fn parse_bool(&mut self) -> Result<Value, TemplateError> {
            if self.bytes[self.pos..].starts_with(b"true") {
                self.pos += 4;
                Ok(Value::Bool(true))
            } else if self.bytes[self.pos..].starts_with(b"false") {
                self.pos += 5;
                Ok(Value::Bool(false))
            } else {
                Err(TemplateError::json(format!(
                    "invalid literal at {}",
                    self.pos
                )))
            }
        }

        fn parse_null(&mut self) -> Result<Value, TemplateError> {
            if self.bytes[self.pos..].starts_with(b"null") {
                self.pos += 4;
                Ok(Value::Null)
            } else {
                Err(TemplateError::json(format!(
                    "invalid literal at {}",
                    self.pos
                )))
            }
        }

        fn parse_number(&mut self) -> Result<Value, TemplateError> {
            let start = self.pos;
            if self.peek() == Some(b'-') {
                self.pos += 1;
            }
            while let Some(c) = self.peek() {
                match c {
                    b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-' => self.pos += 1,
                    _ => break,
                }
            }
            let s = std::str::from_utf8(&self.bytes[start..self.pos])
                .map_err(|_| TemplateError::json("invalid number"))?;
            s.parse::<f64>()
                .map(Value::Number)
                .map_err(|_| TemplateError::json(format!("invalid number '{s}'")))
        }
    }
}

pub(crate) fn json_stringify(value: &Value) -> String {
    let mut out = String::new();
    write_json(value, None, 0, &mut out);
    out
}

/// Like [`json_stringify`] but pretty-printed with `indent` spaces per level.
pub(crate) fn json_stringify_pretty(value: &Value, indent: usize) -> String {
    let mut out = String::new();
    write_json(value, Some(indent), 0, &mut out);
    out
}

fn write_json_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            other => out.push(other),
        }
    }
    out.push('"');
}

fn write_json(value: &Value, indent: Option<usize>, level: usize, out: &mut String) {
    match value {
        Value::String(s) => write_json_string(s, out),
        Value::Number(n) => out.push_str(&format_number(*n)),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Null => out.push_str("null"),
        Value::Array(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                newline_indent(indent, level + 1, out);
                write_json(item, indent, level + 1, out);
            }
            newline_indent(indent, level, out);
            out.push(']');
        }
        Value::Object(map) => {
            if map.is_empty() {
                out.push_str("{}");
                return;
            }
            // Sort keys so serialization is deterministic and reproducible.
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                newline_indent(indent, level + 1, out);
                write_json_string(k, out);
                out.push(':');
                if indent.is_some() {
                    out.push(' ');
                }
                write_json(&map[*k], indent, level + 1, out);
            }
            newline_indent(indent, level, out);
            out.push('}');
        }
    }
}

fn newline_indent(indent: Option<usize>, level: usize, out: &mut String) {
    if let Some(n) = indent {
        out.push('\n');
        for _ in 0..(n * level) {
            out.push(' ');
        }
    }
}

fn format_number(n: f64) -> String {
    if n.fract() == 0.0 && n.is_finite() {
        format!("{n:.0}")
    } else {
        n.to_string()
    }
}

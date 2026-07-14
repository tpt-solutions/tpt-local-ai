//! Template scanning and parsing (statements + expressions).

use crate::ast::{Arg, Expr, Node, Op, SetTarget, Template};
use crate::error::TemplateError;

/// A flat, pre-scanned item from the raw template source.
#[derive(Debug)]
enum Item {
    Text(String),
    Output(String),
    Stmt(Stmt),
}

/// A single `{% ... %}` statement with its inner expression source preserved.
#[derive(Debug)]
enum Stmt {
    Set { name: String, expr: String },
    For { var: String, iterable: String },
    If { cond: String },
    Elif { cond: String },
    Else,
    EndFor,
    EndIf,
}

/// Scan the raw template into a flat list of [`Item`]s.
///
/// Supports Jinja whitespace control: a `-` immediately after an opening
/// delimiter (`{{-`, `{%-`) trims trailing whitespace from the preceding text,
/// and a `-` immediately before a closing delimiter (`-}}`, `-%}`) trims leading
/// whitespace from the following text.
fn scan(src: &str) -> Result<Vec<Item>, TemplateError> {
    let bytes = src.as_bytes();
    let mut items = Vec::new();
    let mut i = 0;
    let mut text = String::new();
    // When set, the next literal text pushed should have its leading whitespace
    // stripped (a `-%}` / `-}}` on the preceding tag requested this).
    let mut trim_next_text = false;

    while i < bytes.len() {
        if bytes[i] == b'{' && i + 1 < bytes.len() {
            let second = bytes[i + 1];
            if second == b'{' || second == b'%' || second == b'#' {
                let is_comment = second == b'#';
                let close = if second == b'{' {
                    "}}"
                } else if is_comment {
                    "#}"
                } else {
                    "%}"
                };
                // Optional `{{-` / `{%-` whitespace control marker.
                let mut k = i + 2;
                let trim_left = bytes.get(k) == Some(&b'-');
                if trim_left {
                    k += 1;
                }
                // Locate the closing delimiter, allowing a `-` immediately before
                // it (`-}}` / `-%}`) for right-side whitespace control.
                let (ci, cend, trim_right) = find_close(src, k, close)
                    .ok_or_else(|| TemplateError::parse(format!("expected '{close}'"), i))?;
                let inner = src[k..ci].to_string();
                i = cend;
                if trim_left {
                    while text.ends_with(|c: char| c.is_whitespace()) {
                        text.pop();
                    }
                }
                if !text.is_empty() {
                    items.push(Item::Text(std::mem::take(&mut text)));
                }
                if is_comment {
                    // Comment — discard its body; only whitespace control applies.
                } else if close == "}}" {
                    items.push(Item::Output(inner.trim().to_string()));
                } else {
                    items.push(Item::Stmt(parse_stmt(&inner, i)?));
                }
                trim_next_text = trim_right;
                continue;
            }
        }
        // Ordinary character. Copy a whole UTF-8 scalar, never a raw byte, so
        // multi-byte text (accents, CJK, emoji) is preserved verbatim.
        let ch = src[i..]
            .chars()
            .next()
            .expect("i is on a char boundary and in bounds");
        let ch_len = ch.len_utf8();
        if trim_next_text && ch.is_whitespace() {
            i += ch_len;
            continue;
        }
        trim_next_text = false;
        text.push(ch);
        i += ch_len;
    }
    if !text.is_empty() {
        items.push(Item::Text(text));
    }
    Ok(items)
}

/// Find the first occurrence of `close` at or after `start`, returning the index
/// of the closing delimiter's start, the index just past it, and whether the
/// close was whitespace-controlled (`-` immediately before it).
///
/// Scans by raw bytes so it can never panic on a non-char-boundary slice; the
/// delimiters (`}}`, `%}`, `#}`) are pure ASCII, and their bytes can never occur
/// inside a multi-byte UTF-8 sequence, so byte matching stays correct.
fn find_close(src: &str, start: usize, close: &str) -> Option<(usize, usize, bool)> {
    let b = src.as_bytes();
    let close_b = close.as_bytes();
    let mut i = start;
    while i + close_b.len() <= b.len() {
        if &b[i..i + close_b.len()] == close_b {
            let trim_right = i > 0 && b[i - 1] == b'-';
            return Some((i, i + close_b.len(), trim_right));
        }
        i += 1;
    }
    None
}

/// Parse a `{% ... %}` body into a [`Stmt`].
fn parse_stmt(raw: &str, pos: usize) -> Result<Stmt, TemplateError> {
    let trimmed = raw.trim();
    let mut parts = trimmed.split_whitespace();
    let keyword = parts
        .next()
        .ok_or_else(|| TemplateError::parse("empty statement", pos))?;
    match keyword {
        "set" => {
            let rest = &trimmed[keyword.len()..];
            let eq = rest
                .find('=')
                .ok_or_else(|| TemplateError::parse("expected '=' in set", pos))?;
            let name = rest[..eq].trim().to_string();
            let expr = rest[eq + 1..].trim().to_string();
            if name.is_empty() || expr.is_empty() {
                return Err(TemplateError::parse("malformed set statement", pos));
            }
            Ok(Stmt::Set { name, expr })
        }
        "for" => {
            let rest = trimmed[keyword.len()..].trim();
            let in_pos = rest
                .find(" in ")
                .ok_or_else(|| TemplateError::parse("expected 'in' in for", pos))?;
            let var = rest[..in_pos].trim().to_string();
            let iterable = rest[in_pos + 4..].trim().to_string();
            if var.is_empty() || iterable.is_empty() {
                return Err(TemplateError::parse("malformed for statement", pos));
            }
            Ok(Stmt::For { var, iterable })
        }
        "if" => Ok(Stmt::If {
            cond: trimmed[keyword.len()..].trim().to_string(),
        }),
        "elif" => Ok(Stmt::Elif {
            cond: trimmed[keyword.len()..].trim().to_string(),
        }),
        "else" => Ok(Stmt::Else),
        "endif" => Ok(Stmt::EndIf),
        "endfor" => Ok(Stmt::EndFor),
        other => Err(TemplateError::parse(
            format!("unknown statement '{other}'"),
            pos,
        )),
    }
}

/// Parse the full template into a [`Template`] AST.
pub(crate) fn parse(src: &str) -> Result<Template, TemplateError> {
    let items = scan(src)?;
    let mut iter = items.into_iter().peekable();
    let nodes = parse_block(&mut iter, None)?;
    if iter.peek().is_some() {
        return Err(TemplateError::parse(
            "unexpected token after template end",
            0,
        ));
    }
    Ok(Template { nodes })
}

/// Parse a block of nodes.
///
/// `terminator` is the statement keyword that ends this block (`"endfor"` or
/// `"endif"`), or `None` at the top level. The terminator (when present and
/// matched) is **consumed**; `elif`/`else` are left in the stream so the
/// surrounding `if` chain can consume them.
fn parse_block(
    iter: &mut std::iter::Peekable<std::vec::IntoIter<Item>>,
    terminator: Option<&str>,
) -> Result<Vec<Node>, TemplateError> {
    let mut nodes = Vec::new();
    loop {
        match iter.peek() {
            None => {
                if let Some(t) = terminator {
                    return Err(TemplateError::parse(format!("missing '{t}'"), 0));
                }
                return Ok(nodes);
            }
            Some(Item::Stmt(Stmt::EndFor)) => {
                if terminator == Some("endfor") {
                    // Stop before the terminator; the `for` handler consumes it.
                    return Ok(nodes);
                }
                return Err(TemplateError::parse("unexpected 'endfor'", 0));
            }
            Some(Item::Stmt(Stmt::EndIf)) => {
                if terminator == Some("endif") {
                    // Stop before the terminator; `parse_if_chain` consumes it.
                    return Ok(nodes);
                }
                return Err(TemplateError::parse("unexpected 'endif'", 0));
            }
            Some(Item::Stmt(Stmt::Elif { .. })) | Some(Item::Stmt(Stmt::Else)) => {
                if terminator == Some("endif") {
                    // Belongs to the enclosing if-chain; leave for parse_if_chain.
                    return Ok(nodes);
                }
                return Err(TemplateError::parse("unexpected 'elif'/'else'", 0));
            }
            Some(Item::Stmt(Stmt::If { .. })) => {
                let (branches, else_body) = parse_if_chain(iter)?;
                nodes.push(Node::If {
                    branches,
                    else_body,
                });
            }
            Some(Item::Stmt(Stmt::For { .. })) => {
                let Item::Stmt(Stmt::For { var, iterable }) = iter.next().unwrap() else {
                    unreachable!()
                };
                let iterable = parse_expression(&iterable, 0)?;
                // Support tuple unpacking targets: `for k, v in ...`.
                let targets: Vec<String> = var
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                if targets.is_empty() {
                    return Err(TemplateError::parse("empty for target", 0));
                }
                let body = parse_block(iter, Some("endfor"))?;
                // `parse_block` stops *before* the matching `endfor`; consume it.
                match iter.peek() {
                    Some(Item::Stmt(Stmt::EndFor)) => {
                        let _ = iter.next();
                    }
                    _ => return Err(TemplateError::parse("missing 'endfor'", 0)),
                }
                nodes.push(Node::For {
                    targets,
                    iterable,
                    body,
                });
            }
            Some(Item::Stmt(Stmt::Set { .. })) => {
                let Item::Stmt(Stmt::Set { name, expr }) = iter.next().unwrap() else {
                    unreachable!()
                };
                let value = parse_expression(&expr, 0)?;
                let target = match name.split_once('.') {
                    Some((base, attr)) => SetTarget::Attr {
                        base: base.trim().to_string(),
                        attr: attr.trim().to_string(),
                    },
                    None => SetTarget::Var(name),
                };
                nodes.push(Node::Set { target, value });
            }
            Some(Item::Output(e)) => {
                let e = e.clone();
                iter.next();
                nodes.push(Node::Output(parse_expression(&e, 0)?));
            }
            Some(Item::Text(t)) => {
                let t = t.clone();
                iter.next();
                nodes.push(Node::Text(t));
            }
        }
    }
}

/// Parse a full `if` / `elif` / `else` / `endif` chain (the first item must be
/// the opening `if`).
#[allow(clippy::type_complexity)]
fn parse_if_chain(
    iter: &mut std::iter::Peekable<std::vec::IntoIter<Item>>,
) -> Result<(Vec<(Expr, Vec<Node>)>, Vec<Node>), TemplateError> {
    let mut branches: Vec<(Expr, Vec<Node>)> = Vec::new();
    let mut else_body: Vec<Node> = Vec::new();

    // Opening `if`.
    match iter.next() {
        Some(Item::Stmt(Stmt::If { cond })) => {
            let expr = parse_expression(&cond, 0)?;
            let body = parse_block(iter, Some("endif"))?;
            branches.push((expr, body));
        }
        other => {
            return Err(TemplateError::parse(
                format!("expected 'if', found {other:?}"),
                0,
            ))
        }
    }

    loop {
        match iter.peek() {
            Some(Item::Stmt(Stmt::Elif { cond })) => {
                let cond = cond.clone();
                iter.next();
                let expr = parse_expression(&cond, 0)?;
                let body = parse_block(iter, Some("endif"))?;
                branches.push((expr, body));
            }
            Some(Item::Stmt(Stmt::Else)) => {
                iter.next();
                else_body = parse_block(iter, Some("endif"))?;
            }
            Some(Item::Stmt(Stmt::EndIf)) => {
                iter.next();
                break;
            }
            _ => return Err(TemplateError::parse("unexpected token inside if block", 0)),
        }
    }
    Ok((branches, else_body))
}

// ---------------------------------------------------------------------------
// Expression tokenizer + recursive-descent parser
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Ident(String),
    Str(String),
    Num(f64),
    Bool(bool),
    Op(Op),
    LParen,
    RParen,
    LBracket,
    RBracket,
    Dot,
    Comma,
    Minus,
    Assign,
    Not,
    Pipe,
    Is,
    None,
}
fn tokenize_expr(src: &str) -> Result<Vec<Tok>, TemplateError> {
    let bytes = src.as_bytes();
    let mut toks = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        match c {
            b' ' | b'\t' | b'\n' | b'\r' => {
                i += 1;
            }
            b'(' => {
                toks.push(Tok::LParen);
                i += 1;
            }
            b')' => {
                toks.push(Tok::RParen);
                i += 1;
            }
            b'[' => {
                toks.push(Tok::LBracket);
                i += 1;
            }
            b']' => {
                toks.push(Tok::RBracket);
                i += 1;
            }
            b'.' => {
                toks.push(Tok::Dot);
                i += 1;
            }
            b',' => {
                toks.push(Tok::Comma);
                i += 1;
            }
            b'+' => {
                toks.push(Tok::Op(Op::Add));
                i += 1;
            }
            b'*' => {
                toks.push(Tok::Op(Op::Mul));
                i += 1;
            }
            b'~' => {
                toks.push(Tok::Op(Op::Concat));
                i += 1;
            }
            b'|' => {
                toks.push(Tok::Pipe);
                i += 1;
            }
            b'/' => {
                toks.push(Tok::Op(Op::Div));
                i += 1;
            }
            b'=' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'=' {
                    toks.push(Tok::Op(Op::Eq));
                    i += 2;
                } else {
                    toks.push(Tok::Assign);
                    i += 1;
                }
            }
            b'!' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'=' {
                    toks.push(Tok::Op(Op::Ne));
                    i += 2;
                } else {
                    return Err(TemplateError::parse("unexpected '!'", i));
                }
            }
            b'<' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'=' {
                    toks.push(Tok::Op(Op::Le));
                    i += 2;
                } else {
                    toks.push(Tok::Op(Op::Lt));
                    i += 1;
                }
            }
            b'>' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'=' {
                    toks.push(Tok::Op(Op::Ge));
                    i += 2;
                } else {
                    toks.push(Tok::Op(Op::Gt));
                    i += 1;
                }
            }
            b'-' => {
                toks.push(Tok::Minus);
                i += 1;
            }
            b'"' | b'\'' => {
                let quote = c;
                i += 1;
                let start = i;
                let mut s = String::new();
                while i < bytes.len() && bytes[i] != quote {
                    if bytes[i] == b'\\' && i + 1 < bytes.len() {
                        let esc = bytes[i + 1];
                        match esc {
                            b'n' => s.push('\n'),
                            b't' => s.push('\t'),
                            b'r' => s.push('\r'),
                            b'\\' => s.push('\\'),
                            b'\'' => s.push('\''),
                            b'"' => s.push('"'),
                            other => s.push(other as char),
                        }
                        i += 2;
                    } else {
                        let ch = src[i..]
                            .chars()
                            .next()
                            .expect("i is on a char boundary and in bounds");
                        s.push(ch);
                        i += ch.len_utf8();
                    }
                }
                if i >= bytes.len() {
                    return Err(TemplateError::parse("unterminated string literal", start));
                }
                i += 1; // closing quote
                toks.push(Tok::Str(s));
            }
            c if c.is_ascii_digit()
                || (c == b'-' && i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit()) =>
            {
                let start = i;
                i += 1;
                while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.') {
                    i += 1;
                }
                let s = &src[start..i];
                let n: f64 = s
                    .parse()
                    .map_err(|_| TemplateError::parse(format!("invalid number '{s}'"), start))?;
                toks.push(Tok::Num(n));
            }
            c if c.is_ascii_alphabetic() || c == b'_' => {
                let start = i;
                while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                    i += 1;
                }
                let word = &src[start..i];
                match word {
                    "and" => toks.push(Tok::Op(Op::And)),
                    "or" => toks.push(Tok::Op(Op::Or)),
                    "in" => toks.push(Tok::Op(Op::In)),
                    "not" => toks.push(Tok::Not),
                    "is" => toks.push(Tok::Is),
                    "true" | "True" => toks.push(Tok::Bool(true)),
                    "false" | "False" => toks.push(Tok::Bool(false)),
                    "none" | "None" | "null" => toks.push(Tok::None),
                    _ => toks.push(Tok::Ident(word.to_string())),
                }
            }
            other => {
                return Err(TemplateError::parse(
                    format!("unexpected character '{}' in expression", other as char),
                    i,
                ))
            }
        }
    }
    Ok(toks)
}

struct ExprParser {
    toks: Vec<Tok>,
    pos: usize,
}

impl ExprParser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn next(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    fn expect(&mut self, tok: &Tok) -> Result<(), TemplateError> {
        match self.next() {
            Some(t) if &t == tok => Ok(()),
            other => Err(TemplateError::parse(
                format!("expected {tok:?}, found {other:?}"),
                self.pos,
            )),
        }
    }

    fn parse(&mut self) -> Result<Expr, TemplateError> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<Expr, TemplateError> {
        let mut left = self.parse_and()?;
        while matches!(self.peek(), Some(Tok::Op(Op::Or))) {
            self.next();
            let right = self.parse_and()?;
            left = Expr::Bin(Op::Or, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Expr, TemplateError> {
        let mut left = self.parse_not()?;
        while matches!(self.peek(), Some(Tok::Op(Op::And))) {
            self.next();
            let right = self.parse_not()?;
            left = Expr::Bin(Op::And, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_not(&mut self) -> Result<Expr, TemplateError> {
        if matches!(self.peek(), Some(Tok::Not)) {
            self.next();
            let e = self.parse_not()?;
            return Ok(Expr::Not(Box::new(e)));
        }
        self.parse_test()
    }

    /// Parse an `is` / `is not` test: `expr is [not] name(args?)`.
    fn parse_test(&mut self) -> Result<Expr, TemplateError> {
        let left = self.parse_cmp()?;
        if matches!(self.peek(), Some(Tok::Is)) {
            self.next();
            let negated = if matches!(self.peek(), Some(Tok::Not)) {
                self.next();
                true
            } else {
                false
            };
            let name = match self.next() {
                Some(Tok::Ident(n)) => n,
                Some(Tok::None) => "none".to_string(),
                other => {
                    return Err(TemplateError::parse(
                        format!("expected test name after 'is', found {other:?}"),
                        self.pos,
                    ))
                }
            };
            let args = if matches!(self.peek(), Some(Tok::LParen)) {
                self.next();
                self.parse_args()?
            } else {
                Vec::new()
            };
            return Ok(Expr::Test(Box::new(left), name, args, negated));
        }
        Ok(left)
    }

    fn parse_cmp(&mut self) -> Result<Expr, TemplateError> {
        let mut left = self.parse_add()?;
        loop {
            // `not in` is a two-token operator; detect it before the rest.
            if matches!(self.peek(), Some(Tok::Not))
                && matches!(self.toks.get(self.pos + 1), Some(Tok::Op(Op::In)))
            {
                self.next();
                self.next();
                let right = self.parse_add()?;
                left = Expr::Bin(Op::NotIn, Box::new(left), Box::new(right));
                continue;
            }
            let op = match self.peek() {
                Some(Tok::Op(Op::Eq)) => Op::Eq,
                Some(Tok::Op(Op::Ne)) => Op::Ne,
                Some(Tok::Op(Op::Lt)) => Op::Lt,
                Some(Tok::Op(Op::Gt)) => Op::Gt,
                Some(Tok::Op(Op::Le)) => Op::Le,
                Some(Tok::Op(Op::Ge)) => Op::Ge,
                Some(Tok::Op(Op::In)) => Op::In,
                _ => break,
            };
            self.next();
            let right = self.parse_add()?;
            left = Expr::Bin(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_add(&mut self) -> Result<Expr, TemplateError> {
        let mut left = self.parse_mul()?;
        loop {
            match self.peek() {
                Some(Tok::Op(Op::Add)) => {
                    self.next();
                    let right = self.parse_mul()?;
                    left = Expr::Bin(Op::Add, Box::new(left), Box::new(right));
                }
                Some(Tok::Minus) => {
                    self.next();
                    let right = self.parse_mul()?;
                    left = Expr::Bin(Op::Sub, Box::new(left), Box::new(right));
                }
                Some(Tok::Op(Op::Concat)) => {
                    self.next();
                    let right = self.parse_mul()?;
                    left = Expr::Bin(Op::Concat, Box::new(left), Box::new(right));
                }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_mul(&mut self) -> Result<Expr, TemplateError> {
        let mut left = self.parse_unary()?;
        loop {
            let op = match self.peek() {
                Some(Tok::Op(Op::Mul)) => Op::Mul,
                Some(Tok::Op(Op::Div)) => Op::Div,
                _ => break,
            };
            self.next();
            let right = self.parse_unary()?;
            left = Expr::Bin(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Expr, TemplateError> {
        if matches!(self.peek(), Some(Tok::Minus)) {
            self.next();
            let e = self.parse_unary()?;
            return Ok(Expr::Neg(Box::new(e)));
        }
        self.parse_filter()
    }

    /// Parse a chain of `| filter(args?)` applications over a postfix expression.
    fn parse_filter(&mut self) -> Result<Expr, TemplateError> {
        let mut e = self.parse_postfix()?;
        while matches!(self.peek(), Some(Tok::Pipe)) {
            self.next();
            let name = match self.next() {
                Some(Tok::Ident(n)) => n,
                other => {
                    return Err(TemplateError::parse(
                        format!("expected filter name after '|', found {other:?}"),
                        self.pos,
                    ))
                }
            };
            let args = if matches!(self.peek(), Some(Tok::LParen)) {
                self.next();
                self.parse_args()?
            } else {
                Vec::new()
            };
            e = Expr::Filter(Box::new(e), name, args);
        }
        Ok(e)
    }

    /// Parse a comma-separated argument list up to and including the closing
    /// `)`. Each argument is either positional (`expr`) or a keyword (`k=expr`).
    fn parse_args(&mut self) -> Result<Vec<Arg>, TemplateError> {
        let mut args = Vec::new();
        if matches!(self.peek(), Some(Tok::RParen)) {
            self.next();
            return Ok(args);
        }
        loop {
            // Keyword argument: `ident = expr` (but not `==`).
            let name = if let Some(Tok::Ident(n)) = self.peek().cloned() {
                if matches!(self.toks.get(self.pos + 1), Some(Tok::Assign)) {
                    self.next(); // ident
                    self.next(); // '='
                    Some(n)
                } else {
                    None
                }
            } else {
                None
            };
            let value = self.parse()?;
            args.push(Arg { name, value });
            match self.next() {
                Some(Tok::Comma) => continue,
                Some(Tok::RParen) => break,
                other => {
                    return Err(TemplateError::parse(
                        format!("expected ',' or ')' in argument list, found {other:?}"),
                        self.pos,
                    ))
                }
            }
        }
        Ok(args)
    }

    fn parse_postfix(&mut self) -> Result<Expr, TemplateError> {
        let mut e = self.parse_primary()?;
        loop {
            match self.peek() {
                Some(Tok::Dot) => {
                    self.next();
                    match self.next() {
                        Some(Tok::Ident(name)) => {
                            e = Expr::Get(Box::new(e), name);
                        }
                        other => {
                            return Err(TemplateError::parse(
                                format!("expected field name after '.', found {other:?}"),
                                self.pos,
                            ))
                        }
                    }
                }
                Some(Tok::LBracket) => {
                    self.next();
                    let idx = self.parse()?;
                    self.expect(&Tok::RBracket)?;
                    e = Expr::Index(Box::new(e), Box::new(idx));
                }
                Some(Tok::LParen) => {
                    self.next();
                    let args = self.parse_args()?;
                    e = Expr::Call(Box::new(e), args);
                }
                _ => break,
            }
        }
        Ok(e)
    }

    fn parse_primary(&mut self) -> Result<Expr, TemplateError> {
        match self.next() {
            Some(Tok::Ident(name)) => Ok(Expr::Var(name)),
            Some(Tok::Str(s)) => Ok(Expr::Str(s)),
            Some(Tok::Num(n)) => Ok(Expr::Num(n)),
            Some(Tok::Bool(b)) => Ok(Expr::Bool(b)),
            Some(Tok::None) => Ok(Expr::None),
            Some(Tok::LParen) => {
                let e = self.parse()?;
                self.expect(&Tok::RParen)?;
                Ok(e)
            }
            Some(Tok::LBracket) => {
                let mut items = Vec::new();
                if matches!(self.peek(), Some(Tok::RBracket)) {
                    self.next();
                    return Ok(Expr::List(items));
                }
                loop {
                    items.push(self.parse()?);
                    match self.next() {
                        Some(Tok::Comma) => {
                            // Allow a trailing comma before `]`.
                            if matches!(self.peek(), Some(Tok::RBracket)) {
                                self.next();
                                break;
                            }
                            continue;
                        }
                        Some(Tok::RBracket) => break,
                        other => {
                            return Err(TemplateError::parse(
                                format!("expected ',' or ']' in list, found {other:?}"),
                                self.pos,
                            ))
                        }
                    }
                }
                Ok(Expr::List(items))
            }
            other => Err(TemplateError::parse(
                format!("unexpected token in expression: {other:?}"),
                self.pos,
            )),
        }
    }
}

/// Tokenize and parse an expression string into an [`Expr`].
fn parse_expression(src: &str, _pos: usize) -> Result<Expr, TemplateError> {
    let toks = tokenize_expr(src)?;
    let mut p = ExprParser { toks, pos: 0 };
    let expr = p.parse()?;
    if p.pos != p.toks.len() {
        return Err(TemplateError::parse(
            format!("trailing tokens in expression at {}", p.pos),
            p.pos,
        ));
    }
    Ok(expr)
}

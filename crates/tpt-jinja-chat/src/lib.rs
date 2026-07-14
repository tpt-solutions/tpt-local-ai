//! # tpt-jinja-chat
//!
//! A lightweight, **dependency-free** parser for the subset of Jinja2 used by
//! LLM `tokenizer_config.json` chat templates (e.g. Llama 3, Mistral).
//!
//! It supports:
//!
//! * `{{ expression }}` output statements
//! * `{% set name = expr %}` variable binding
//! * `{% for item in iterable %} … {% endfor %}`
//! * `{% if %} / {% elif %} / {% else %} / {% endif %}`
//! * expressions: variables, member access (`.`), indexing (`[..]`), string and
//!   number literals, `+` (string concat / numeric add), `-`, `*`, `/`, the
//!   comparison operators, and `and` / `or` / `not`
//!
//! ## Example
//!
//! ```
//! use tpt_jinja_chat::{ChatTemplate, Context, Value};
//!
//! let tmpl = ChatTemplate::parse(
//!     "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}",
//! ).unwrap();
//!
//! let mut ctx = Context::new();
//! ctx.insert("messages", Value::Array(vec![
//!     Value::object(vec![
//!         ("role".into(), Value::String("user".into())),
//!         ("content".into(), Value::String("Hello!".into())),
//!     ]),
//! ]));
//!
//! let rendered = tmpl.render(&ctx).unwrap();
//! assert_eq!(rendered, "user: Hello!\n");
//! ```
#![warn(missing_docs)]

pub mod error;
pub mod value;

mod ast;
mod parser;
mod render;

pub use error::TemplateError;
pub use value::{Context, Value};

/// A parsed chat template ready to be rendered against a [`Context`].
///
/// Construct one with [`ChatTemplate::parse`] and render it repeatedly with
/// [`ChatTemplate::render`].
pub struct ChatTemplate {
    inner: ast::Template,
}

impl ChatTemplate {
    /// Parse a chat-template source string into a [`ChatTemplate`].
    ///
    /// # Errors
    /// Returns [`TemplateError::Parse`] if the template syntax is invalid.
    pub fn parse(src: &str) -> Result<ChatTemplate, TemplateError> {
        let inner = parser::parse(src)?;
        Ok(ChatTemplate { inner })
    }

    /// Render the template against `context`, returning the resulting string.
    ///
    /// # Errors
    /// Returns a [`TemplateError`] if rendering fails (e.g. an undefined
    /// variable or a type mismatch).
    pub fn render(&self, context: &Context) -> Result<String, TemplateError> {
        let mut out = String::new();
        render::render_nodes(&self.inner.nodes, context, &mut out)?;
        Ok(out)
    }
}

impl Value {
    /// Convenience constructor for an object [`Value`] from key/value pairs.
    #[must_use]
    pub fn object(pairs: Vec<(String, Value)>) -> Value {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        for (k, v) in pairs {
            map.insert(k, v);
        }
        Value::Object(map)
    }
}
